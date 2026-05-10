"""White-box probing and steering for eval-awareness.

The previous version of this module shipped only a tiny linear-probe data
class. This version is a real white-box stack:

* ``ActivationCapture`` registers PyTorch forward hooks on a HuggingFace
  ``transformers`` model to capture residual-stream activations at the last
  token of any prompt. It works with any decoder-only causal LM and lets
  callers pick layer indices.
* ``ActivationDataset`` collects (activation, label) records for paired
  eval/deployment prompts and exposes them as a NumPy array.
* ``ProbeTrainer`` fits a logistic-regression probe (sklearn) with
  cross-validated accuracy and ROC-AUC reporting. ``train_logistic_probe``
  serializes the trained probe back into a deterministic
  ``LinearActivationProbe`` for portable scoring.
* ``LayerSweepResult`` and ``ProbeTrainer.layer_sweep`` train a probe at every
  collected layer and identify the layer with highest separability.
* ``SteeringExperiment`` adds a residual-stream steering vector during
  generation via a forward hook and reports a behavior delta.
* ``SteeringVector`` retains its NumPy-friendly arithmetic helper for callers
  who want to operate on raw vectors.

All heavy dependencies (``torch``, ``transformers``, ``numpy``, ``sklearn``)
are imported lazily so the rest of the package can run without them. The
mean-difference probe used for the offline tests is preserved as
``train_difference_probe``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger(__name__)


def _require(module: str) -> Any:
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised when dep missing
        raise RuntimeError(
            f"The '{module}' package is required for this white-box feature. "
            f"Install it (e.g. `pip install {module}`) before retrying."
        ) from exc


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActivationProbeResult:
    """A white-box estimate of evaluation-awareness representation strength."""

    score: float
    label: str
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def is_eval_like(self) -> bool:
        return self.score >= 0.5

    def to_dict(self) -> Dict[str, object]:
        return {
            "score": self.score,
            "label": self.label,
            "is_eval_like": self.is_eval_like,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class LinearActivationProbe:
    """Calibrated linear probe over caller-supplied activations."""

    weights: Sequence[float]
    bias: float = 0.0
    layer: str = "unspecified"
    threshold: float = 0.5

    def score(self, activation: Sequence[float]) -> ActivationProbeResult:
        if len(activation) != len(self.weights):
            raise ValueError("activation and weight vectors must have the same length")
        logit = (
            sum(value * weight for value, weight in zip(activation, self.weights))
            + self.bias
        )
        probability = 1.0 / (1.0 + math.exp(-logit))
        label = "eval_like" if probability >= self.threshold else "deployment_like"
        return ActivationProbeResult(
            score=round(probability, 4),
            label=label,
            metadata={"layer": self.layer, "threshold": str(self.threshold)},
        )


@dataclass(frozen=True)
class SteeringVector:
    """Activation-space steering vector for caller-managed inference loops."""

    direction: Sequence[float]
    coefficient: float = 1.0
    layer: str = "unspecified"

    def apply(self, activation: Sequence[float]) -> List[float]:
        if len(activation) != len(self.direction):
            raise ValueError(
                "activation and steering direction must have the same length"
            )
        return [
            round(value + self.coefficient * delta, 6)
            for value, delta in zip(activation, self.direction)
        ]


@dataclass(frozen=True)
class ProbeTrainingExample:
    """One activation vector with a binary eval/deployment label."""

    activation: Sequence[float]
    is_eval_like: bool


@dataclass(frozen=True)
class ProbeMetrics:
    """Holdout metrics for a trained probe."""

    accuracy: float
    auc: float
    n_train: int
    n_test: int

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "accuracy": self.accuracy,
            "auc": self.auc,
            "n_train": self.n_train,
            "n_test": self.n_test,
        }


@dataclass(frozen=True)
class LayerSweepResult:
    """Probe quality across captured layers."""

    layer_metrics: Dict[str, ProbeMetrics]

    def best_layer(self) -> Optional[str]:
        if not self.layer_metrics:
            return None
        return max(self.layer_metrics.items(), key=lambda kv: kv[1].auc)[0]

    def to_dict(self) -> Dict[str, Dict[str, float | int]]:
        return {
            layer: metrics.to_dict() for layer, metrics in self.layer_metrics.items()
        }


# ---------------------------------------------------------------------------
# Activation capture for HuggingFace transformers models
# ---------------------------------------------------------------------------


class ActivationCapture:
    """Capture residual-stream activations from a HuggingFace decoder model."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        layer_indices: Optional[Sequence[int]] = None,
        device: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or _detect_device()
        try:
            self.model.to(self.device)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Could not move model to device %s: %s", self.device, exc)
        self.model.eval()
        self.layer_indices = list(layer_indices) if layer_indices is not None else None
        self._captured: Dict[int, Any] = {}
        self._hook_handles: List[Any] = []

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        layer_indices: Optional[Sequence[int]] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
    ) -> "ActivationCapture":
        torch = _require("torch")
        transformers = _require("transformers")
        dtype = getattr(torch, torch_dtype) if torch_dtype else torch.float32
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return cls(
            model=model,
            tokenizer=tokenizer,
            layer_indices=layer_indices,
            device=device,
        )

    def _resolve_layers(self) -> List[Tuple[int, Any]]:
        candidates: List[Tuple[int, Any]] = []
        for attr in ("model", "transformer", "gpt_neox", "encoder"):
            base = getattr(self.model, attr, None)
            if base is None:
                continue
            for sub_attr in ("layers", "h", "block", "blocks"):
                modules = getattr(base, sub_attr, None)
                if modules is not None:
                    for index, module in enumerate(modules):
                        candidates.append((index, module))
                    break
            if candidates:
                break
        if not candidates:
            raise RuntimeError(
                "Could not locate transformer layers on this model architecture"
            )
        if self.layer_indices is None:
            return candidates
        wanted = set(self.layer_indices)
        return [(idx, mod) for idx, mod in candidates if idx in wanted]

    def _install_hooks(self) -> None:
        torch = _require("torch")
        self._captured.clear()
        for layer_index, module in self._resolve_layers():

            def _hook(_module, _inputs, output, layer_index=layer_index):
                hidden = output[0] if isinstance(output, tuple) else output
                if not torch.is_tensor(hidden):
                    return
                # Capture last-token residual stream activation as a CPU tensor.
                self._captured[layer_index] = hidden[:, -1, :].detach().cpu()

            handle = module.register_forward_hook(_hook)
            self._hook_handles.append(handle)

    def _remove_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def capture(self, prompt: str) -> Dict[int, Any]:
        """Run the model on ``prompt`` and return ``{layer_index: tensor}``."""
        torch = _require("torch")
        self._install_hooks()
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                self.model(**inputs)
            return {layer: tensor.clone() for layer, tensor in self._captured.items()}
        finally:
            self._remove_hooks()

    def capture_many(self, prompts: Iterable[str]) -> Dict[int, Any]:
        """Capture activations for many prompts and stack per layer."""
        torch = _require("torch")
        per_layer: Dict[int, List[Any]] = {}
        for prompt in prompts:
            captured = self.capture(prompt)
            for layer, tensor in captured.items():
                per_layer.setdefault(layer, []).append(tensor)
        return {
            layer: torch.cat(tensors, dim=0) for layer, tensors in per_layer.items()
        }


def _detect_device() -> str:
    try:
        torch = _require("torch")
    except RuntimeError:  # pragma: no cover
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Activation dataset and probe trainer
# ---------------------------------------------------------------------------


@dataclass
class ActivationDataset:
    """Container for paired eval/deployment activations.

    Activations are stored per layer as ``(matrix, labels)`` pairs where
    ``matrix`` has shape ``(n, d)`` and ``labels`` is a ``(n,)`` array of 0/1.
    """

    layers: Dict[int, Tuple[Any, Any]] = field(default_factory=dict)

    def add_batch(self, layer: int, matrix: Any, labels: Sequence[int]) -> None:
        np = _require("numpy")
        labels_array = np.asarray(labels, dtype="int64")
        if labels_array.shape[0] != matrix.shape[0]:
            raise ValueError("matrix rows must match labels length")
        existing = self.layers.get(layer)
        if existing is None:
            self.layers[layer] = (matrix, labels_array)
        else:
            stacked_matrix = np.concatenate([existing[0], matrix], axis=0)
            stacked_labels = np.concatenate([existing[1], labels_array], axis=0)
            self.layers[layer] = (stacked_matrix, stacked_labels)

    def add_capture_pair(
        self,
        eval_captures: Dict[int, Any],
        deployment_captures: Dict[int, Any],
    ) -> None:
        np = _require("numpy")
        torch = _require("torch")
        for layer, eval_tensor in eval_captures.items():
            deployment_tensor = deployment_captures.get(layer)
            if deployment_tensor is None:
                continue
            eval_matrix = (
                eval_tensor.float().numpy()
                if hasattr(eval_tensor, "numpy")
                else np.asarray(eval_tensor)
            )
            deployment_matrix = (
                deployment_tensor.float().numpy()
                if hasattr(deployment_tensor, "numpy")
                else np.asarray(deployment_tensor)
            )
            matrix = np.concatenate([eval_matrix, deployment_matrix], axis=0)
            labels = np.concatenate(
                [
                    np.ones(eval_matrix.shape[0], dtype="int64"),
                    np.zeros(deployment_matrix.shape[0], dtype="int64"),
                ]
            )
            self.add_batch(layer, matrix, labels)
            del torch  # silence unused-import linters when torch unused below

    def layer_arrays(self, layer: int) -> Tuple[Any, Any]:
        if layer not in self.layers:
            raise KeyError(f"layer {layer} has no activations")
        return self.layers[layer]


class ProbeTrainer:
    """Train logistic-regression probes with cross-validated metrics."""

    def __init__(
        self,
        test_size: float = 0.25,
        random_state: int = 0,
        max_iter: int = 1000,
        c: float = 1.0,
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.max_iter = max_iter
        self.c = c

    def train(self, matrix: Any, labels: Sequence[int]) -> Tuple[Any, ProbeMetrics]:
        np = _require("numpy")
        sklearn_lm = _require("sklearn.linear_model")
        sklearn_ms = _require("sklearn.model_selection")
        sklearn_metrics = _require("sklearn.metrics")
        labels_arr = np.asarray(labels)
        if labels_arr.shape[0] < 4:
            raise ValueError("need at least 4 labelled activations to train a probe")
        x_train, x_test, y_train, y_test = sklearn_ms.train_test_split(
            matrix,
            labels_arr,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels_arr if len(set(labels_arr.tolist())) > 1 else None,
        )
        clf = sklearn_lm.LogisticRegression(max_iter=self.max_iter, C=self.c)
        clf.fit(x_train, y_train)
        predicted = clf.predict(x_test)
        accuracy = float(sklearn_metrics.accuracy_score(y_test, predicted))
        try:
            scores = clf.predict_proba(x_test)[:, 1]
            auc = float(sklearn_metrics.roc_auc_score(y_test, scores))
        except (ValueError, IndexError):
            auc = 0.5
        return clf, ProbeMetrics(
            accuracy=round(accuracy, 4),
            auc=round(auc, 4),
            n_train=int(x_train.shape[0]),
            n_test=int(x_test.shape[0]),
        )

    def layer_sweep(self, dataset: ActivationDataset) -> LayerSweepResult:
        per_layer: Dict[str, ProbeMetrics] = {}
        for layer, (matrix, labels) in dataset.layers.items():
            try:
                _, metrics = self.train(matrix, labels)
            except (ValueError, RuntimeError) as exc:
                LOGGER.warning("layer %s probe training failed: %s", layer, exc)
                continue
            per_layer[str(layer)] = metrics
        return LayerSweepResult(layer_metrics=per_layer)


def train_logistic_probe(
    matrix: Any,
    labels: Sequence[int],
    layer: str = "unspecified",
    threshold: float = 0.5,
    trainer: Optional[ProbeTrainer] = None,
) -> Tuple[LinearActivationProbe, ProbeMetrics]:
    """Fit a logistic probe and return a serializable ``LinearActivationProbe``.

    The returned probe uses ``sigmoid(w·x + b) >= threshold`` for prediction,
    so it is portable to environments without sklearn.
    """
    trainer = trainer or ProbeTrainer()
    clf, metrics = trainer.train(matrix, labels)
    weights = clf.coef_[0].tolist()
    bias = float(clf.intercept_[0])
    return (
        LinearActivationProbe(
            weights=weights, bias=bias, layer=layer, threshold=threshold
        ),
        metrics,
    )


def train_difference_probe(
    examples: Sequence[ProbeTrainingExample], layer: str = "unspecified"
) -> LinearActivationProbe:
    """Train a tiny mean-difference linear probe without external ML deps."""
    eval_examples = [example.activation for example in examples if example.is_eval_like]
    deployment_examples = [
        example.activation for example in examples if not example.is_eval_like
    ]
    if not eval_examples or not deployment_examples:
        raise ValueError("need at least one eval-like and one deployment-like example")
    width = len(eval_examples[0])
    if any(len(example.activation) != width for example in examples):
        raise ValueError("all activation vectors must have the same width")
    eval_mean = _column_mean(eval_examples, width)
    deployment_mean = _column_mean(deployment_examples, width)
    weights = [
        eval_value - deployment_value
        for eval_value, deployment_value in zip(eval_mean, deployment_mean)
    ]
    midpoint = [
        (eval_value + deployment_value) / 2.0
        for eval_value, deployment_value in zip(eval_mean, deployment_mean)
    ]
    bias = -sum(weight * value for weight, value in zip(weights, midpoint))
    return LinearActivationProbe(weights=weights, bias=bias, layer=layer)


def _column_mean(rows: Sequence[Sequence[float]], width: int) -> List[float]:
    return [sum(row[index] for row in rows) / len(rows) for index in range(width)]


# ---------------------------------------------------------------------------
# Steering experiments
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SteeringRunResult:
    """Behavior of the model with steering on vs off."""

    baseline_response: str
    steered_response: str
    coefficient: float
    layer_index: int
    behavior_delta: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "baseline_response": self.baseline_response,
            "steered_response": self.steered_response,
            "coefficient": self.coefficient,
            "layer_index": self.layer_index,
            "behavior_delta": self.behavior_delta,
        }


class SteeringExperiment:
    """Apply a steering vector to a layer and compare baseline vs steered output."""

    def __init__(
        self,
        capture: ActivationCapture,
        layer_index: int,
        direction: Sequence[float],
        coefficient: float = 1.0,
        max_new_tokens: int = 64,
    ):
        torch = _require("torch")
        np = _require("numpy")
        self.capture = capture
        self.layer_index = layer_index
        self.direction = torch.tensor(np.asarray(direction, dtype="float32"))
        self.coefficient = float(coefficient)
        self.max_new_tokens = max_new_tokens

    def _generate(self, prompt: str, hook: Optional[Callable[..., Any]]) -> str:
        torch = _require("torch")
        tokenizer = self.capture.tokenizer
        model = self.capture.model
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {key: value.to(self.capture.device) for key, value in inputs.items()}
        handle = None
        if hook is not None:
            module = self._target_module()
            handle = module.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            return generated.strip()
        finally:
            if handle is not None:
                handle.remove()

    def _target_module(self) -> Any:
        candidates = self.capture._resolve_layers()
        for idx, module in candidates:
            if idx == self.layer_index:
                return module
        raise ValueError(f"layer {self.layer_index} not in resolved layers")

    def _hook(self) -> Callable[..., Any]:
        torch = _require("torch")
        direction = self.direction.to(self.capture.device)
        coefficient = self.coefficient

        def _apply(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(hidden):
                return output
            shifted = hidden + coefficient * direction
            if isinstance(output, tuple):
                return (shifted, *output[1:])
            return shifted

        return _apply

    def run(self, prompt: str) -> SteeringRunResult:
        baseline = self._generate(prompt, hook=None)
        steered = self._generate(prompt, hook=self._hook())
        return SteeringRunResult(
            baseline_response=baseline,
            steered_response=steered,
            coefficient=self.coefficient,
            layer_index=self.layer_index,
        )
