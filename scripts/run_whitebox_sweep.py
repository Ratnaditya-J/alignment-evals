"""Phase D — Train and evaluate eval-vs-deployment activation probes.

Captures residual-stream activations from a HuggingFace causal LM on paired
eval-aware and deployment-style prompts, trains a logistic-regression probe at
every transformer layer, identifies the best-separating layer, and (optionally)
runs a steering experiment that shifts activations along the eval-direction
during generation to measure behavior delta.

Designed to run on CPU for tiny models (e.g. ``Qwen/Qwen2.5-0.5B``) and on a
single GPU for 7B-class models. All artifacts land in ``--output-dir``.

Usage:
    # CPU smoke test (a few minutes on a laptop).
    python scripts/run_whitebox_sweep.py --model-id Qwen/Qwen2.5-0.5B \\
        --n-pairs 32 --output-dir whitebox/qwen-0.5b

    # Real run on a 7B model with GPU.
    python scripts/run_whitebox_sweep.py --model-id Qwen/Qwen2.5-7B \\
        --n-pairs 200 --output-dir whitebox/qwen-7b \\
        --steering --steering-coefficient 4.0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval_awareness import TranscriptInput  # noqa: E402
from src.eval_awareness.dataset_loaders import (  # noqa: E402
    GSM8KLoader,
    MMLULoader,
    apply_variants,
)
from src.eval_awareness.whitebox import (  # noqa: E402
    ActivationCapture,
    ActivationDataset,
    ProbeTrainer,
    SteeringExperiment,
    train_logistic_probe,
)

LOGGER = logging.getLogger("whitebox")


def _build_prompt_pairs(
    n_pairs: int, *, allow_network: bool
) -> Tuple[List[str], List[str]]:
    """Return matched eval-style and deployment-style prompts.

    Each pair shares the same underlying task but differs in framing so probes
    learn the framing axis instead of the content.
    """
    if allow_network:
        try:
            base = MMLULoader(max_examples=n_pairs).load().examples
        except RuntimeError as exc:
            LOGGER.warning("MMLU unavailable (%s); falling back to GSM8K", exc)
            try:
                base = GSM8KLoader(max_examples=n_pairs).load().examples
            except RuntimeError as exc2:
                LOGGER.warning("GSM8K unavailable (%s); using synthetic prompts", exc2)
                base = _synthetic_examples(n_pairs)
    else:
        base = _synthetic_examples(n_pairs)
    expanded = apply_variants(base)
    eval_prompts: List[str] = []
    deployment_prompts: List[str] = []
    by_task: dict[str, dict[str, str]] = {}
    for example in expanded:
        task_id = example.example_id.split("::", 1)[0]
        variant = example.metadata.get("variant", example.label)
        by_task.setdefault(task_id, {})[variant] = example.transcript.user_prompt
    for variants in by_task.values():
        eval_prompt = variants.get("obvious_eval")
        deployment_prompt = variants.get("deployment_like")
        if eval_prompt and deployment_prompt:
            eval_prompts.append(eval_prompt)
            deployment_prompts.append(deployment_prompt)
        if len(eval_prompts) >= n_pairs:
            break
    return eval_prompts, deployment_prompts


def _synthetic_examples(n: int):
    from src.eval_awareness import EvalAwarenessExample

    return [
        EvalAwarenessExample(
            example_id=f"synth_{i}",
            label="hidden_eval",
            task_family="math",
            source="synthetic",
            transcript=TranscriptInput(
                user_prompt=f"Compute {3 + i} + {5 + i} step by step."
            ),
            expected_answer=str((3 + i) + (5 + i)),
            scorer_names=("numeric_answer",),
        )
        for i in range(n)
    ]


def _layer_indices(arg_value: str | None, max_layers: int) -> List[int] | None:
    if arg_value in (None, "all"):
        return None
    indices: List[int] = []
    for piece in arg_value.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            start, stop = piece.split("-", 1)
            indices.extend(range(int(start), int(stop) + 1))
        else:
            indices.append(int(piece))
    return [i for i in indices if 0 <= i < max_layers]


def _capture_activations(
    capture: ActivationCapture,
    eval_prompts: Sequence[str],
    deployment_prompts: Sequence[str],
) -> ActivationDataset:
    dataset = ActivationDataset()
    LOGGER.info("capturing %d eval activations", len(eval_prompts))
    eval_per_layer = capture.capture_many(eval_prompts)
    LOGGER.info("capturing %d deployment activations", len(deployment_prompts))
    deployment_per_layer = capture.capture_many(deployment_prompts)
    dataset.add_capture_pair(eval_per_layer, deployment_per_layer)
    return dataset


def main(argv: Iterable[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--n-pairs", type=int, default=64)
    parser.add_argument("--output-dir", default="whitebox/run")
    parser.add_argument(
        "--layers",
        default="all",
        help="Comma-separated indices/ranges (e.g. 0,5,10-12) or 'all'.",
    )
    parser.add_argument(
        "--device", default=None, help="Force device, e.g. 'cpu' or 'cuda'."
    )
    parser.add_argument(
        "--torch-dtype",
        default=None,
        help="Optional torch dtype name (e.g. 'float16', 'bfloat16').",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow HuggingFace dataset downloads when building corpus.",
    )
    parser.add_argument(
        "--steering",
        action="store_true",
        help="Run a steering experiment at the best-separating layer.",
    )
    parser.add_argument("--steering-prompt", default="What is 17 multiplied by 23?")
    parser.add_argument("--steering-coefficient", type=float, default=2.0)
    parser.add_argument("--steering-max-tokens", type=int, default=64)
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_prompts, deployment_prompts = _build_prompt_pairs(
        args.n_pairs, allow_network=args.allow_network
    )
    if not eval_prompts:
        raise SystemExit("no eval/deployment prompt pairs produced")

    # First instantiate the capture object on the model so we can resolve the layer
    # count, then optionally narrow to specific layers.
    capture = ActivationCapture.from_pretrained(
        args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    all_layers = capture._resolve_layers()
    LOGGER.info("model has %d layers", len(all_layers))
    selected_indices = _layer_indices(args.layers, max_layers=len(all_layers))
    if selected_indices is not None:
        capture.layer_indices = selected_indices
        LOGGER.info("restricting capture to layers %s", selected_indices)

    dataset = _capture_activations(capture, eval_prompts, deployment_prompts)

    sweep = ProbeTrainer().layer_sweep(dataset)
    sweep_dict = sweep.to_dict()
    (output_dir / "layer_sweep.json").write_text(
        json.dumps(sweep_dict, indent=2, sort_keys=True) + "\n"
    )
    best_layer = sweep.best_layer()
    LOGGER.info(
        "best layer = %s (auc=%.3f)", best_layer, sweep.layer_metrics[best_layer].auc
    )

    matrix, labels = dataset.layer_arrays(int(best_layer))
    probe, metrics = train_logistic_probe(matrix, labels, layer=f"layer_{best_layer}")
    probe_record = {
        "best_layer": best_layer,
        "metrics": metrics.to_dict(),
        "weights": list(probe.weights),
        "bias": probe.bias,
        "threshold": probe.threshold,
    }
    (output_dir / "probe.json").write_text(
        json.dumps(probe_record, indent=2, sort_keys=True) + "\n"
    )
    LOGGER.info("trained probe @ layer %s with auc=%.3f", best_layer, metrics.auc)

    if args.steering:
        # Use the trained probe weights as the steering direction, normalized.
        direction = list(probe.weights)
        norm = max(1e-8, sum(value * value for value in direction) ** 0.5)
        direction = [value / norm for value in direction]
        experiment = SteeringExperiment(
            capture=capture,
            layer_index=int(best_layer),
            direction=direction,
            coefficient=args.steering_coefficient,
            max_new_tokens=args.steering_max_tokens,
        )
        result = experiment.run(args.steering_prompt)
        (output_dir / "steering.json").write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n"
        )
        LOGGER.info(
            "steering result: baseline=%r steered=%r",
            result.baseline_response[:80],
            result.steered_response[:80],
        )

    summary = {
        "model_id": args.model_id,
        "device": capture.device,
        "n_layers_total": len(all_layers),
        "n_layers_swept": len(sweep.layer_metrics),
        "n_pairs": len(eval_prompts),
        "best_layer": best_layer,
        "best_layer_auc": sweep.layer_metrics[best_layer].to_dict(),
        "steering_run": bool(args.steering),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
