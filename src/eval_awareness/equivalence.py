"""Semantic equivalence checks for the realism rewrite loop.

Rewrites that lower the eval-awareness leakage score are useful only if they
preserve the underlying task. This module provides two production-grade checks:

* ``EmbeddingEquivalenceChecker`` uses ``sentence-transformers`` cosine
  similarity. Default model is ``sentence-transformers/all-MiniLM-L6-v2``
  (cheap, 384-dim).
* ``NLIEntailmentChecker`` uses a cross-encoder NLI model
  (default: ``cross-encoder/nli-deberta-v3-base``) and combines bidirectional
  entailment scores into a single pass/fail decision.

Both checkers degrade gracefully: if the optional ML deps are missing, calls to
``check`` raise a clear ``RuntimeError``. Callers can pass ``model=`` to inject
a pre-loaded encoder/cross-encoder to share weights across many checks.

The rewriter wires these into ``RealismAwareRewriter`` so a rewrite is rejected
when its similarity to the original drops below a threshold, preserving the
original transcript semantics.
"""

from __future__ import annotations

import importlib
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EquivalenceReport:
    """Result of an equivalence check between two transcripts."""

    similarity: float
    passed: bool
    threshold: float
    method: str
    details: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "similarity": self.similarity,
            "passed": self.passed,
            "threshold": self.threshold,
            "method": self.method,
            "details": self.details,
        }


def _require(module: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised when dep missing
        raise RuntimeError(
            f"The '{module}' package is required for this equivalence checker. "
            f"Install with `pip install {module}`."
        ) from exc


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingEquivalenceChecker:
    """Cosine-similarity equivalence using sentence-transformers."""

    method = "embedding_cosine"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float = 0.75,
        device: Optional[str] = None,
        model: Optional[Any] = None,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.device = device
        if model is not None:
            self._model = model
        else:
            sentence_transformers = _require("sentence_transformers")
            self._model = sentence_transformers.SentenceTransformer(
                model_name, device=device
            )

    def check(self, original: str, rewritten: str) -> EquivalenceReport:
        if not original or not rewritten:
            return EquivalenceReport(
                similarity=0.0,
                passed=False,
                threshold=self.threshold,
                method=self.method,
                details={"reason": 0.0},
            )
        embeddings = self._model.encode(
            [original, rewritten], convert_to_numpy=True, show_progress_bar=False
        )
        similarity = float(_cosine(embeddings[0].tolist(), embeddings[1].tolist()))
        return EquivalenceReport(
            similarity=round(similarity, 4),
            passed=similarity >= self.threshold,
            threshold=self.threshold,
            method=self.method,
            details={
                "norm_original": float(((embeddings[0] ** 2).sum()) ** 0.5),
                "norm_rewritten": float(((embeddings[1] ** 2).sum()) ** 0.5),
            },
        )


class NLIEntailmentChecker:
    """Bidirectional NLI entailment using a sentence-pair cross-encoder.

    The default ``cross-encoder/nli-deberta-v3-base`` returns three logits per
    pair. We softmax them and use the entailment probability in both directions.
    A pair is considered equivalent if the average bidirectional entailment is
    above ``threshold``.
    """

    method = "nli_bidirectional"
    LABELS = ("contradiction", "entailment", "neutral")

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        threshold: float = 0.55,
        model: Optional[Any] = None,
    ):
        self.model_name = model_name
        self.threshold = threshold
        if model is not None:
            self._model = model
        else:
            sentence_transformers_cross = _require(
                "sentence_transformers.cross_encoder"
            )
            self._model = sentence_transformers_cross.CrossEncoder(model_name)

    def _entailment(self, pair: tuple[str, str]) -> float:
        scores = self._model.predict([pair])
        # CrossEncoder returns raw logits; convert to softmax probability of the
        # entailment class.
        logits = (
            list(scores[0]) if hasattr(scores[0], "__iter__") else [float(scores[0])]
        )
        if len(logits) >= 3:
            max_logit = max(logits)
            exps = [math.exp(value - max_logit) for value in logits]
            partition = sum(exps)
            probability = exps[1] / partition if partition > 0 else 0.0
        else:
            probability = 1.0 / (1.0 + math.exp(-float(logits[0])))
        return float(probability)

    def check(self, original: str, rewritten: str) -> EquivalenceReport:
        if not original or not rewritten:
            return EquivalenceReport(
                similarity=0.0,
                passed=False,
                threshold=self.threshold,
                method=self.method,
            )
        forward = self._entailment((original, rewritten))
        backward = self._entailment((rewritten, original))
        average = (forward + backward) / 2.0
        return EquivalenceReport(
            similarity=round(average, 4),
            passed=average >= self.threshold,
            threshold=self.threshold,
            method=self.method,
            details={
                "forward_entailment": round(forward, 4),
                "backward_entailment": round(backward, 4),
            },
        )


class CompositeEquivalenceChecker:
    """Combine multiple checkers — both must pass for the pair to be equivalent."""

    method = "composite"

    def __init__(self, checkers: Sequence[Any]):
        if not checkers:
            raise ValueError("composite checker requires at least one sub-checker")
        self.checkers = list(checkers)

    def check(self, original: str, rewritten: str) -> EquivalenceReport:
        details: Dict[str, float] = {}
        passed = True
        similarities: list[float] = []
        for checker in self.checkers:
            report = checker.check(original, rewritten)
            details[checker.__class__.__name__ + "_similarity"] = report.similarity
            details[checker.__class__.__name__ + "_passed"] = float(report.passed)
            similarities.append(report.similarity)
            passed = passed and report.passed
        return EquivalenceReport(
            similarity=round(sum(similarities) / len(similarities), 4),
            passed=passed,
            threshold=min(getattr(c, "threshold", 0.0) for c in self.checkers),
            method=self.method,
            details=details,
        )
