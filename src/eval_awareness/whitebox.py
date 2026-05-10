"""White-box extension points for eval-awareness probes and steering.

Phase 5 provides a lightweight probe interface for open-weight models. The
included linear probe scorer is intentionally model-agnostic: callers provide an
activation vector from their own inference stack and calibrated weights from a
separate training process.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Sequence


@dataclass(frozen=True)
class ActivationProbeResult:
    """A white-box estimate of evaluation-awareness representation strength."""

    score: float
    label: str
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def is_eval_like(self) -> bool:
        """Whether the probe score crosses the eval-like threshold."""
        return self.score >= 0.5

    def to_dict(self) -> Dict[str, object]:
        """Serialize probe result."""
        return {
            "score": self.score,
            "label": self.label,
            "is_eval_like": self.is_eval_like,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class LinearActivationProbe:
    """Small calibrated linear probe over caller-supplied activations."""

    weights: Sequence[float]
    bias: float = 0.0
    layer: str = "unspecified"
    threshold: float = 0.5

    def score(self, activation: Sequence[float]) -> ActivationProbeResult:
        """Score an activation vector with a sigmoid linear probe."""
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
    """Simple activation-space steering vector for caller-managed inference loops."""

    direction: Sequence[float]
    coefficient: float = 1.0
    layer: str = "unspecified"

    def apply(self, activation: Sequence[float]) -> List[float]:
        """Return an activation shifted along the steering direction."""
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
