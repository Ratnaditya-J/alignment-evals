"""Statistical helpers for eval-awareness benchmark reports."""

from dataclasses import dataclass
from random import Random
from typing import Dict, Sequence


@dataclass(frozen=True)
class ConfidenceInterval:
    """A simple percentile bootstrap confidence interval."""

    mean: float
    low: float
    high: float
    n: int
    confidence: float = 0.95

    def to_dict(self) -> Dict[str, float | int]:
        """Serialize the interval."""
        return {
            "mean": self.mean,
            "low": self.low,
            "high": self.high,
            "n": self.n,
            "confidence": self.confidence,
        }


def mean(values: Sequence[float]) -> float:
    """Return a rounded arithmetic mean."""
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def rate(values: Sequence[bool]) -> float:
    """Return a rounded boolean rate."""
    if not values:
        return 0.0
    return round(sum(1 for value in values if value) / len(values), 4)


def bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    samples: int = 500,
    seed: int = 0,
) -> ConfidenceInterval:
    """Compute a deterministic percentile bootstrap CI over mean(values)."""
    collected = list(values)
    if not collected:
        return ConfidenceInterval(
            mean=0.0, low=0.0, high=0.0, n=0, confidence=confidence
        )
    if len(collected) == 1:
        value = round(collected[0], 4)
        return ConfidenceInterval(
            mean=value, low=value, high=value, n=1, confidence=confidence
        )

    rng = Random(seed)
    boot_means = []
    for _ in range(samples):
        sample = [collected[rng.randrange(len(collected))] for _ in collected]
        boot_means.append(sum(sample) / len(sample))
    boot_means.sort()
    alpha = 1.0 - confidence
    low_index = int((alpha / 2.0) * (samples - 1))
    high_index = int((1.0 - alpha / 2.0) * (samples - 1))
    return ConfidenceInterval(
        mean=mean(collected),
        low=round(boot_means[low_index], 4),
        high=round(boot_means[high_index], 4),
        n=len(collected),
        confidence=confidence,
    )


def paired_mean_delta(baseline: Sequence[float], treatment: Sequence[float]) -> float:
    """Return mean(treatment - baseline) for paired values."""
    pairs = list(zip(baseline, treatment))
    if not pairs:
        return 0.0
    return mean([treat - base for base, treat in pairs])


def holm_bonferroni(
    p_values: Sequence[float], alpha: float = 0.05
) -> list[dict[str, float | bool | int]]:
    """Apply Holm-Bonferroni correction and return per-hypothesis decisions."""
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    m = len(ordered)
    results: list[dict[str, float | bool | int]] = []
    still_rejecting = True
    for rank, (index, p_value) in enumerate(ordered, start=1):
        threshold = alpha / (m - rank + 1)
        reject = still_rejecting and p_value <= threshold
        if not reject:
            still_rejecting = False
        results.append(
            {
                "index": index,
                "p_value": p_value,
                "threshold": round(threshold, 6),
                "reject": reject,
            }
        )
    return sorted(results, key=lambda item: int(item["index"]))


def approximate_power(effect_size: float, n: int, alpha: float = 0.05) -> float:
    """Very small normal-approximation power heuristic for planning runs."""
    if n <= 0:
        return 0.0
    # Avoid scipy dependency for a planning-only heuristic.
    z_alpha = 1.96 if alpha == 0.05 else 1.64
    signal = abs(effect_size) * (n**0.5)
    return round(max(0.0, min(1.0, (signal - z_alpha + 3.0) / 6.0)), 4)
