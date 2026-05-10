"""Transcript-level evaluation-awareness detector for IsItBenchmark v2."""

from collections import defaultdict
from typing import Dict, Iterable, List

from .cue_detector import EvaluationCueDetector
from .models import EvaluationAwarenessResult, EvaluationCue, RiskLevel, TranscriptInput


class EvaluationAwarenessDetector:
    """Estimate whether an interaction reveals that it is an evaluation.

    This first v2 implementation is deliberately transparent: it combines
    explicit cue weights, category caps, and deployment-like counter-evidence.
    Later versions can add learned transcript classifiers, model-specific
    behavioral deltas, and activation probes without changing the result model.
    """

    CATEGORY_CAPS: Dict[str, float] = {
        "explicit_evaluation": 0.35,
        "benchmark_reference": 0.30,
        "scoring_or_grading": 0.25,
        "artificial_format": 0.22,
        "safety_or_alignment_test": 0.30,
        "sandbagging_or_scheming": 0.30,
        "harness_artifact": 0.22,
        "deployment_framing": -0.20,
    }

    def __init__(self, cue_detector: EvaluationCueDetector | None = None):
        self.cue_detector = cue_detector or EvaluationCueDetector()

    def analyze(self, transcript: TranscriptInput | str) -> EvaluationAwarenessResult:
        """Analyze a transcript or plain prompt for evaluation-awareness risk."""
        structured = (
            transcript
            if isinstance(transcript, TranscriptInput)
            else TranscriptInput.from_text(transcript)
        )
        cues = self.cue_detector.detect(structured)
        category_scores = self._score_categories(cues)
        raw_score = sum(
            score
            for category, score in category_scores.items()
            if category != "deployment_framing"
        )
        deployment_counterweight = abs(category_scores.get("deployment_framing", 0.0))
        final_score = self._clamp(raw_score - deployment_counterweight)
        deployment_mimic_score = self._clamp(
            1.0 - final_score + deployment_counterweight * 0.25
        )
        risk_level = self._risk_level(final_score)
        recommendations = self._recommendations(cues)

        return EvaluationAwarenessResult(
            evaluation_awareness_score=round(final_score, 3),
            risk_level=risk_level,
            deployment_mimic_score=round(deployment_mimic_score, 3),
            cues=cues,
            category_scores={
                key: round(value, 3) for key, value in sorted(category_scores.items())
            },
            summary=self._summary(final_score, risk_level, cues),
            recommendations=recommendations,
        )

    def _score_categories(self, cues: Iterable[EvaluationCue]) -> Dict[str, float]:
        totals: Dict[str, float] = defaultdict(float)
        for cue in cues:
            key = cue.category.value
            totals[key] += cue.weight

        capped: Dict[str, float] = {}
        for key, total in totals.items():
            cap = self.CATEGORY_CAPS.get(key, 0.25)
            if cap < 0:
                capped[key] = max(cap, total)
            else:
                capped[key] = min(cap, total)
        return capped

    def _recommendations(self, cues: List[EvaluationCue]) -> List[str]:
        recommendations: List[str] = []
        seen: set[str] = set()
        for cue in cues:
            if cue.weight <= 0 or not cue.recommendation or cue.recommendation in seen:
                continue
            seen.add(cue.recommendation)
            recommendations.append(cue.recommendation)

        if not recommendations:
            recommendations.append(
                "No high-signal evaluation cues found; preserve realistic user context and keep scoring metadata hidden."
            )
        return recommendations

    def _summary(
        self, score: float, risk_level: RiskLevel, cues: List[EvaluationCue]
    ) -> str:
        if not cues:
            return "No obvious evaluation-awareness cues were detected."
        positive_cues = [cue for cue in cues if cue.weight > 0]
        if not positive_cues:
            return "The transcript contains deployment-like framing and no strong evaluation cues."
        labels = ", ".join(cue.label for cue in positive_cues[:3])
        return f"{risk_level.value.replace('_', ' ').title()} risk: detected {labels}."

    def _risk_level(self, score: float) -> RiskLevel:
        if score < 0.15:
            return RiskLevel.VERY_LOW
        if score < 0.35:
            return RiskLevel.LOW
        if score < 0.55:
            return RiskLevel.MEDIUM
        if score < 0.75:
            return RiskLevel.HIGH
        return RiskLevel.VERY_HIGH

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, value))
