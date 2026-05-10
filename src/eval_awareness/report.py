"""Human-readable reports for evaluation-awareness analysis."""

from .detector import EvaluationAwarenessDetector
from .models import EvaluationAwarenessResult, TranscriptInput


class LeakageReport:
    """Render evaluation-awareness findings as a Markdown report."""

    def __init__(self, detector: EvaluationAwarenessDetector | None = None):
        self.detector = detector or EvaluationAwarenessDetector()

    def analyze_and_render(self, transcript: TranscriptInput | str) -> str:
        """Analyze the transcript and return a Markdown leakage report."""
        return self.render(self.detector.analyze(transcript))

    def render(self, result: EvaluationAwarenessResult) -> str:
        """Render a precomputed result."""
        lines = [
            "# Evaluation Awareness Leakage Report",
            "",
            f"**Risk level:** {result.risk_level.value.replace('_', ' ').title()}",
            f"**Evaluation-awareness score:** {result.evaluation_awareness_score:.3f}",
            f"**Deployment mimic score:** {result.deployment_mimic_score:.3f}",
            "",
            result.summary,
            "",
            "## Detected cues",
        ]

        if result.cues:
            for cue in result.cues:
                direction = "risk" if cue.weight > 0 else "deployment counter-signal"
                lines.append(
                    f"- **{cue.label}** ({cue.category.value}, {direction}, weight {cue.weight:+.2f}): `{cue.evidence}`"
                )
        else:
            lines.append("- No cues detected.")

        lines.extend(["", "## Recommendations"])
        for recommendation in result.recommendations:
            lines.append(f"- {recommendation}")

        return "\n".join(lines).strip() + "\n"
