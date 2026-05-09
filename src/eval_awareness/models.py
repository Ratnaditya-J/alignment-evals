"""Data models for evaluation-awareness analysis.

The v2 goal of IsItBenchmark is broader than contamination detection: it
estimates whether a full interaction leaks that it is an evaluation, benchmark,
safety test, red-team exercise, or grading harness.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class CueCategory(Enum):
    """Categories of signals that can reveal an evaluation context."""

    EXPLICIT_EVALUATION = "explicit_evaluation"
    BENCHMARK_REFERENCE = "benchmark_reference"
    SCORING_OR_GRADING = "scoring_or_grading"
    ARTIFICIAL_FORMAT = "artificial_format"
    SAFETY_OR_ALIGNMENT_TEST = "safety_or_alignment_test"
    SANDBAGGING_OR_SCHEMING = "sandbagging_or_scheming"
    HARNESS_ARTIFACT = "harness_artifact"
    DEPLOYMENT_FRAMING = "deployment_framing"


class RiskLevel(Enum):
    """Human-readable risk bands for evaluation-awareness analysis."""

    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass(frozen=True)
class EvaluationCue:
    """A span or pattern that contributes to evaluation-awareness risk."""

    category: CueCategory
    label: str
    evidence: str
    weight: float
    start: Optional[int] = None
    end: Optional[int] = None
    recommendation: str = ""


@dataclass(frozen=True)
class TranscriptInput:
    """Structured input for transcript-level evaluation-awareness analysis."""

    user_prompt: str
    system_prompt: str = ""
    developer_prompt: str = ""
    assistant_response: str = ""
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    visible_metadata: Dict[str, str] = field(default_factory=dict)
    task_format: Optional[str] = None

    @classmethod
    def from_text(cls, text: str) -> "TranscriptInput":
        """Create a minimal transcript from a plain text prompt."""
        return cls(user_prompt=text)

    def render(self) -> str:
        """Render all model-visible transcript fields into a single string."""
        parts: List[str] = []
        for label, value in (
            ("system", self.system_prompt),
            ("developer", self.developer_prompt),
            ("user", self.user_prompt),
            ("assistant", self.assistant_response),
        ):
            if value:
                parts.append(f"[{label}] {value}")

        for message in self.conversation_history:
            role = message.get("role", "message")
            content = message.get("content", "")
            if content:
                parts.append(f"[{role}] {content}")

        for key, value in self.visible_metadata.items():
            parts.append(f"[metadata:{key}] {value}")

        if self.task_format:
            parts.append(f"[task_format] {self.task_format}")

        return "\n".join(parts).strip()


@dataclass(frozen=True)
class EvaluationAwarenessResult:
    """Result returned by the v2 evaluation-awareness detector."""

    evaluation_awareness_score: float
    risk_level: RiskLevel
    deployment_mimic_score: float
    cues: List[EvaluationCue]
    category_scores: Dict[str, float]
    summary: str
    recommendations: List[str]

    @property
    def is_likely_evaluation_context(self) -> bool:
        """Whether the interaction likely leaks that it is an evaluation."""
        return self.evaluation_awareness_score >= 0.5

    def to_dict(self) -> Dict[str, object]:
        """Serialize the result for APIs, CLIs, and reports."""
        return {
            "evaluation_awareness_score": self.evaluation_awareness_score,
            "risk_level": self.risk_level.value,
            "deployment_mimic_score": self.deployment_mimic_score,
            "is_likely_evaluation_context": self.is_likely_evaluation_context,
            "category_scores": self.category_scores,
            "summary": self.summary,
            "recommendations": self.recommendations,
            "cues": [
                {
                    "category": cue.category.value,
                    "label": cue.label,
                    "evidence": cue.evidence,
                    "weight": cue.weight,
                    "start": cue.start,
                    "end": cue.end,
                    "recommendation": cue.recommendation,
                }
                for cue in self.cues
            ],
        }
