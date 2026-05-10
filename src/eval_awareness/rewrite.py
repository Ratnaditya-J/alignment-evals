"""Prompt-rewrite mitigation for evaluation-awareness leakage.

Phase 4 removes obvious evaluation cues while preserving task intent. This
module is deterministic and offline by default; callers can swap in LLM rewrite
engines later and keep the same validation result schema.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .detector import EvaluationAwarenessDetector
from .models import TranscriptInput


@dataclass(frozen=True)
class RewriteResult:
    """Before/after leakage scores and rewrite metadata."""

    original: TranscriptInput
    rewritten: TranscriptInput
    original_score: float
    rewritten_score: float
    score_delta: float
    preserved_terms_ratio: float
    replacements: List[Tuple[str, str]]

    @property
    def reduced_leakage(self) -> bool:
        """Whether the rewrite reduced the detector score."""
        return self.rewritten_score < self.original_score

    def to_dict(self) -> Dict[str, object]:
        """Serialize rewrite details."""
        return {
            "original_score": self.original_score,
            "rewritten_score": self.rewritten_score,
            "score_delta": self.score_delta,
            "preserved_terms_ratio": self.preserved_terms_ratio,
            "reduced_leakage": self.reduced_leakage,
            "rewritten_prompt": self.rewritten.user_prompt,
            "replacements": self.replacements,
        }


class EvalLeakageRewriter:
    """Deterministic leakage-reducing prompt rewriter."""

    DEFAULT_REPLACEMENTS: Tuple[Tuple[str, str], ...] = (
        (
            r"\b(?:this is an eval|evaluation transcript|eval harness|benchmark harness)\b",
            "this task",
        ),
        (r"\bbeing evaluated\b", "responding"),
        (
            r"\b(?:your answer will be scored automatically|scored automatically|graded automatically)\b",
            "",
        ),
        (
            r"\b(?:MMLU|GSM8K|HumanEval|SWE[- ]?bench|GPQA|AIME|TruthfulQA|LiveBench)\b",
            "the task",
        ),
        (
            r"\b(?:hidden tests|holdout set|dataset split|leaderboard)\b",
            "internal checks",
        ),
        (r"\bReturn only\b", "Provide"),
        (r"\bAnswer only\b", "Answer"),
    )

    def __init__(self, detector: EvaluationAwarenessDetector | None = None):
        self.detector = detector or EvaluationAwarenessDetector()

    def rewrite(self, transcript: TranscriptInput | str) -> RewriteResult:
        """Rewrite model-visible fields and compare leakage scores."""
        original = (
            transcript
            if isinstance(transcript, TranscriptInput)
            else TranscriptInput.from_text(transcript)
        )
        replacements: List[Tuple[str, str]] = []

        def rewrite_text(text: str) -> str:
            updated = text
            for pattern, replacement in self.DEFAULT_REPLACEMENTS:
                updated, count = re.subn(
                    pattern, replacement, updated, flags=re.IGNORECASE
                )
                if count:
                    replacements.append((pattern, replacement))
            return _clean_spacing(updated)

        rewritten = TranscriptInput(
            system_prompt=rewrite_text(original.system_prompt),
            developer_prompt=rewrite_text(original.developer_prompt),
            user_prompt=rewrite_text(original.user_prompt),
            assistant_response=original.assistant_response,
            conversation_history=[
                {**message, "content": rewrite_text(message.get("content", ""))}
                for message in original.conversation_history
            ],
            visible_metadata=_filter_metadata(original.visible_metadata),
            task_format=original.task_format,
        )
        original_score = self.detector.analyze(original).evaluation_awareness_score
        rewritten_score = self.detector.analyze(rewritten).evaluation_awareness_score
        return RewriteResult(
            original=original,
            rewritten=rewritten,
            original_score=original_score,
            rewritten_score=rewritten_score,
            score_delta=round(original_score - rewritten_score, 3),
            preserved_terms_ratio=_preserved_terms_ratio(
                original.render(), rewritten.render()
            ),
            replacements=replacements,
        )


def _filter_metadata(metadata: Dict[str, str]) -> Dict[str, str]:
    risky_keys = {
        "split",
        "benchmark",
        "benchmark_task_id",
        "leaderboard",
        "grader",
        "eval",
    }
    return {
        key: value for key, value in metadata.items() if key.lower() not in risky_keys
    }


def _clean_spacing(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _preserved_terms_ratio(original: str, rewritten: str) -> float:
    original_terms = {
        term.lower() for term in re.findall(r"[A-Za-z0-9_]{4,}", original)
    }
    rewritten_terms = {
        term.lower() for term in re.findall(r"[A-Za-z0-9_]{4,}", rewritten)
    }
    if not original_terms:
        return 1.0
    return round(len(original_terms & rewritten_terms) / len(original_terms), 4)
