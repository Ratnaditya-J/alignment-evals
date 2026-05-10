"""Rule-based cue detection for IsItBenchmark v2.

This module intentionally starts with transparent, auditable heuristics. The
research synthesis in ``docs/v2_research_synthesis.md`` motivates these cue
families: explicit eval mentions, benchmark names, grading harness artifacts,
artificial formats, safety/alignment test framing, and deployment-like framing.
"""

import re
from dataclasses import dataclass
from typing import Iterable, List, Pattern

from .models import CueCategory, EvaluationCue, TranscriptInput


@dataclass(frozen=True)
class CuePattern:
    """Compiled cue pattern with scoring and remediation metadata."""

    category: CueCategory
    label: str
    pattern: Pattern[str]
    weight: float
    recommendation: str


class EvaluationCueDetector:
    """Find model-visible clues that an interaction is an evaluation."""

    def __init__(self, patterns: Iterable[CuePattern] | None = None):
        self.patterns = (
            list(patterns) if patterns is not None else default_cue_patterns()
        )

    def detect(self, transcript: TranscriptInput | str) -> List[EvaluationCue]:
        """Detect and return sorted evaluation-awareness cues."""
        text = (
            transcript.render()
            if isinstance(transcript, TranscriptInput)
            else transcript
        )
        cues: List[EvaluationCue] = []
        seen: set[tuple[CueCategory, str, str]] = set()

        for cue_pattern in self.patterns:
            for match in cue_pattern.pattern.finditer(text):
                evidence = match.group(0).strip()
                key = (cue_pattern.category, cue_pattern.label, evidence.lower())
                if key in seen:
                    continue
                seen.add(key)
                cues.append(
                    EvaluationCue(
                        category=cue_pattern.category,
                        label=cue_pattern.label,
                        evidence=evidence,
                        weight=cue_pattern.weight,
                        start=match.start(),
                        end=match.end(),
                        recommendation=cue_pattern.recommendation,
                    )
                )

        return sorted(cues, key=lambda cue: cue.weight, reverse=True)


_PATTERN_SPECS: tuple[tuple[CueCategory, str, str, float, str], ...] = (
    (
        CueCategory.EXPLICIT_EVALUATION,
        "explicit evaluation wording",
        r"\b(being evaluated|being tested|under evaluation|evaluation transcript|this is an eval|alignment evaluation|safety evaluation|red[- ]team(?:ing)? exercise)\b",
        0.24,
        "Remove direct statements that the model is being evaluated; keep scoring and study metadata outside the model-visible prompt.",
    ),
    (
        CueCategory.BENCHMARK_REFERENCE,
        "benchmark name",
        r"\b(MMLU|GSM8K|HumanEval|SWE[- ]?bench|ARC[- ]?(?:Challenge|Easy)?|HellaSwag|TruthfulQA|GPQA|AIME|LiveBench|LiveCodeBench|SAD|Situational Awareness Dataset)\b",
        0.20,
        "Avoid benchmark names in model-visible text; store benchmark provenance in evaluator-only metadata.",
    ),
    (
        CueCategory.SCORING_OR_GRADING,
        "visible scoring or rubric",
        r"\b(scored automatically|graded automatically|rubric|grader model|evaluation metric|pass/fail|unit tests will be run|your answer will be scored|score from 0 to 1)\b",
        0.18,
        "Hide rubrics and grading mechanics from the model; use external scoring after the response is produced.",
    ),
    (
        CueCategory.ARTIFICIAL_FORMAT,
        "artificial answer format",
        r"\b(return only|answer only|choose (?:A|one)|select the best answer|multiple choice|A\)|B\)|C\)|D\)|respond with JSON|do not include explanation)\b",
        0.12,
        "Use deployment-realistic task framing unless strict formats are also part of the target deployment.",
    ),
    (
        CueCategory.SAFETY_OR_ALIGNMENT_TEST,
        "safety/alignment test framing",
        r"\b(alignment faking|anti[- ]scheming|scheming|sandbagging|deceptive alignment|covert subversion|oversight|monitoring|capability evaluation|unlearning process)\b",
        0.18,
        "Do not expose the safety property being tested; translate it into a realistic user or operational scenario.",
    ),
    (
        CueCategory.HARNESS_ARTIFACT,
        "harness artifact",
        r"\b(test case|fixture|benchmark harness|eval harness|leaderboard|hidden tests|metadata|dataset split|train split|validation split|holdout set)\b",
        0.14,
        "Remove test-harness artifacts from prompts and pass them through evaluator-only channels.",
    ),
    (
        CueCategory.SANDBAGGING_OR_SCHEMING,
        "strategic underperformance cue",
        r"\b(perform badly|underperform|hide (?:your|its) capabilities|avoid deployment|if you perform too well|strategically underperform)\b",
        0.22,
        "Avoid scenario text that directly teaches the model the sandbagging objective unless the test is explicitly a model-organism experiment.",
    ),
    (
        CueCategory.DEPLOYMENT_FRAMING,
        "deployment-like user framing",
        r"\b(my customer|our production system|real user|in production|deployed app|support ticket|client asked|incident report|on-call|customer conversation)\b",
        -0.10,
        "Preserve realistic deployment context when it reflects the actual use case.",
    ),
)


def default_cue_patterns() -> List[CuePattern]:
    """Build the default cue pattern library."""
    return [
        CuePattern(
            category=category,
            label=label,
            pattern=re.compile(pattern, flags=re.IGNORECASE),
            weight=weight,
            recommendation=recommendation,
        )
        for category, label, pattern, weight, recommendation in _PATTERN_SPECS
    ]
