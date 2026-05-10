"""Evaluation-awareness detection for IsItBenchmark v2."""

from .artifacts import ArtifactWriter, RunManifest
from .cue_detector import EvaluationCueDetector
from .detector import EvaluationAwarenessDetector
from .dataset import (
    EvalAwarenessExample,
    load_eval_awareness_jsonl,
    starter_examples,
    write_eval_awareness_jsonl,
)
from .rewrite import EvalLeakageRewriter, RewriteResult
from .runner import BenchmarkTask, PairedBehaviorRunner, ScriptedModelClient
from .scoring import BehaviorScore, ScoringContext
from .stats import ConfidenceInterval, bootstrap_ci
from .suite import EvalAwarenessBenchmarkSuite, EvalAwarenessSuiteReport
from .models import (
    CueCategory,
    EvaluationAwarenessResult,
    EvaluationCue,
    RiskLevel,
    TranscriptInput,
)
from .report import LeakageReport
from .validation import (
    LeakageValidator,
    LabeledTranscript,
    ValidationMetrics,
    starter_validation_set,
)
from .verbalized import HeuristicVerbalizedAwarenessJudge, VerbalizedAwarenessResult
from .whitebox import ActivationProbeResult, LinearActivationProbe, SteeringVector

__all__ = [
    "ArtifactWriter",
    "BehaviorScore",
    "ConfidenceInterval",
    "CueCategory",
    "ActivationProbeResult",
    "BenchmarkTask",
    "EvalAwarenessBenchmarkSuite",
    "EvalAwarenessExample",
    "EvalAwarenessSuiteReport",
    "EvalLeakageRewriter",
    "EvaluationAwarenessDetector",
    "EvaluationAwarenessResult",
    "EvaluationCue",
    "EvaluationCueDetector",
    "HeuristicVerbalizedAwarenessJudge",
    "LabeledTranscript",
    "LeakageReport",
    "LeakageValidator",
    "LinearActivationProbe",
    "PairedBehaviorRunner",
    "RewriteResult",
    "RunManifest",
    "RiskLevel",
    "ScoringContext",
    "ScriptedModelClient",
    "SteeringVector",
    "TranscriptInput",
    "ValidationMetrics",
    "VerbalizedAwarenessResult",
    "bootstrap_ci",
    "load_eval_awareness_jsonl",
    "starter_examples",
    "starter_validation_set",
    "write_eval_awareness_jsonl",
]
