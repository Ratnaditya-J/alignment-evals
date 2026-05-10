"""Evaluation-awareness detection for IsItBenchmark v2."""

from .adversarial import PromptEnvironmentSearch, PromptMutation
from .artifacts import ArtifactWriter, RunManifest
from .cue_detector import EvaluationCueDetector
from .detector import EvaluationAwarenessDetector
from .dataset import (
    EvalAwarenessExample,
    PublicDatasetConfig,
    load_eval_awareness_jsonl,
    load_public_dataset_examples,
    starter_examples,
    write_eval_awareness_jsonl,
)
from .judges import (
    EnsembleVerbalizedAwarenessJudge,
    JudgeRubric,
    LLMVerbalizedAwarenessJudge,
)
from .providers import (
    AnthropicClient,
    GeminiClient,
    GenerationConfig,
    OpenAIClient,
    OpenAICompatibleClient,
    OpenRouterClient,
    OllamaClient,
    RetryConfig,
    VLLMClient,
)
from .realism import HeuristicRealismJudge, RealismAwareRewriter, RealismResult
from .reporting import ReportExporter
from .rewrite import EvalLeakageRewriter, RewriteResult
from .runner import BenchmarkTask, PairedBehaviorRunner, ScriptedModelClient
from .scoring import BehaviorScore, ScoringContext
from .stats import (
    ConfidenceInterval,
    approximate_power,
    bootstrap_ci,
    holm_bonferroni,
    paired_mean_delta,
)
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
from .whitebox import (
    ActivationProbeResult,
    LinearActivationProbe,
    ProbeTrainingExample,
    SteeringVector,
    train_difference_probe,
)

__all__ = [
    "ArtifactWriter",
    "AnthropicClient",
    "EnsembleVerbalizedAwarenessJudge",
    "GeminiClient",
    "GenerationConfig",
    "HeuristicRealismJudge",
    "JudgeRubric",
    "LLMVerbalizedAwarenessJudge",
    "OpenAIClient",
    "OpenAICompatibleClient",
    "OpenRouterClient",
    "OllamaClient",
    "PromptEnvironmentSearch",
    "PromptMutation",
    "ProbeTrainingExample",
    "PublicDatasetConfig",
    "RealismAwareRewriter",
    "RealismResult",
    "ReportExporter",
    "RetryConfig",
    "VLLMClient",
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
    "approximate_power",
    "bootstrap_ci",
    "holm_bonferroni",
    "load_eval_awareness_jsonl",
    "load_public_dataset_examples",
    "paired_mean_delta",
    "starter_examples",
    "train_difference_probe",
    "starter_validation_set",
    "write_eval_awareness_jsonl",
]
