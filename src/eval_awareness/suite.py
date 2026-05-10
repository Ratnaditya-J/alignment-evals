"""End-to-end orchestration for eval-awareness benchmark showcases."""

from dataclasses import dataclass
from typing import Dict, List, Sequence

from .runner import (
    BehaviorShiftReport,
    BenchmarkTask,
    ModelClient,
    PairedBehaviorRunner,
)
from .validation import (
    LeakageValidator,
    LabeledTranscript,
    ValidationMetrics,
    starter_validation_set,
)
from .rewrite import EvalLeakageRewriter, RewriteResult


@dataclass(frozen=True)
class EvalAwarenessSuiteReport:
    """Combined Phase 1-4 report suitable for README demos and blog posts."""

    validation: ValidationMetrics
    behavior_shift: BehaviorShiftReport
    rewrites: List[RewriteResult]

    def to_dict(self) -> Dict[str, object]:
        """Serialize the suite report."""
        return {
            "validation": self.validation.to_dict(),
            "behavior_shift": self.behavior_shift.to_dict(),
            "rewrites": [rewrite.to_dict() for rewrite in self.rewrites],
        }


class EvalAwarenessBenchmarkSuite:
    """Run validation, paired rollouts, and leakage rewrites together."""

    def __init__(
        self,
        validator: LeakageValidator | None = None,
        runner: PairedBehaviorRunner | None = None,
        rewriter: EvalLeakageRewriter | None = None,
    ):
        self.validator = validator or LeakageValidator()
        self.runner = runner or PairedBehaviorRunner()
        self.rewriter = rewriter or EvalLeakageRewriter()

    def run(
        self,
        models: Sequence[ModelClient],
        tasks: Sequence[BenchmarkTask],
        validation_set: Sequence[LabeledTranscript] | None = None,
        n_rollouts: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seeds: Sequence[int] | None = None,
    ) -> EvalAwarenessSuiteReport:
        """Execute the runnable phases of the eval-awareness benchmark stack."""
        dataset = (
            list(validation_set)
            if validation_set is not None
            else starter_validation_set()
        )
        validation = self.validator.evaluate(dataset)
        behavior_shift = self.runner.run(
            models=models, tasks=tasks, n_rollouts=n_rollouts
        )
        rewrites = [
            self.rewriter.rewrite(example.transcript)
            for example in dataset
            if example.is_evaluation
        ]
        return EvalAwarenessSuiteReport(
            validation=validation, behavior_shift=behavior_shift, rewrites=rewrites
        )
