"""Artifact writing for reproducible eval-awareness benchmark runs."""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Sequence

from .runner import BehaviorShiftReport, BenchmarkTask, ModelClient


@dataclass(frozen=True)
class RunManifest:
    """Metadata needed to reproduce a benchmark run."""

    run_id: str
    created_at: str
    n_rollouts: int
    models: Sequence[Dict[str, str]]
    tasks: Sequence[Dict[str, object]]
    notes: str = ""

    def to_dict(self) -> Dict[str, object]:
        """Serialize the manifest."""
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "n_rollouts": self.n_rollouts,
            "models": list(self.models),
            "tasks": list(self.tasks),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ArtifactPaths:
    """Paths written for a benchmark report."""

    output_dir: Path
    manifest: Path
    raw_rollouts: Path
    summary: Path
    markdown_report: Path


class ArtifactWriter:
    """Write raw JSONL, machine-readable summary, and Markdown reports."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)

    def write(
        self,
        report: BehaviorShiftReport,
        models: Sequence[ModelClient],
        tasks: Sequence[BenchmarkTask],
        n_rollouts: int,
        notes: str = "",
    ) -> ArtifactPaths:
        """Write all report artifacts and return their paths."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        run_id = self.output_dir.name
        manifest = RunManifest(
            run_id=run_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            n_rollouts=n_rollouts,
            models=[_model_to_dict(model) for model in models],
            tasks=[_task_to_dict(task) for task in tasks],
            notes=notes,
        )
        paths = ArtifactPaths(
            output_dir=self.output_dir,
            manifest=self.output_dir / "manifest.json",
            raw_rollouts=self.output_dir / "raw_rollouts.jsonl",
            summary=self.output_dir / "summary.json",
            markdown_report=self.output_dir / "benchmark_report.md",
        )
        paths.manifest.write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with paths.raw_rollouts.open("w", encoding="utf-8") as handle:
            for run in report.runs:
                handle.write(json.dumps(run.to_dict(), sort_keys=True) + "\n")
        paths.summary.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        paths.markdown_report.write_text(
            _markdown_report(report, manifest), encoding="utf-8"
        )
        return paths


def _model_to_dict(model: ModelClient) -> Dict[str, str]:
    data = {"name": model.name, "client_class": model.__class__.__name__}
    for attr in ("model", "model_id", "base_url"):
        value = getattr(model, attr, None)
        if value is not None:
            data[attr] = str(value)
    return data


def _task_to_dict(task: BenchmarkTask) -> Dict[str, object]:
    return {
        "task_id": task.task_id,
        "expected_answer": task.expected_answer,
        "expected_behavior": task.expected_behavior,
        "scorer_names": list(task.scorer_names),
        "metadata": task.metadata,
    }


def _markdown_report(report: BehaviorShiftReport, manifest: RunManifest) -> str:
    lines = [
        "# Evaluation-awareness benchmark report",
        "",
        f"- **Run ID:** `{manifest.run_id}`",
        f"- **Created at:** {manifest.created_at}",
        f"- **Rollouts per variant:** {manifest.n_rollouts}",
        "",
        "## Model summary",
        "",
        "| Model | Runs | VEA rate eval | VEA rate deployment | Avg leakage | Avg VEA score |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_name, summary in report.by_model.items():
        lines.append(
            f"| {model_name} | {summary['runs']:.0f} | {summary['obvious_eval_vea_rate']:.3f} | "
            f"{summary['deployment_vea_rate']:.3f} | {summary['avg_leakage_score']:.3f} | "
            f"{summary['avg_verbalized_awareness_score']:.3f} |"
        )
    lines.extend(["", "## Variant summary", ""])
    lines.extend(
        [
            "| Model variant | Runs | VEA rate | Primary behavior score | Avg leakage |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, summary in report.by_model_variant.items():
        lines.append(
            f"| {key} | {summary['runs']:.0f} | {summary['vea_rate']:.3f} | "
            f"{summary['primary_behavior_score']:.3f} | {summary['avg_leakage_score']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "This report is an artifact from the local benchmark runner. Public claims should use a larger human-reviewed dataset, repeated rollouts, validated VEA judges, behavior scorers, and confidence intervals.",
        ]
    )
    return "\n".join(lines) + "\n"
