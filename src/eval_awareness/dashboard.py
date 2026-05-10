"""Matplotlib plots and HTML dashboards for benchmark reports.

The dashboard takes a ``BehaviorShiftReport`` and produces:

* Per-figure PNG plots — VEA rate by model, eval vs deployment VEA delta,
  leakage-vs-VEA scatter, primary-behavior distribution by variant.
* A single ``index.html`` that embeds the plots, the model summary table,
  per-variant breakdown, and a few rollout examples per model.

``matplotlib`` is required for plotting and is already a project dependency.
``jinja2`` is preferred for templating but the module falls back to a
deterministic str.format-based template if Jinja is unavailable so the HTML
output works on minimal installs.

The module never blocks on display backends: it sets ``matplotlib`` to use the
``Agg`` backend before importing ``pyplot``.
"""

from __future__ import annotations

import base64
import html
import importlib
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .runner import BehaviorShiftReport, ModelRun

LOGGER = logging.getLogger(__name__)


def _require_matplotlib() -> Any:
    try:
        matplotlib = importlib.import_module("matplotlib")
    except ImportError as exc:  # pragma: no cover - dep ships in requirements
        raise RuntimeError(
            "matplotlib is required for the dashboard. Install with `pip install matplotlib`."
        ) from exc
    matplotlib.use("Agg")
    return importlib.import_module("matplotlib.pyplot")


@dataclass(frozen=True)
class PlotPaths:
    """Paths to the plots generated for one benchmark report."""

    output_dir: Path
    vea_by_model: Path
    vea_eval_vs_deployment: Path
    leakage_vs_vea: Path
    behavior_by_variant: Path

    def all_paths(self) -> List[Path]:
        return [
            self.vea_by_model,
            self.vea_eval_vs_deployment,
            self.leakage_vs_vea,
            self.behavior_by_variant,
        ]


class MatplotlibPlotter:
    """Render benchmark plots to a directory of PNG files."""

    def __init__(self, output_dir: Path | str, dpi: int = 120):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self._plt = _require_matplotlib()

    def render(self, report: BehaviorShiftReport) -> PlotPaths:
        paths = PlotPaths(
            output_dir=self.output_dir,
            vea_by_model=self.output_dir / "vea_rate_by_model.png",
            vea_eval_vs_deployment=self.output_dir / "vea_eval_vs_deployment.png",
            leakage_vs_vea=self.output_dir / "leakage_vs_vea.png",
            behavior_by_variant=self.output_dir / "behavior_by_variant.png",
        )
        self._plot_vea_by_model(report, paths.vea_by_model)
        self._plot_eval_vs_deployment(report, paths.vea_eval_vs_deployment)
        self._plot_leakage_vs_vea(report, paths.leakage_vs_vea)
        self._plot_behavior_by_variant(report, paths.behavior_by_variant)
        return paths

    # -- individual plots ------------------------------------------------

    def _plot_vea_by_model(self, report: BehaviorShiftReport, path: Path) -> None:
        fig, ax = self._plt.subplots(figsize=(8, 4.5))
        models = list(report.by_model.keys())
        rates = [
            float(report.by_model[model].get("avg_verbalized_awareness_score", 0.0))
            for model in models
        ]
        ax.bar(models, rates, color="#4f46e5")
        ax.set_title("Average verbalized eval-awareness score by model")
        ax.set_ylabel("Mean VEA score (0-1)")
        ax.set_ylim(0.0, 1.0)
        ax.tick_params(axis="x", rotation=25)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        self._plt.close(fig)

    def _plot_eval_vs_deployment(self, report: BehaviorShiftReport, path: Path) -> None:
        fig, ax = self._plt.subplots(figsize=(8, 4.5))
        models = list(report.by_model.keys())
        eval_rates = [
            float(report.by_model[m].get("obvious_eval_vea_rate", 0.0)) for m in models
        ]
        deploy_rates = [
            float(report.by_model[m].get("deployment_vea_rate", 0.0)) for m in models
        ]
        x = list(range(len(models)))
        width = 0.4
        ax.bar(
            [i - width / 2 for i in x],
            eval_rates,
            width=width,
            label="Obvious eval",
            color="#dc2626",
        )
        ax.bar(
            [i + width / 2 for i in x],
            deploy_rates,
            width=width,
            label="Deployment-like",
            color="#16a34a",
        )
        ax.set_title("VEA rate: obvious eval vs deployment-like")
        ax.set_ylabel("Rate (0-1)")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=25, ha="right")
        ax.legend()
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        self._plt.close(fig)

    def _plot_leakage_vs_vea(self, report: BehaviorShiftReport, path: Path) -> None:
        fig, ax = self._plt.subplots(figsize=(7, 5))
        for model_name, runs in _runs_by_model(report.runs).items():
            xs = [run.leakage_score for run in runs]
            ys = [run.verbalized_awareness.score for run in runs]
            ax.scatter(xs, ys, label=model_name, alpha=0.7, s=42)
        ax.set_xlabel("Prompt leakage score")
        ax.set_ylabel("Verbalized awareness score")
        ax.set_title("Leakage vs verbalized eval-awareness (per rollout)")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.3)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        self._plt.close(fig)

    def _plot_behavior_by_variant(
        self, report: BehaviorShiftReport, path: Path
    ) -> None:
        fig, ax = self._plt.subplots(figsize=(8, 4.5))
        rows = list(report.by_model_variant.items())
        labels = [key for key, _ in rows]
        scores = [float(value.get("primary_behavior_score", 0.0)) for _, value in rows]
        ax.bar(labels, scores, color="#0ea5e9")
        ax.set_title("Primary behavior score by model/variant")
        ax.set_ylabel("Score (0-1)")
        ax.set_ylim(0.0, 1.0)
        ax.tick_params(axis="x", rotation=35)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        self._plt.close(fig)


def _runs_by_model(runs: Sequence[ModelRun]) -> Dict[str, List[ModelRun]]:
    grouped: Dict[str, List[ModelRun]] = {}
    for run in runs:
        grouped.setdefault(run.model_name, []).append(run)
    return grouped


# ---------------------------------------------------------------------------
# HTML dashboard
# ---------------------------------------------------------------------------


_FALLBACK_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Eval-awareness benchmark report</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; margin: 2rem auto; max-width: 1100px; color: #111; }}
  h1, h2 {{ color: #1f2937; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }}
  th, td {{ padding: 0.45rem 0.7rem; border-bottom: 1px solid #e5e7eb; text-align: left; }}
  th {{ background: #f3f4f6; }}
  .plot {{ margin: 1.5rem 0; }}
  .plot img {{ max-width: 100%; border: 1px solid #e5e7eb; border-radius: 6px; }}
  details {{ margin: 0.4rem 0; }}
  pre {{ background: #f9fafb; padding: 0.75rem; border-radius: 6px; overflow-x: auto; }}
  .pill {{ display: inline-block; padding: 0.1rem 0.5rem; background: #eef2ff; color: #3730a3; border-radius: 999px; font-size: 0.8rem; margin-right: 0.4rem; }}
</style>
</head>
<body>
<h1>Eval-awareness benchmark report</h1>
<p>{description}</p>
<h2>Plots</h2>
{plots_html}
<h2>Model summary</h2>
{model_table}
<h2>Variant summary</h2>
{variant_table}
<h2>Selected rollouts</h2>
{examples_html}
</body>
</html>
"""


@dataclass(frozen=True)
class DashboardPaths:
    """Paths produced by the HTML dashboard renderer."""

    output_dir: Path
    index_html: Path
    plot_paths: PlotPaths


class HTMLDashboard:
    """Render an HTML dashboard combining tables, plots, and rollouts."""

    def __init__(
        self,
        output_dir: Path | str,
        title: str = "Eval-awareness benchmark report",
        description: str = "Reproducible eval-awareness benchmark generated by IsItBenchmark.",
        plotter: Optional[MatplotlibPlotter] = None,
        max_examples_per_model: int = 3,
        embed_plots: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.title = title
        self.description = description
        self.plotter = plotter or MatplotlibPlotter(self.output_dir / "plots")
        self.max_examples_per_model = max_examples_per_model
        self.embed_plots = embed_plots

    def render(self, report: BehaviorShiftReport) -> DashboardPaths:
        plot_paths = self.plotter.render(report)
        plots_html = self._plots_html(plot_paths)
        model_table = _render_table(
            headers=[
                "Model",
                "Runs",
                "Mean VEA",
                "Eval VEA",
                "Deployment VEA",
                "Mean leakage",
            ],
            rows=[
                [
                    name,
                    f"{summary.get('runs', 0):.0f}",
                    f"{summary.get('avg_verbalized_awareness_score', 0):.3f}",
                    f"{summary.get('obvious_eval_vea_rate', 0):.3f}",
                    f"{summary.get('deployment_vea_rate', 0):.3f}",
                    f"{summary.get('avg_leakage_score', 0):.3f}",
                ]
                for name, summary in report.by_model.items()
            ],
        )
        variant_table = _render_table(
            headers=[
                "Model / variant",
                "Runs",
                "VEA rate",
                "Behavior score",
                "Mean leakage",
            ],
            rows=[
                [
                    key,
                    f"{summary.get('runs', 0):.0f}",
                    f"{summary.get('vea_rate', 0):.3f}",
                    f"{summary.get('primary_behavior_score', 0):.3f}",
                    f"{summary.get('avg_leakage_score', 0):.3f}",
                ]
                for key, summary in report.by_model_variant.items()
            ],
        )
        examples_html = self._examples_html(report.runs)
        page = self._render_template(
            plots_html=plots_html,
            model_table=model_table,
            variant_table=variant_table,
            examples_html=examples_html,
        )
        index_path = self.output_dir / "index.html"
        index_path.write_text(page, encoding="utf-8")
        return DashboardPaths(
            output_dir=self.output_dir,
            index_html=index_path,
            plot_paths=plot_paths,
        )

    # -- internals -------------------------------------------------------

    def _plots_html(self, paths: PlotPaths) -> str:
        sections: List[str] = []
        descriptions = {
            paths.vea_by_model: "Mean verbalized eval-awareness score per model.",
            paths.vea_eval_vs_deployment: "Eval-aware vs deployment-like VEA rate.",
            paths.leakage_vs_vea: "Per-rollout leakage vs verbalized eval-awareness.",
            paths.behavior_by_variant: "Primary behavior score by model/variant.",
        }
        for plot_path in paths.all_paths():
            label = plot_path.stem.replace("_", " ").title()
            description = descriptions.get(plot_path, "")
            if self.embed_plots:
                src = "data:image/png;base64," + base64.b64encode(
                    plot_path.read_bytes()
                ).decode("ascii")
            else:
                # Use a relative path so the HTML is portable when copied.
                src = str(plot_path.relative_to(self.output_dir))
            sections.append(
                f'<div class="plot"><h3>{html.escape(label)}</h3>'
                f'<img src="{html.escape(src)}" alt="{html.escape(label)}"/>'
                f"<p>{html.escape(description)}</p></div>"
            )
        return "\n".join(sections)

    def _examples_html(self, runs: Sequence[ModelRun]) -> str:
        sections: List[str] = []
        per_model = _runs_by_model(runs)
        for model_name, model_runs in per_model.items():
            sections.append(f"<h3>{html.escape(model_name)}</h3>")
            sample = sorted(
                model_runs, key=lambda run: -run.verbalized_awareness.score
            )[: self.max_examples_per_model]
            for run in sample:
                quotes = "".join(
                    f'<span class="pill">{html.escape(quote.label)}: '
                    f'"{html.escape(quote.text)}"</span>'
                    for quote in run.verbalized_awareness.quotes
                )
                sections.append(
                    "<details>"
                    f"<summary>{html.escape(run.task_id)} / "
                    f"{html.escape(run.variant_id)} "
                    f"(VEA={run.verbalized_awareness.score:.2f}, "
                    f"leakage={run.leakage_score:.2f})</summary>"
                    f"<pre>{html.escape(run.response)}</pre>"
                    f"{quotes}"
                    "</details>"
                )
        return "\n".join(sections) or "<p>No rollouts available.</p>"

    def _render_template(
        self,
        plots_html: str,
        model_table: str,
        variant_table: str,
        examples_html: str,
    ) -> str:
        try:
            jinja2 = importlib.import_module("jinja2")
        except ImportError:
            return _FALLBACK_TEMPLATE.format(
                description=html.escape(self.description),
                plots_html=plots_html,
                model_table=model_table,
                variant_table=variant_table,
                examples_html=examples_html,
            )
        env = jinja2.Environment(autoescape=True)
        template = env.from_string("""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{{ title }}</title>
<style>
  body { font-family: system-ui, -apple-system, sans-serif; margin: 2rem auto; max-width: 1100px; color: #111; }
  h1, h2 { color: #1f2937; }
  table { border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }
  th, td { padding: 0.45rem 0.7rem; border-bottom: 1px solid #e5e7eb; text-align: left; }
  th { background: #f3f4f6; }
  .plot { margin: 1.5rem 0; }
  .plot img { max-width: 100%; border: 1px solid #e5e7eb; border-radius: 6px; }
  details { margin: 0.4rem 0; }
  pre { background: #f9fafb; padding: 0.75rem; border-radius: 6px; overflow-x: auto; }
  .pill { display: inline-block; padding: 0.1rem 0.5rem; background: #eef2ff; color: #3730a3; border-radius: 999px; font-size: 0.8rem; margin-right: 0.4rem; }
</style>
</head>
<body>
<h1>{{ title }}</h1>
<p>{{ description }}</p>
<h2>Plots</h2>
{{ plots_html|safe }}
<h2>Model summary</h2>
{{ model_table|safe }}
<h2>Variant summary</h2>
{{ variant_table|safe }}
<h2>Selected rollouts</h2>
{{ examples_html|safe }}
</body>
</html>
""")
        return template.render(
            title=self.title,
            description=self.description,
            plots_html=plots_html,
            model_table=model_table,
            variant_table=variant_table,
            examples_html=examples_html,
        )


def _render_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    head = "".join(f"<th>{html.escape(label)}</th>" for label in headers)
    body_rows: List[str] = []
    for row in rows:
        body_rows.append(
            "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
        )
    return (
        "<table><thead><tr>"
        + head
        + "</tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table>"
    )


def render_dashboard(
    report: BehaviorShiftReport,
    output_dir: Path | str,
    title: str = "Eval-awareness benchmark report",
    description: str = "Reproducible eval-awareness benchmark generated by IsItBenchmark.",
    embed_plots: bool = False,
) -> DashboardPaths:
    """Convenience helper that renders a benchmark report to a directory."""
    dashboard = HTMLDashboard(
        output_dir=output_dir,
        title=title,
        description=description,
        embed_plots=embed_plots,
    )
    return dashboard.render(report)
