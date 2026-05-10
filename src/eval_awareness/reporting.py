"""CSV/HTML reporting helpers for benchmark artifacts."""

import csv
from pathlib import Path
from typing import Dict, Sequence

from .runner import BehaviorShiftReport


class ReportExporter:
    """Export benchmark summaries to CSV and HTML."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_model_summary_csv(self, report: BehaviorShiftReport) -> Path:
        path = self.output_dir / "model_summary.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "model",
                    "runs",
                    "avg_leakage_score",
                    "avg_verbalized_awareness_score",
                    "obvious_eval_vea_rate",
                    "deployment_vea_rate",
                ],
            )
            writer.writeheader()
            for model, summary in report.by_model.items():
                writer.writerow({"model": model, **_flatten_scalar_summary(summary)})
        return path

    def export_html(self, report: BehaviorShiftReport) -> Path:
        path = self.output_dir / "benchmark_report.html"
        rows = []
        for model, summary in report.by_model.items():
            rows.append(
                "<tr>"
                f"<td>{model}</td>"
                f"<td>{summary.get('runs', 0)}</td>"
                f"<td>{summary.get('obvious_eval_vea_rate', 0)}</td>"
                f"<td>{summary.get('deployment_vea_rate', 0)}</td>"
                f"<td>{summary.get('avg_leakage_score', 0)}</td>"
                "</tr>"
            )
        html = """
<!doctype html>
<html><head><meta charset="utf-8"><title>Eval-awareness report</title></head>
<body>
<h1>Evaluation-awareness benchmark report</h1>
<table border="1" cellspacing="0" cellpadding="4">
<thead><tr><th>Model</th><th>Runs</th><th>Eval VEA</th><th>Deployment VEA</th><th>Leakage</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
</body></html>
""".format(rows="\n".join(rows))
        path.write_text(html, encoding="utf-8")
        return path


def _flatten_scalar_summary(summary: Dict[str, object]) -> Dict[str, object]:
    return {
        key: value
        for key, value in summary.items()
        if isinstance(value, (str, int, float, bool))
    }
