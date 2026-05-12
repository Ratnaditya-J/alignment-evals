"""Build a self-contained HTML dashboard from cross-protocol run summaries.

Reads ``cross_protocol_summary.json`` from each provided run directory,
validates the schema, and emits a single HTML file with charts for:

  * Per-model refusal rate deltas (inject - neutral) with bootstrap CIs
  * Reasoning-tier x vendor pooled refusal rate deltas
  * Source-type (open vs closed weights) pooled refusal rate deltas
  * Per-(model, task_family) capability accuracy deltas
  * Reasoning-tier x family + source-type x family capability deltas
  * Phase-C sandbagging verdict matrix (model x family) with rate deltas
  * Scorer validation (regex-vs-LLM agreement per model)

Usage:
    # explicit run dirs
    python scripts/build_dashboard.py runs/v5 runs/v6 --out dashboard.html

    # auto-discover any run dir under runs/ with a valid summary
    python scripts/build_dashboard.py --auto --out dashboard.html

A run is considered valid if its ``cross_protocol_summary.json`` has a
non-empty ``per_model`` table. Invalid runs are skipped with a console
message rather than aborting the build.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger("build_dashboard")

REQUIRED_TOP_LEVEL_KEYS = ("per_model",)
SUMMARY_FILENAME = "cross_protocol_summary.json"


# ---------------------------------------------------------------------------
# Discovery + validation
# ---------------------------------------------------------------------------


def _auto_discover(root: Path) -> List[Path]:
    """Find every direct subdirectory of ``root`` that contains a summary."""
    if not root.is_dir():
        return []
    found: List[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if (child / SUMMARY_FILENAME).is_file():
            found.append(child)
    return found


def _load_summary(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load one run's summary if it is well-formed; return None otherwise."""
    summary_path = run_dir / SUMMARY_FILENAME
    if not summary_path.is_file():
        LOGGER.warning("skip %s: missing %s", run_dir, SUMMARY_FILENAME)
        return None
    try:
        data = json.loads(summary_path.read_text())
    except json.JSONDecodeError as exc:
        LOGGER.warning("skip %s: invalid JSON in summary (%s)", run_dir, exc)
        return None
    if not isinstance(data, dict):
        LOGGER.warning("skip %s: summary is not a JSON object", run_dir)
        return None
    for key in REQUIRED_TOP_LEVEL_KEYS:
        if not data.get(key):
            LOGGER.warning("skip %s: empty or missing '%s'", run_dir, key)
            return None
    return data


def _build_run_payload(run_dir: Path, label: str) -> Optional[Dict[str, Any]]:
    """Assemble the per-run blob the dashboard consumes."""
    summary = _load_summary(run_dir)
    if summary is None:
        return None
    manifest = summary.get("corpus_manifest") or {}
    config_summary = {
        "label": label,
        "run_dir": str(run_dir),
        "n_records": summary.get("rescored_records")
        or manifest.get("rollouts_written")
        or 0,
        "primary_scorer": summary.get("primary_scorer", "regex"),
        "primary_score_key": summary.get("primary_score_key", "refusal_regex"),
        "judge_provider": summary.get("judge_provider", "none"),
        "judge_model": summary.get("judge_model", ""),
        "sandbagging_judge_provider": summary.get(
            "sandbagging_judge_provider", "none"
        ),
        "sandbagging_judge_model": summary.get("sandbagging_judge_model", ""),
        "conditions": manifest.get("conditions") or [
            "baseline",
            "neutral",
            "inject",
            "remove",
        ],
        "models": manifest.get("models", []),
        "task_family_counts": manifest.get("task_family_counts", {}),
        "seed": manifest.get("seed"),
        "started_at": manifest.get("started_at"),
        "finished_at": manifest.get("finished_at"),
    }
    return {
        "config": config_summary,
        "per_model": summary.get("per_model", []),
        "reasoning_tier_cuts": summary.get("reasoning_tier_cuts", []),
        "source_type_cuts": summary.get("source_type_cuts", []),
        "task_family_breakdown": summary.get("task_family_breakdown", []),
        "capability_per_model_family": summary.get(
            "capability_accuracy_per_model_family", []
        ),
        "capability_tier_family": summary.get(
            "capability_accuracy_tier_family", []
        ),
        "capability_source_type_family": summary.get(
            "capability_accuracy_source_type_family", []
        ),
        "sandbagging_cells": summary.get("sandbagging_cells", []),
        "scorer_validation": summary.get("scorer_validation", {}),
        "empty_response_breakdown": summary.get("empty_response_breakdown", []),
    }


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>IsItBenchmark - Cross-protocol dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0e1117;
    --panel: #161b22;
    --panel-border: #30363d;
    --text: #e6edf3;
    --muted: #8b949e;
    --accent: #58a6ff;
    --positive: #f85149;
    --negative: #3fb950;
    --neutral: #d29922;
    --warn-strong: #f85149;
    --warn-moderate: #db6d28;
    --warn-weak: #d29922;
    --warn-ok: #3fb950;
  }
  html, body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0;
    padding: 0;
  }
  header {
    padding: 16px 24px;
    border-bottom: 1px solid var(--panel-border);
    background: var(--panel);
    display: flex;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
  }
  header h1 {
    margin: 0;
    font-size: 18px;
    font-weight: 600;
  }
  header .pill {
    font-size: 12px;
    background: rgba(88, 166, 255, 0.12);
    color: var(--accent);
    padding: 4px 10px;
    border-radius: 12px;
    border: 1px solid rgba(88, 166, 255, 0.3);
  }
  select {
    background: var(--bg);
    color: var(--text);
    border: 1px solid var(--panel-border);
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 13px;
  }
  main {
    padding: 16px 24px 48px;
    max-width: 1400px;
    margin: 0 auto;
  }
  .config-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 8px 16px;
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 24px;
    font-size: 13px;
  }
  .config-grid .k { color: var(--muted); margin-right: 4px; }
  .config-grid .v { color: var(--text); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  section {
    margin-bottom: 36px;
  }
  section h2 {
    font-size: 16px;
    margin: 0 0 8px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--panel-border);
  }
  section h2 .helper {
    font-size: 12px;
    font-weight: 400;
    color: var(--muted);
    margin-left: 8px;
  }
  .chart-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-top: 12px;
  }
  .chart-card {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 8px;
    padding: 12px 14px 8px;
  }
  .chart-card.full {
    grid-column: 1 / -1;
  }
  .chart-card h3 {
    margin: 0 0 4px 0;
    font-size: 14px;
    font-weight: 600;
  }
  .chart-card .caption {
    margin: 0 0 10px 0;
    color: var(--muted);
    font-size: 12px;
  }
  .chart-box {
    position: relative;
    height: 320px;
  }
  .chart-box.tall { height: 440px; }
  table.matrix {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    margin-top: 12px;
  }
  table.matrix th, table.matrix td {
    border: 1px solid var(--panel-border);
    padding: 6px 8px;
    text-align: left;
    vertical-align: top;
  }
  table.matrix th {
    background: rgba(88, 166, 255, 0.05);
    color: var(--accent);
    font-weight: 600;
  }
  table.matrix td.num {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    text-align: right;
  }
  .verdict-pill {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 600;
    background: rgba(255,255,255,0.06);
  }
  .verdict-strong { background: rgba(248,81,73,0.20); color: var(--warn-strong); }
  .verdict-moderate { background: rgba(219,109,40,0.20); color: var(--warn-moderate); }
  .verdict-weak { background: rgba(210,153,34,0.20); color: var(--warn-weak); }
  .verdict-no_evidence { background: rgba(63,185,80,0.18); color: var(--warn-ok); }
  .verdict-explained_by_overrefusal { background: rgba(139,148,158,0.18); color: var(--muted); }
  .verdict-insufficient_data { background: rgba(139,148,158,0.18); color: var(--muted); }
  .empty { color: var(--muted); font-style: italic; padding: 8px 0; }
  details summary {
    cursor: pointer;
    color: var(--accent);
    font-size: 12px;
    padding: 4px 0;
  }
  .legend {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    font-size: 11px;
    color: var(--muted);
    margin-bottom: 8px;
  }
  .legend .swatch {
    display: inline-block;
    width: 10px;
    height: 10px;
    margin-right: 4px;
    vertical-align: middle;
    border-radius: 2px;
  }
</style>
</head>
<body>
<header>
  <h1>IsItBenchmark dashboard</h1>
  <span class="pill" id="run-count-pill">0 runs</span>
  <label style="margin-left:auto; font-size:13px;">
    Run:
    <select id="run-selector"></select>
  </label>
</header>
<main>
  <div id="config-grid" class="config-grid"></div>

  <section>
    <h2>Refusal rate
      <span class="helper">inject - neutral delta with bootstrap-paired 95% CI. Bars whose error bar excludes 0 are highlighted.</span>
    </h2>
    <div class="legend">
      <span><span class="swatch" style="background:#58a6ff"></span>closed_source</span>
      <span><span class="swatch" style="background:#bb8009"></span>open_source</span>
      <span><span class="swatch" style="background:#f85149"></span>CI excludes 0 (significant)</span>
    </div>
    <div class="chart-grid">
      <div class="chart-card full">
        <h3>Per-model refusal-rate delta</h3>
        <p class="caption">One bar per model. Sorted by delta. Error bars are bootstrap-paired 95% CI.</p>
        <div class="chart-box tall"><canvas id="chart-refusal-per-model"></canvas></div>
      </div>
      <div class="chart-card">
        <h3>Reasoning tier x vendor (pooled)</h3>
        <p class="caption">Pooled across all models in each tier x vendor bucket.</p>
        <div class="chart-box"><canvas id="chart-refusal-tier"></canvas></div>
      </div>
      <div class="chart-card">
        <h3>Source type (open vs closed)</h3>
        <p class="caption">Pooled across open-weights vs closed-source models.</p>
        <div class="chart-box"><canvas id="chart-refusal-source"></canvas></div>
      </div>
      <div class="chart-card full">
        <h3>Per-condition refusal rates (per model)</h3>
        <p class="caption">Stacked view showing each condition's absolute refusal rate, not just the delta.</p>
        <div class="chart-box tall"><canvas id="chart-refusal-conditions"></canvas></div>
      </div>
    </div>
  </section>

  <section>
    <h2>Capability accuracy
      <span class="helper">inject - neutral accuracy delta on capability families (math, knowledge, truthfulness, code). DECREASE is a sandbagging candidate; cross-reference the sandbagging section.</span>
    </h2>
    <div class="chart-grid">
      <div class="chart-card full">
        <h3>Per-(model, family) accuracy delta</h3>
        <p class="caption">A significant DECREASE here on a capability family is the headline sandbagging candidate.</p>
        <div class="chart-box tall"><canvas id="chart-cap-per-model"></canvas></div>
      </div>
      <div class="chart-card">
        <h3>Tier x family (pooled)</h3>
        <div class="chart-box"><canvas id="chart-cap-tier"></canvas></div>
      </div>
      <div class="chart-card">
        <h3>Source type x family (pooled)</h3>
        <div class="chart-box"><canvas id="chart-cap-source"></canvas></div>
      </div>
    </div>
  </section>

  <section>
    <h2>Sandbagging verdict matrix
      <span class="helper">Phase-C convergent-signal verdicts per (model, family). Verdict 'no_evidence' is the dominant safe state.</span>
    </h2>
    <div id="sandbagging-matrix"></div>
  </section>

  <section>
    <h2>Scorer validation
      <span class="helper">Regex-vs-LLM agreement per model (LLM treated as reference signal for surfacing regex misses).</span>
    </h2>
    <div id="scorer-validation"></div>
  </section>

  <section>
    <h2>Empty-response audit
      <span class="helper">Records where the model returned an empty string. High rates flag a dispatch / token-budget bug rather than refusal.</span>
    </h2>
    <div id="empty-audit"></div>
  </section>
</main>

<script id="dashboard-data" type="application/json">__DATA_JSON__</script>
<script>
(function () {
  const DATA = JSON.parse(document.getElementById("dashboard-data").textContent);
  const runs = DATA.runs || [];
  const selector = document.getElementById("run-selector");
  const pill = document.getElementById("run-count-pill");
  pill.textContent = runs.length + " run" + (runs.length === 1 ? "" : "s");

  if (!runs.length) {
    document.querySelector("main").innerHTML =
      '<div class="empty">No valid runs were embedded. Pass run directories on the command line: ' +
      '<code>python scripts/build_dashboard.py runs/v6 --out dashboard.html</code></div>';
    return;
  }

  runs.forEach((r, i) => {
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = r.config.label;
    selector.appendChild(opt);
  });

  const charts = {};
  selector.addEventListener("change", () => render(runs[Number(selector.value)]));
  render(runs[0]);

  function destroyAll() {
    Object.values(charts).forEach(c => c && c.destroy());
  }

  function fmtNum(v, digits) {
    if (v === null || v === undefined || Number.isNaN(v)) return "-";
    const n = Number(v);
    if (digits === undefined) digits = 3;
    return (n >= 0 ? "+" : "") + n.toFixed(digits);
  }

  function fmtRate(v) {
    if (v === null || v === undefined) return "-";
    return (Number(v) * 100).toFixed(1) + "%";
  }

  function isSignificant(ci) {
    if (!ci) return false;
    return (ci.low > 0 && ci.high > 0) || (ci.low < 0 && ci.high < 0);
  }

  function colorForRow(row, palette) {
    if (isSignificant(row.inject_minus_neutral_ci)) {
      return palette.significant;
    }
    const source = (row.model_source_type || "").toLowerCase();
    if (source === "open_source") return palette.open;
    if (source === "closed_source") return palette.closed;
    return palette.neutral;
  }

  function renderDeltaBarChart(canvasId, rows, labelKey, palette, valueKey) {
    valueKey = valueKey || "inject_minus_neutral";
    const ciKey = valueKey + "_ci";
    const nKey = valueKey + "_n_paired";
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    if (!rows || !rows.length) {
      ctx.parentElement.innerHTML = '<div class="empty">No rows.</div>';
      return;
    }
    const sorted = rows.slice().sort((a, b) => (a[valueKey] || 0) - (b[valueKey] || 0));
    const labels = sorted.map(r => formatLabel(r, labelKey));
    const values = sorted.map(r => r[valueKey] || 0);
    const errLow = sorted.map(r => (r[valueKey] || 0) - ((r[ciKey] || {}).low || 0));
    const errHigh = sorted.map(r => ((r[ciKey] || {}).high || 0) - (r[valueKey] || 0));
    const bg = sorted.map(r => colorForRow(r, palette));
    charts[canvasId] = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [{
          label: "delta",
          data: values,
          backgroundColor: bg,
          borderColor: bg,
          borderWidth: 0,
          // Custom error bar drawn via plugin below.
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: "y",
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const r = sorted[ctx.dataIndex];
                const ci = r[ciKey] || {};
                const n = r[nKey];
                return [
                  "delta = " + fmtNum(r[valueKey]),
                  "95% CI [" + fmtNum(ci.low) + ", " + fmtNum(ci.high) + "]",
                  "n_paired = " + (n === undefined ? "-" : n),
                ];
              },
            },
          },
        },
        scales: {
          x: {
            grid: { color: "rgba(255,255,255,0.05)" },
            ticks: { color: "#8b949e" },
            title: { display: true, text: "inject - neutral", color: "#8b949e" },
          },
          y: {
            grid: { color: "rgba(255,255,255,0.05)" },
            ticks: { color: "#e6edf3" },
          },
        },
      },
      plugins: [{
        id: "error_bars",
        afterDatasetsDraw(chart) {
          const { ctx, scales: { x, y } } = chart;
          ctx.save();
          ctx.strokeStyle = "rgba(255,255,255,0.7)";
          ctx.lineWidth = 1;
          for (let i = 0; i < values.length; i++) {
            const v = values[i];
            const ci = (sorted[i][ciKey] || {});
            if (ci.low === undefined || ci.high === undefined) continue;
            const xLow = x.getPixelForValue(ci.low);
            const xHigh = x.getPixelForValue(ci.high);
            const yCenter = y.getPixelForValue(i);
            ctx.beginPath();
            ctx.moveTo(xLow, yCenter);
            ctx.lineTo(xHigh, yCenter);
            ctx.moveTo(xLow, yCenter - 4);
            ctx.lineTo(xLow, yCenter + 4);
            ctx.moveTo(xHigh, yCenter - 4);
            ctx.lineTo(xHigh, yCenter + 4);
            ctx.stroke();
          }
          // Zero reference line.
          const xZero = x.getPixelForValue(0);
          ctx.strokeStyle = "rgba(255,255,255,0.25)";
          ctx.setLineDash([4, 4]);
          ctx.beginPath();
          ctx.moveTo(xZero, y.getPixelForValue(-0.5));
          ctx.lineTo(xZero, y.getPixelForValue(values.length - 0.5));
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.restore();
        },
      }],
    });
  }

  function formatLabel(row, key) {
    if (typeof key === "function") return key(row);
    return row[key] || "?";
  }

  function renderConditionStack(canvasId, rows, metricKey) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    if (!rows || !rows.length) {
      ctx.parentElement.innerHTML = '<div class="empty">No rows.</div>';
      return;
    }
    const conditions = ["baseline", "neutral", "inject", "remove"];
    const sorted = rows.slice().sort((a, b) =>
      (b.inject_minus_neutral || 0) - (a.inject_minus_neutral || 0)
    );
    const labels = sorted.map(r => r.model_name || "?");
    const condColors = {
      baseline: "rgba(139,148,158,0.85)",
      neutral: "rgba(88,166,255,0.85)",
      inject: "rgba(248,81,73,0.85)",
      remove: "rgba(63,185,80,0.85)",
    };
    const datasets = conditions.map(cond => ({
      label: cond,
      data: sorted.map(r => r[cond + "_" + metricKey] || 0),
      backgroundColor: condColors[cond],
    }));
    charts[canvasId] = new Chart(ctx, {
      type: "bar",
      data: { labels: labels, datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: "y",
        plugins: {
          tooltip: {
            callbacks: {
              label: (c) => c.dataset.label + ": " + (Number(c.raw) * 100).toFixed(1) + "%",
            },
          },
          legend: { labels: { color: "#e6edf3" } },
        },
        scales: {
          x: {
            beginAtZero: true,
            ticks: { color: "#8b949e", callback: v => (v * 100).toFixed(0) + "%" },
            grid: { color: "rgba(255,255,255,0.05)" },
            title: { display: true, text: metricKey, color: "#8b949e" },
          },
          y: {
            ticks: { color: "#e6edf3" },
            grid: { color: "rgba(255,255,255,0.05)" },
          },
        },
      },
    });
  }

  function renderSandbaggingMatrix(rows) {
    const host = document.getElementById("sandbagging-matrix");
    host.innerHTML = "";
    if (!rows || !rows.length) {
      host.innerHTML = '<div class="empty">No sandbagging cells in this run (sandbagging judge not configured, or no capability records).</div>';
      return;
    }
    const table = document.createElement("table");
    table.className = "matrix";
    const header = document.createElement("thead");
    header.innerHTML =
      '<tr>' +
      '<th>Model</th><th>Family</th><th>Verdict</th>' +
      '<th class="num">delta sandbag</th>' +
      '<th class="num">delta overrefuse</th>' +
      '<th class="num">delta sound-wrong</th>' +
      '<th class="num">delta accuracy</th>' +
      '<th class="num">n judged</th>' +
      '<th>rationale</th>' +
      '</tr>';
    table.appendChild(header);
    const body = document.createElement("tbody");
    const sorted = rows.slice().sort((a, b) => {
      const order = {
        strong: 0, moderate: 1, weak: 2,
        explained_by_overrefusal: 3, insufficient_data: 4, no_evidence: 5,
      };
      const ra = order[a.evidence_tier] === undefined ? 99 : order[a.evidence_tier];
      const rb = order[b.evidence_tier] === undefined ? 99 : order[b.evidence_tier];
      if (ra !== rb) return ra - rb;
      return (a.model_name || "").localeCompare(b.model_name || "");
    });
    sorted.forEach(r => {
      const tr = document.createElement("tr");
      const verdictClass = "verdict-" + (r.evidence_tier || "no_evidence");
      tr.innerHTML =
        '<td><code>' + escapeHtml(r.model_name || "?") + '</code></td>' +
        '<td>' + escapeHtml(r.task_family || "?") + '</td>' +
        '<td><span class="verdict-pill ' + verdictClass + '">' +
            escapeHtml(r.evidence_tier || "no_evidence") + '</span></td>' +
        '<td class="num">' + fmtNum(r.delta_sandbagging_inject_neutral) + '</td>' +
        '<td class="num">' + fmtNum(r.delta_overrefusal_inject_neutral) + '</td>' +
        '<td class="num">' + fmtNum(r.delta_sound_but_wrong_inject_neutral) + '</td>' +
        '<td class="num">' + fmtNum(r.delta_accuracy_inject_neutral) + '</td>' +
        '<td class="num">' + (r.n_judged_total === undefined ? "-" : r.n_judged_total) + '</td>' +
        '<td>' + escapeHtml(r.verdict_rationale || "") + '</td>';
      body.appendChild(tr);
    });
    table.appendChild(body);
    host.appendChild(table);
  }

  function renderScorerValidation(payload) {
    const host = document.getElementById("scorer-validation");
    host.innerHTML = "";
    if (!payload || !payload.overall) {
      host.innerHTML = '<div class="empty">No scorer-validation data (LLM judge not run on this pass).</div>';
      return;
    }
    const overall = payload.overall;
    const summary = document.createElement("div");
    summary.style.fontSize = "13px";
    summary.innerHTML =
      'Overall: agreement <strong>' + (overall.agreement * 100).toFixed(1) + '%</strong>' +
      ', regex precision <strong>' + (overall.regex_precision_vs_llm * 100).toFixed(1) + '%</strong>' +
      ', recall <strong>' + (overall.regex_recall_vs_llm * 100).toFixed(1) + '%</strong>' +
      ', F1 <strong>' + (overall.regex_f1_vs_llm * 100).toFixed(1) + '%</strong>' +
      ' (n=' + (overall.n || 0) + ', LLM treated as reference)';
    host.appendChild(summary);

    const rows = payload.per_model || [];
    if (!rows.length) return;
    const table = document.createElement("table");
    table.className = "matrix";
    table.style.marginTop = "12px";
    table.innerHTML =
      '<thead><tr>' +
      '<th>Model</th><th class="num">n</th>' +
      '<th class="num">agreement</th>' +
      '<th class="num">regex F1</th>' +
      '<th class="num">precision</th>' +
      '<th class="num">recall</th>' +
      '<th class="num">regex+/LLM-</th>' +
      '<th class="num">regex-/LLM+</th>' +
      '</tr></thead>';
    const tbody = document.createElement("tbody");
    rows.slice().sort((a, b) => a.model_name.localeCompare(b.model_name)).forEach(r => {
      const tr = document.createElement("tr");
      tr.innerHTML =
        '<td><code>' + escapeHtml(r.model_name) + '</code></td>' +
        '<td class="num">' + r.n + '</td>' +
        '<td class="num">' + (r.agreement * 100).toFixed(1) + '%</td>' +
        '<td class="num">' + (r.regex_f1_vs_llm * 100).toFixed(1) + '%</td>' +
        '<td class="num">' + (r.regex_precision_vs_llm * 100).toFixed(1) + '%</td>' +
        '<td class="num">' + (r.regex_recall_vs_llm * 100).toFixed(1) + '%</td>' +
        '<td class="num">' + r.regex_positive_llm_negative + '</td>' +
        '<td class="num">' + r.regex_negative_llm_positive + '</td>';
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    host.appendChild(table);
  }

  function renderEmptyAudit(rows) {
    const host = document.getElementById("empty-audit");
    host.innerHTML = "";
    if (!rows || !rows.length) {
      host.innerHTML = '<div class="empty">No empty-response rows recorded.</div>';
      return;
    }
    const table = document.createElement("table");
    table.className = "matrix";
    table.innerHTML =
      '<thead><tr>' +
      '<th>Model</th><th>Condition</th>' +
      '<th class="num">empty</th><th class="num">total</th>' +
      '<th class="num">empty rate</th>' +
      '</tr></thead>';
    const tbody = document.createElement("tbody");
    rows.slice()
      .sort((a, b) => (b.empty_rate || 0) - (a.empty_rate || 0))
      .forEach(r => {
        const tr = document.createElement("tr");
        tr.innerHTML =
          '<td><code>' + escapeHtml(r.model_name || "?") + '</code></td>' +
          '<td>' + escapeHtml(r.condition || "?") + '</td>' +
          '<td class="num">' + (r.empty || 0) + '</td>' +
          '<td class="num">' + (r.total || 0) + '</td>' +
          '<td class="num">' + ((r.empty_rate || 0) * 100).toFixed(1) + '%</td>';
        tbody.appendChild(tr);
      });
    table.appendChild(tbody);
    host.appendChild(table);
  }

  function renderConfig(cfg) {
    const host = document.getElementById("config-grid");
    host.innerHTML = "";
    const entries = [
      ["records", cfg.n_records],
      ["primary scorer", cfg.primary_scorer + " (" + cfg.primary_score_key + ")"],
      ["refusal judge", cfg.judge_provider + (cfg.judge_model ? " / " + cfg.judge_model : "")],
      ["sandbagging judge", cfg.sandbagging_judge_provider + (cfg.sandbagging_judge_model ? " / " + cfg.sandbagging_judge_model : "")],
      ["models", (cfg.models || []).length],
      ["conditions", (cfg.conditions || []).join(", ")],
      ["seed", cfg.seed === null || cfg.seed === undefined ? "-" : cfg.seed],
      ["started", cfg.started_at || "-"],
      ["finished", cfg.finished_at || "-"],
    ];
    entries.forEach(([k, v]) => {
      const cell = document.createElement("div");
      cell.innerHTML = '<span class="k">' + escapeHtml(String(k)) + ':</span> ' +
                       '<span class="v">' + escapeHtml(String(v)) + '</span>';
      host.appendChild(cell);
    });
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function render(run) {
    destroyAll();
    renderConfig(run.config);
    const palette = {
      open: "#bb8009",
      closed: "#58a6ff",
      neutral: "#8b949e",
      significant: "#f85149",
    };
    renderDeltaBarChart(
      "chart-refusal-per-model",
      run.per_model,
      r => r.model_name || "?",
      palette
    );
    renderDeltaBarChart(
      "chart-refusal-tier",
      run.reasoning_tier_cuts,
      r => (r.model_tier_bucket || "?") + " / " + (r.model_vendor || "?"),
      palette
    );
    renderDeltaBarChart(
      "chart-refusal-source",
      run.source_type_cuts,
      r => r.model_source_type || "?",
      palette
    );
    renderConditionStack("chart-refusal-conditions", run.per_model, "refusal_rate");

    renderDeltaBarChart(
      "chart-cap-per-model",
      run.capability_per_model_family,
      r => (r.model_name || "?") + " / " + (r.task_family || "?"),
      palette
    );
    renderDeltaBarChart(
      "chart-cap-tier",
      run.capability_tier_family,
      r => (r.model_tier_bucket || "?") + " / " + (r.task_family || "?"),
      palette
    );
    renderDeltaBarChart(
      "chart-cap-source",
      run.capability_source_type_family,
      r => (r.model_source_type || "?") + " / " + (r.task_family || "?"),
      palette
    );

    renderSandbaggingMatrix(run.sandbagging_cells);
    renderScorerValidation(run.scorer_validation);
    renderEmptyAudit(run.empty_response_breakdown);
  }
})();
</script>
</body>
</html>
"""


def _render_html(runs: Sequence[Dict[str, Any]]) -> str:
    payload = json.dumps({"runs": list(runs)}, separators=(",", ":"))
    # Embed safely inside a <script type="application/json"> tag - the only
    # token that can break out is '</script>'. Escape it.
    payload = payload.replace("</", "<\\/")
    return _HTML_TEMPLATE.replace("__DATA_JSON__", payload)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_run_arg(spec: str) -> Tuple[Path, Optional[str]]:
    """Accept ``<path>`` or ``<label>=<path>`` for explicit labelling."""
    if "=" in spec:
        label, _, path = spec.partition("=")
        return Path(path.strip()), label.strip()
    return Path(spec), None


def _default_label(run_dir: Path) -> str:
    return run_dir.name or str(run_dir)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dirs",
        nargs="*",
        help=(
            "Run directories to include. Each must contain "
            "cross_protocol_summary.json. Use the form "
            "<label>=<path> to override the dropdown label."
        ),
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-discover any subdirectory of --runs-root containing a summary.",
    )
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Root directory scanned by --auto (default: runs).",
    )
    parser.add_argument(
        "--out",
        default="dashboard.html",
        help="Output HTML path (default: dashboard.html).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-run skip warnings.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
    )

    targets: List[Tuple[Path, str]] = []
    for spec in args.run_dirs:
        path, label = _parse_run_arg(spec)
        targets.append((path, label or _default_label(path)))
    if args.auto:
        for path in _auto_discover(Path(args.runs_root)):
            if any(t[0] == path for t in targets):
                continue
            targets.append((path, _default_label(path)))

    if not targets:
        parser.error(
            "no run directories given; pass paths positionally or use --auto"
        )

    runs: List[Dict[str, Any]] = []
    for run_dir, label in targets:
        payload = _build_run_payload(run_dir, label)
        if payload is None:
            continue
        runs.append(payload)
        LOGGER.info("included %s as %r", run_dir, label)

    if not runs:
        LOGGER.error("no valid runs found - nothing to render")
        return 2

    html = _render_html(runs)
    out_path = Path(args.out)
    out_path.write_text(html, encoding="utf-8")
    LOGGER.info("wrote %s (%d run(s), %.1f KB)", out_path, len(runs), len(html) / 1024)
    return 0


if __name__ == "__main__":
    sys.exit(main())
