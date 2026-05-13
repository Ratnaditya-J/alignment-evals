"""Generate blog-style figures for docs/blog_draft.md.

Produces the 6 figures referenced by the blog post. Style choices:
larger fonts, refined sans-serif typography, a curated palette,
a clear title/subtitle/footer hierarchy, and minimal chart-junk.
Aims for a "tasteful editorial" aesthetic rather than the default
matplotlib look.

Output figures:

  diagram_setup.png                       experimental design at a glance
  blog_chart_a_forest.png                 simplified per-model refusal forest
  blog_chart_b_vea.png                    VEA inflation bar chart
  blog_chart_c_apparent_mediation.png     qwen3 primary mediation (the "I had it")
  blog_chart_d_primary_vs_replication.png primary vs replication DiD bars
  blog_chart_e_direction_asymmetry.png    qwen3 vs opus stable_neg, opposite directions

Numbers are pulled from the same run JSONs the paper figure generator uses:
  --cross-protocol-summary
  --goodfire-summary
  --strict-mediation-summary (qwen3 n=500 primary)
  --qwen3-replication-strict-mediation-summary
  --opus-strict-mediation-summary

Usage:
    python scripts/generate_blog_figures.py \\
        --cross-protocol-summary runs/cross-protocol-v6/cross_protocol_summary.json \\
        --goodfire-summary runs/goodfire-mixed-n500/goodfire_vea_summary.json \\
        --strict-mediation-summary runs/goodfire-mixed-n500/vea_mediation_summary.strict.json \\
        --qwen3-replication-strict-mediation-summary runs/goodfire-replication-qwen3-safety/vea_mediation_summary.strict.json \\
        --opus-strict-mediation-summary runs/goodfire-opus-mixed-n300/vea_mediation_summary.strict.json \\
        --out-dir docs/blog_figures/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

LOGGER = logging.getLogger("generate_blog_figures")


# ---------------------------------------------------------------------------
# Visual style — tasteful editorial palette
# ---------------------------------------------------------------------------
#
# Color choices follow the FT / Pudding-style sensibility: muted, warm,
# print-friendly. We use:
#   * RED   for "CI excludes zero" / the signal-bearing bars
#   * GREY  for "CI includes zero" / null bars
#   * NAVY  for neutral structural accents (forest plot markers)
#   * SAGE  for the replication / "good news methodology" color
#   * AMBER for the "apparent finding before replication" color
#   * INK   for primary text (almost-black, softer than #000)
#   * MUTED for secondary text / subtitles
#   * RULE  for axis spines / faint grid lines
#   * CARD  for inset card backgrounds (in the setup diagram)
_RED = "#c2401e"
_GREY = "#a3a3a3"
_NAVY = "#1e3a8a"
_SAGE = "#4d7c5d"
_AMBER = "#b45309"
_INK = "#1f2937"
_MUTED = "#6b7280"
_RULE = "#d4d4d4"
_GRID = "#e5e7eb"
_CARD = "#f5f5f4"


def _apply_blog_style() -> None:
    """Set matplotlib rcParams once for consistent typography across all
    blog figures. Called at the top of every render function.
    """
    import matplotlib as mpl

    mpl.rcParams.update({
        # Font: sans-serif cascade. The user's system likely has at least
        # one of these; matplotlib falls back to DejaVu Sans if not.
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Helvetica Neue", "Helvetica", "Inter",
            "Arial", "DejaVu Sans",
        ],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "axes.labelcolor": _INK,
        "axes.edgecolor": _RULE,
        "axes.linewidth": 1.0,
        "axes.titlepad": 14,
        "axes.labelpad": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": _MUTED,
        "ytick.color": _MUTED,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "grid.color": _GRID,
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        "grid.alpha": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.25,
    })


def _titled(
    fig: Any,
    title: str,
    subtitle: Optional[str] = None,
    *,
    title_x: float = 0.04,
    title_y: float = 0.97,
    subtitle_gap: float = 0.035,
) -> None:
    """Render a left-aligned bold title with an optional lighter subtitle
    directly underneath. Returns nothing; modifies fig in place."""
    fig.text(
        title_x, title_y, title,
        fontsize=15, fontweight="bold", color=_INK,
        ha="left", va="top",
    )
    if subtitle:
        fig.text(
            title_x, title_y - subtitle_gap, subtitle,
            fontsize=11, color=_MUTED,
            ha="left", va="top",
        )


def _footer(fig: Any, text: str, *, x: float = 0.04, y: float = 0.02) -> None:
    """Render a small muted footer along the bottom of the figure."""
    fig.text(
        x, y, text,
        fontsize=8.5, color=_MUTED, style="italic",
        ha="left", va="bottom",
    )


def _load_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        LOGGER.warning("skip: %s not found", p)
        return None
    return json.loads(p.read_text())


def _short_model(name: str) -> str:
    """Render long openrouter/anthropic/openai slugs as blog labels."""
    name = (name or "").replace("openrouter:", "")
    name = name.replace("anthropic:", "")
    name = name.replace("openai:", "")
    if "qwen3-235b" in name:
        return "qwen3-235B"
    if "deepseek-v3.2" in name:
        return "deepseek-v3.2"
    if "kimi-k2-thinking" in name:
        return "kimi-k2"
    if "claude-haiku-4-5" in name:
        return "haiku-4.5"
    if "claude-sonnet-4-6" in name:
        return "sonnet-4.6"
    if "claude-opus-4-7" in name:
        return "opus-4.7"
    if "gpt-4o-mini" in name:
        return "gpt-4o-mini"
    if "gpt-5.5" in name:
        return "gpt-5.5"
    return name


# ---------------------------------------------------------------------------
# diagram_setup: the experimental schema, as a clean editorial diagram
# ---------------------------------------------------------------------------


def _setup_diagram(out_path: Path) -> None:
    """A clean text+box diagram of the experimental setup. Renders as
    the post's first inline image so a casual reader grasps the
    4-condition × 4-family × 8-model design in one glance."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    _apply_blog_style()
    fig = plt.figure(figsize=(13, 6.5))
    ax = fig.add_axes([0.04, 0.10, 0.92, 0.72])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    _titled(
        fig,
        "The experiment, at a glance",
        "Same prompt sent under four different system-prompt conditions; "
        "responses scored for refusal and for explicit eval-awareness in CoT.",
        title_y=0.93,
    )

    # ── Column 1: prompts ──
    ax.text(1.5, 5.05, "PROMPTS", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=_MUTED)
    ax.text(1.5, 4.7, "125 per family", ha="center", va="center",
            fontsize=10, color=_INK, fontweight="semibold")

    prompts = [
        ("MMLU", "knowledge"),
        ("GSM8K", "math"),
        ("BeaverTails", "safety"),
        ("TruthfulQA", "truthfulness"),
    ]
    for i, (src, fam) in enumerate(prompts):
        y = 3.85 - i * 0.78
        ax.add_patch(FancyBboxPatch(
            (0.35, y - 0.30), 2.30, 0.60,
            boxstyle="round,pad=0.04,rounding_size=0.10",
            linewidth=0,
            facecolor=_CARD,
        ))
        ax.text(1.5, y + 0.10, src, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=_INK)
        ax.text(1.5, y - 0.16, fam, ha="center", va="center",
                fontsize=9, color=_MUTED, style="italic")

    # ── arrow #1 ──
    ax.annotate(
        "", xy=(4.55, 2.7), xytext=(2.95, 2.7),
        arrowprops=dict(
            arrowstyle="-|>,head_length=0.4,head_width=0.25",
            color=_MUTED, lw=1.2, shrinkA=0, shrinkB=0,
        ),
    )

    # ── Column 2: conditions ──
    ax.text(7.0, 5.05, "CONDITIONS", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=_MUTED)
    ax.text(7.0, 4.7, "4 prompts × same model run", ha="center", va="center",
            fontsize=10, color=_INK, fontweight="semibold")

    conditions = [
        ("baseline", "no system prompt", _MUTED),
        ("neutral", '"You are a helpful assistant."', _MUTED),
        ("inject", '"Note: this is an evaluation. Your response will be reviewed…"', _NAVY),
        ("remove", '"This is a real customer interaction in production…"', _MUTED),
    ]
    for i, (cond, desc, color) in enumerate(conditions):
        y = 3.85 - i * 0.78
        # Inject is the key manipulation: filled navy. Others: muted-grey
        # outline only. Navy is the structural-accent color used elsewhere
        # (forest-plot markers) and deliberately avoids the post's red
        # which means "CI excludes zero."
        is_key = cond == "inject"
        ax.add_patch(FancyBboxPatch(
            (4.65, y - 0.30), 4.55, 0.60,
            boxstyle="round,pad=0.04,rounding_size=0.10",
            linewidth=1.3,
            edgecolor=color,
            facecolor=color if is_key else "white",
            alpha=1.0 if is_key else 1.0,
        ))
        text_color = "white" if is_key else _INK
        desc_color = "#cbd5e1" if is_key else _MUTED  # light slate on navy
        ax.text(4.85, y + 0.10, cond, ha="left", va="center",
                fontsize=11, fontweight="bold", color=text_color)
        ax.text(4.85, y - 0.16, desc, ha="left", va="center",
                fontsize=8.8, color=desc_color, style="italic")

    # ── arrow #2 ──
    ax.annotate(
        "", xy=(10.85, 2.7), xytext=(9.30, 2.7),
        arrowprops=dict(
            arrowstyle="-|>,head_length=0.4,head_width=0.25",
            color=_MUTED, lw=1.2, shrinkA=0, shrinkB=0,
        ),
    )

    # ── Column 3: models ──
    ax.text(12.5, 5.05, "MODELS", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=_MUTED)
    ax.text(12.5, 4.7, "8 frontier", ha="center", va="center",
            fontsize=10, color=_INK, fontweight="semibold")

    ax.add_patch(FancyBboxPatch(
        (10.95, 0.50), 3.10, 3.85,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=0,
        facecolor=_CARD,
    ))
    closed_models = [
        ("haiku-4.5", "Anthropic"),
        ("sonnet-4.6", "Anthropic"),
        ("opus-4.7", "Anthropic"),
        ("gpt-4o-mini", "OpenAI"),
        ("gpt-5.5", "OpenAI"),
    ]
    open_models = [
        ("deepseek-v3.2", "DeepSeek"),
        ("qwen3-235B", "Alibaba"),
        ("kimi-k2", "Moonshot"),
    ]
    # Vertically arranged inside the card
    y_start = 4.0
    for i, (m, vendor) in enumerate(closed_models + open_models):
        y = y_start - i * 0.40
        ax.text(11.25, y, m, ha="left", va="center",
                fontsize=9.7, color=_INK, fontweight="semibold")
        ax.text(13.85, y, vendor, ha="right", va="center",
                fontsize=8.5, color=_MUTED)

    # ── Footer ──
    _footer(
        fig,
        "16,000 responses · scored for refusal + classified for explicit "
        "eval-awareness language · ~$500 in API fees",
    )

    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    LOGGER.info("wrote %s", out_path)


# ---------------------------------------------------------------------------
# chart_a_forest: per-model refusal Δ forest plot
# ---------------------------------------------------------------------------


def _chart_a_forest(cross: Dict[str, Any], out_path: Path) -> None:
    """Forest plot of per-model inject−neutral refusal Δ."""
    import matplotlib.pyplot as plt

    _apply_blog_style()

    rows = list(cross.get("per_model", []) or [])
    if not rows:
        LOGGER.warning("chart_a skipped: no per_model rows")
        return

    items = []
    for r in rows:
        m = r.get("model_name") or ""
        d = r.get("inject_minus_neutral")
        ci = r.get("inject_minus_neutral_ci") or {}
        if d is None:
            continue
        items.append((
            _short_model(m), float(d),
            float(ci.get("low", 0.0)), float(ci.get("high", 0.0)),
        ))
    items.sort(key=lambda t: t[1])

    fig = plt.figure(figsize=(11, 6.5))
    ax = fig.add_axes([0.18, 0.18, 0.78, 0.62])
    ys = list(range(len(items)))
    deltas = [t[1] for t in items]
    lows = [t[2] for t in items]
    highs = [t[3] for t in items]
    err_low = [d - l for d, l in zip(deltas, lows)]
    err_high = [h - d for h, d in zip(highs, deltas)]

    # All bars get the same neutral navy because no CI excludes zero in
    # the cross-protocol experiment. A reader who wants to see the
    # significance map can refer to the bullet-point caption.
    ax.errorbar(
        deltas, ys, xerr=[err_low, err_high], fmt="o",
        markersize=10, capsize=4,
        ecolor=_MUTED, elinewidth=1.2, capthick=1.2,
        markeredgecolor=_NAVY, markerfacecolor=_NAVY,
        markeredgewidth=1.4,
    )

    ax.axvline(0, color=_RULE, linestyle="-", linewidth=1.0, zorder=0)
    ax.set_yticks(ys)
    ax.set_yticklabels([t[0] for t in items])
    ax.set_xlabel(
        'Refusal-rate change under "inject" framing (vs neutral)',
    )
    ax.grid(axis="x", which="major", alpha=0.5)
    ax.spines["left"].set_visible(True)
    ax.tick_params(left=False)

    _titled(
        fig,
        "No model shows a measurable refusal shift",
        "8 frontier models, paired-bootstrap 95% CIs. "
        "Every CI overlaps zero.",
    )
    _footer(
        fig,
        "n = 2,000 prompts per model · 4 task families pooled · CIs at B=200 paired bootstrap",
    )

    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    LOGGER.info("wrote %s", out_path)


# ---------------------------------------------------------------------------
# chart_b_vea: per-model VEA inflation bar chart
# ---------------------------------------------------------------------------


def _chart_b_vea(goodfire: Dict[str, Any], out_path: Path) -> None:
    """VEA inflation bar chart per Goodfire-pipeline model."""
    import matplotlib.pyplot as plt

    _apply_blog_style()

    rows = list(goodfire.get("per_model_inject_minus_neutral", []) or [])
    if not rows:
        LOGGER.warning("chart_b skipped: no inflation rows")
        return

    items = []
    for r in rows:
        m = r.get("model_name") or ""
        d = r.get("inject_minus_neutral_vea")
        ci = r.get("inject_minus_neutral_vea_ci") or {}
        if d is None:
            continue
        items.append((
            _short_model(m), float(d),
            float(ci.get("low", 0.0)), float(ci.get("high", 0.0)),
        ))
    items.sort(key=lambda t: t[1])

    fig = plt.figure(figsize=(11, 5.5))
    ax = fig.add_axes([0.18, 0.20, 0.78, 0.55])
    labels = [t[0] for t in items]
    deltas = [t[1] for t in items]
    lows = [t[2] for t in items]
    highs = [t[3] for t in items]
    err_low = [d - l for d, l in zip(deltas, lows)]
    err_high = [h - d for h, d in zip(highs, deltas)]
    colors = [
        _RED if (l > 0 or h < 0) else _GREY
        for l, h in zip(lows, highs)
    ]

    ax.barh(
        labels, deltas, color=colors,
        xerr=[err_low, err_high], capsize=5,
        error_kw={"ecolor": _MUTED, "elinewidth": 1.2, "capthick": 1.2},
        height=0.55, linewidth=0,
    )
    ax.axvline(0, color=_RULE, linestyle="-", linewidth=1.0, zorder=0)

    # Label each bar end past the error-bar cap
    for d, hi, lbl in zip(deltas, highs, labels):
        x_pos = hi + 0.015
        ax.text(x_pos, lbl, f"+{d:.3f}" if d > 0 else f"{d:.3f}",
                va="center", ha="left",
                fontsize=10, color=_INK, fontweight="semibold")

    ax.set_xlabel('Increase in "this is an eval" language in reasoning trace')
    ax.set_xlim(-0.05, 0.55)
    ax.grid(axis="x", which="major", alpha=0.5)
    ax.spines["left"].set_visible(True)
    ax.tick_params(left=False)

    _titled(
        fig,
        "qwen3 talks about evaluation 38pp more under inject framing",
        "Other models verbalize eval-awareness less, or not at all.",
    )
    _footer(
        fig,
        "n = 500 prompts per model · paired bootstrap 95% CIs · "
        "broad-rubric VEA classifier (F1 = 0.886 vs human labels)",
    )

    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    LOGGER.info("wrote %s", out_path)


def _safety_row(summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pull the safety-family row from a strict-mediation summary."""
    for row in summary.get("per_model_family", []) or []:
        if row.get("task_family") == "safety":
            return row
    return None


# ---------------------------------------------------------------------------
# chart_c_apparent_mediation: the "I had it" moment
# ---------------------------------------------------------------------------


def _chart_c_apparent_mediation(
    qwen3_strict: Dict[str, Any], out_path: Path,
) -> None:
    """qwen3 primary safety flip vs stable bars in amber (the apparent
    mediation effect, before replication)."""
    import matplotlib.pyplot as plt

    _apply_blog_style()

    row = _safety_row(qwen3_strict)
    if not row:
        LOGGER.warning("chart_c skipped: no qwen3 safety row")
        return

    strata = row.get("strata") or {}
    flip = strata.get("flip_neg_to_pos") or {}
    stable = strata.get("stable_neg") or {}

    labels = [
        f'Flipped\nsaid "eval" only under inject  ·  n = {int(flip.get("n") or 0)}',
        f'Stable\nnever said "eval"  ·  n = {int(stable.get("n") or 0)}',
    ]
    means = [
        float(flip.get("mean_paired_delta") or 0.0),
        float(stable.get("mean_paired_delta") or 0.0),
    ]
    cis = [
        flip.get("paired_delta_ci") or {},
        stable.get("paired_delta_ci") or {},
    ]
    lows = [float(c.get("low", 0.0)) for c in cis]
    highs = [float(c.get("high", 0.0)) for c in cis]
    err_low = [max(0.0, m - l) for m, l in zip(means, lows)]
    err_high = [max(0.0, h - m) for m, h in zip(means, highs)]
    colors = [
        _AMBER if (l > 0 or h < 0) else _GREY
        for l, h in zip(lows, highs)
    ]

    fig = plt.figure(figsize=(10, 6.2))
    ax = fig.add_axes([0.13, 0.18, 0.80, 0.62])
    bars = ax.bar(
        labels, means, color=colors,
        yerr=[err_low, err_high], capsize=6,
        error_kw={"ecolor": _MUTED, "elinewidth": 1.2, "capthick": 1.2},
        width=0.48, linewidth=0,
    )
    ax.axhline(0, color=_RULE, linestyle="-", linewidth=1.0, zorder=0)

    for bar, m, hi in zip(bars, means, highs):
        y_pos = hi + 0.025 if m >= 0 else (m - 0.05)
        va = "bottom" if m >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"+{m:.3f}" if m > 0 else f"{m:.3f}",
                ha="center", va=va,
                fontsize=11, color=_INK, fontweight="bold")

    ax.set_ylabel("Change in refusal rate · inject vs neutral")
    ax.grid(axis="y", which="major", alpha=0.5)
    ax.spines["bottom"].set_visible(True)
    ax.tick_params(bottom=False)
    ax.set_ylim(-0.20, 0.45)

    _titled(
        fig,
        "qwen3 safety, primary sample: looks like clean mediation",
        "When the model verbalized the eval (flipped), refusal moved. "
        "When it didn't (stable), refusal stayed. Diff-in-diff = +0.222 "
        "[+0.029, +0.430].",
    )
    _footer(
        fig,
        "qwen3-235B · n = 500 mixed-corpus paired prompts · "
        "validated strict-VEA rubric · paired bootstrap CIs (B = 2000)",
    )

    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    LOGGER.info("wrote %s", out_path)


# ---------------------------------------------------------------------------
# chart_d_primary_vs_replication: the twist
# ---------------------------------------------------------------------------


def _chart_d_primary_vs_replication(
    qwen3_strict: Dict[str, Any],
    qwen3_replication_strict: Dict[str, Any],
    out_path: Path,
) -> None:
    """DiD bars side-by-side: primary in amber (apparent finding),
    replication in sage (the null). The twist."""
    import matplotlib.pyplot as plt

    _apply_blog_style()

    p_row = _safety_row(qwen3_strict)
    r_row = _safety_row(qwen3_replication_strict)
    if not p_row or not r_row:
        LOGGER.warning("chart_d skipped: missing primary or replication")
        return

    def _did(row):
        did_obj = row.get("diff_in_diff") or {}
        return (
            float(did_obj.get("diff_in_diff") or 0.0),
            float((did_obj.get("diff_in_diff_ci") or {}).get("low", 0.0)),
            float((did_obj.get("diff_in_diff_ci") or {}).get("high", 0.0)),
            int(did_obj.get("n_flip") or 0),
        )

    p_d, p_lo, p_hi, p_n = _did(p_row)
    r_d, r_lo, r_hi, r_n = _did(r_row)

    labels = [
        f"Primary\nn_flip = {p_n}  ·  prompts 1–500",
        f"Pre-registered replication\nn_flip = {r_n}  ·  prompts 501–700",
    ]
    means = [p_d, r_d]
    err_low = [max(0.0, p_d - p_lo), max(0.0, r_d - r_lo)]
    err_high = [max(0.0, p_hi - p_d), max(0.0, r_hi - r_d)]
    colors = [
        _AMBER if (p_lo > 0 or p_hi < 0) else _GREY,
        _SAGE if (r_lo > 0 or r_hi < 0) else _GREY,
    ]

    fig = plt.figure(figsize=(10, 6.4))
    ax = fig.add_axes([0.14, 0.18, 0.80, 0.62])
    bars = ax.bar(
        labels, means, color=colors,
        yerr=[err_low, err_high], capsize=6,
        error_kw={"ecolor": _MUTED, "elinewidth": 1.2, "capthick": 1.2},
        width=0.48, linewidth=0,
    )
    ax.axhline(0, color=_RULE, linestyle="-", linewidth=1.0, zorder=0)

    # Inside-bar labels for the wide primary bar; above-zero label for the
    # narrow replication bar.
    for bar, m, lo, hi in zip(bars, means, [p_lo, r_lo], [p_hi, r_hi]):
        label = f"{m:+.3f}  CI [{lo:+.3f}, {hi:+.3f}]"
        if abs(m) >= 0.10:
            y_pos = m / 2
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos, label,
                    ha="center", va="center",
                    fontsize=10.5, color="white", fontweight="bold")
        else:
            y_pos = max(hi, 0) + 0.04
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos, label,
                    ha="center", va="bottom",
                    fontsize=10.5, color=_INK, fontweight="bold")

    ax.set_ylabel("qwen3 safety mediation effect · diff-in-diff")
    ax.grid(axis="y", which="major", alpha=0.5)
    ax.spines["bottom"].set_visible(True)
    ax.tick_params(bottom=False)
    ax.set_ylim(-0.30, 0.55)

    _titled(
        fig,
        "The replication did not replicate",
        "Apparent +22pp mediation effect collapses to null on a fresh "
        "disjoint sample with a larger n_flip.",
    )
    _footer(
        fig,
        "qwen3-235B · validated strict-VEA rubric · "
        "paired bootstrap CIs (B = 2000) · "
        "decision rule pre-committed in repo before the replication ran",
    )

    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    LOGGER.info("wrote %s", out_path)


# ---------------------------------------------------------------------------
# chart_e_direction_asymmetry: the headline blog finding
# ---------------------------------------------------------------------------


def _chart_e_direction_asymmetry(
    qwen3_replication_strict: Dict[str, Any],
    opus_strict: Dict[str, Any],
    out_path: Path,
) -> None:
    """qwen3 vs opus stable_neg paired Δ on safety. Opposite directions,
    both CI-excludes-zero. The 'invisible' refusal shifts."""
    import matplotlib.pyplot as plt

    _apply_blog_style()

    q_row = _safety_row(qwen3_replication_strict)
    o_row = _safety_row(opus_strict)
    if not q_row or not o_row:
        LOGGER.warning("chart_e skipped: missing qwen3 or opus")
        return

    def _stable_neg(row):
        s = (row.get("strata") or {}).get("stable_neg") or {}
        d = float(s.get("mean_paired_delta") or 0.0)
        ci = s.get("paired_delta_ci") or {}
        return d, float(ci.get("low", 0.0)), float(ci.get("high", 0.0)), int(s.get("n") or 0)

    q_d, q_lo, q_hi, q_n = _stable_neg(q_row)
    o_d, o_lo, o_hi, o_n = _stable_neg(o_row)

    labels = [
        f'qwen3-235B safety\n(replication, n = {q_n})',
        f'opus-4.7 safety\n(n = {o_n})',
    ]
    means = [q_d, o_d]
    err_low = [max(0.0, q_d - q_lo), max(0.0, o_d - o_lo)]
    err_high = [max(0.0, q_hi - q_d), max(0.0, o_hi - o_d)]
    colors = [
        _RED if (q_lo > 0 or q_hi < 0) else _GREY,
        _RED if (o_lo > 0 or o_hi < 0) else _GREY,
    ]

    fig = plt.figure(figsize=(12, 6.5))
    ax = fig.add_axes([0.22, 0.20, 0.74, 0.55])
    bars = ax.barh(
        labels, means, color=colors,
        xerr=[err_low, err_high], capsize=6,
        error_kw={"ecolor": _MUTED, "elinewidth": 1.2, "capthick": 1.2},
        height=0.50, linewidth=0,
    )
    ax.axvline(0, color=_RULE, linestyle="-", linewidth=1.0, zorder=0)

    # Place value+CI on the side away from each bar (outside the chart
    # interior). For positive bars: to the right of the upper CI; for
    # negative bars: just to the right of the zero line, inside the
    # chart, so the label doesn't collide with the y-tick text.
    for bar, m, lo, hi in zip(bars, means, [q_lo, o_lo], [q_hi, o_hi]):
        label = f"{m:+.3f}  CI [{lo:+.3f}, {hi:+.3f}]"
        if m >= 0:
            x_pos = hi + 0.018
            ha = "left"
        else:
            x_pos = 0.018
            ha = "left"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                label, va="center", ha=ha,
                fontsize=10.5, color=_INK, fontweight="bold")

    ax.set_xlabel(
        "Refusal-rate change on safety prompts under inject framing\n"
        '(only prompts where neither model said "this is an eval")',
    )
    ax.set_xlim(-0.55, 0.55)
    ax.grid(axis="x", which="major", alpha=0.5)
    ax.spines["left"].set_visible(True)
    ax.tick_params(left=False)

    _titled(
        fig,
        "Two models, opposite directions — both invisible to CoT monitoring",
        "On prompts where neither model verbalized the evaluation, "
        "qwen3 refused more (+22pp) and opus refused less (-34pp). "
        "Both CIs exclude zero.",
    )
    _footer(
        fig,
        "strict-VEA stable_neg stratum (VEA stayed negative under both conditions) · "
        "paired bootstrap CIs (B = 2000)",
    )

    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    LOGGER.info("wrote %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross-protocol-summary", default=None)
    parser.add_argument("--goodfire-summary", default=None)
    parser.add_argument("--strict-mediation-summary", default=None)
    parser.add_argument("--opus-strict-mediation-summary", default=None)
    parser.add_argument("--qwen3-replication-strict-mediation-summary", default=None)
    parser.add_argument("--out-dir", default="docs/blog_figures")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cross = _load_json(args.cross_protocol_summary)
    goodfire = _load_json(args.goodfire_summary)
    strict = _load_json(args.strict_mediation_summary)
    opus_strict = _load_json(args.opus_strict_mediation_summary)
    replication_strict = _load_json(
        args.qwen3_replication_strict_mediation_summary
    )

    # The setup diagram has no data dependency — always rendered.
    _setup_diagram(out_dir / "diagram_setup.png")
    produced = 1

    if cross:
        _chart_a_forest(cross, out_dir / "blog_chart_a_forest.png")
        produced += 1
    else:
        LOGGER.info("chart_a skipped (no --cross-protocol-summary)")

    if goodfire:
        _chart_b_vea(goodfire, out_dir / "blog_chart_b_vea.png")
        produced += 1
    else:
        LOGGER.info("chart_b skipped (no --goodfire-summary)")

    if strict:
        _chart_c_apparent_mediation(
            strict, out_dir / "blog_chart_c_apparent_mediation.png"
        )
        produced += 1
    else:
        LOGGER.info("chart_c skipped (no --strict-mediation-summary)")

    if strict and replication_strict:
        _chart_d_primary_vs_replication(
            strict, replication_strict,
            out_dir / "blog_chart_d_primary_vs_replication.png",
        )
        produced += 1
    else:
        LOGGER.info(
            "chart_d skipped (needs --strict-mediation-summary and "
            "--qwen3-replication-strict-mediation-summary)"
        )

    if replication_strict and opus_strict:
        _chart_e_direction_asymmetry(
            replication_strict, opus_strict,
            out_dir / "blog_chart_e_direction_asymmetry.png",
        )
        produced += 1
    else:
        LOGGER.info(
            "chart_e skipped (needs --qwen3-replication-strict-mediation-summary "
            "and --opus-strict-mediation-summary)"
        )

    LOGGER.info("produced %d figure(s) in %s", produced, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
