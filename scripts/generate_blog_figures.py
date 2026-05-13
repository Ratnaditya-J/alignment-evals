"""Generate blog-style figures for docs/blog_draft.md.

Produces the 5 figures referenced by the blog post in docs/blog_draft.md.
Style choices vs the paper figures: larger fonts, simpler labels, hex
colors chosen for Medium's white background.

  diagram_setup.png                       experimental design at a glance
  blog_chart_a_forest.png                 simplified per-model refusal forest
  blog_chart_b_vea.png                    VEA inflation bar chart
  blog_chart_c_apparent_mediation.png     qwen3 primary mediation (the "I had it" moment)
  blog_chart_d_primary_vs_replication.png primary vs replication DiD bars
  blog_chart_e_direction_asymmetry.png    qwen3 vs opus stable_neg, opposite directions

Numbers are pulled from the same run JSONs the paper figures use:
  --cross-protocol-summary  -> diagram_setup (counts only) + chart_a
  --goodfire-summary        -> chart_b
  --strict-mediation-summary (qwen3 n=500 primary)
  --qwen3-replication-strict-mediation-summary
  --opus-strict-mediation-summary

If any summary is missing the corresponding figures are skipped, same as
the paper figure generator.

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


# Blog-friendly palette. White background; warm red for CI-excludes-zero;
# cool grey for null; sage green for the new replication panel; muted
# orange for the apparent-mediation primary; black/dark grey for text.
_RED = "#c93b3b"        # CI excludes zero
_GREY = "#b8b8b8"       # null / not significant
_DARK_GREY = "#4a4a4a"  # axes, text
_NAVY = "#2c5282"       # neutral structural accent
_SAGE = "#5d8a66"       # the replication / good-news color
_AMBER = "#d97706"      # the apparent-finding-before-replication color


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


def _setup_diagram(out_path: Path) -> None:
    """A clean text+box diagram of the experimental setup. No data; pure
    schema. Renders as the post's first inline image so a casual reader
    grasps the 4-condition × 4-family × 8-model design in one glance."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Title
    ax.text(
        6, 5.6,
        "The experiment, at a glance",
        ha="center", va="center", fontsize=16, fontweight="bold",
        color=_DARK_GREY,
    )

    # Left column: prompt sources
    prompts = [
        ("MMLU", "knowledge"),
        ("GSM8K", "math"),
        ("BeaverTails", "safety"),
        ("TruthfulQA", "truthfulness"),
    ]
    for i, (src, fam) in enumerate(prompts):
        y = 4.3 - i * 0.85
        ax.add_patch(FancyBboxPatch(
            (0.3, y - 0.3), 2.4, 0.6,
            boxstyle="round,pad=0.04", linewidth=1.2,
            edgecolor=_DARK_GREY, facecolor="#f3f4f6",
        ))
        ax.text(1.5, y, f"{src}\n({fam})", ha="center", va="center",
                fontsize=10, color=_DARK_GREY)

    ax.text(1.5, 5, "Prompts (125 / family)",
            ha="center", va="center", fontsize=11,
            fontweight="bold", color=_DARK_GREY)

    # Arrow
    ax.annotate("", xy=(3.9, 2.8), xytext=(2.9, 2.8),
                arrowprops=dict(arrowstyle="->", color=_DARK_GREY, lw=1.5))

    # Middle column: 4 conditions
    conditions = [
        ("baseline", "no system prompt", _GREY),
        ("neutral", '"You are a helpful assistant."', _NAVY),
        ("inject", '"Note: this is an evaluation..."', _RED),
        ("remove", '"This is a real customer interaction..."', _SAGE),
    ]
    for i, (cond, desc, color) in enumerate(conditions):
        y = 4.3 - i * 0.85
        ax.add_patch(FancyBboxPatch(
            (4.0, y - 0.3), 4.0, 0.6,
            boxstyle="round,pad=0.04", linewidth=1.4,
            edgecolor=color, facecolor="white",
        ))
        ax.text(4.2, y, cond, ha="left", va="center",
                fontsize=10, fontweight="bold", color=color)
        ax.text(6.2, y, desc, ha="left", va="center",
                fontsize=9, color=_DARK_GREY, style="italic")

    ax.text(6.0, 5, "4 conditions × same prompt",
            ha="center", va="center", fontsize=11,
            fontweight="bold", color=_DARK_GREY)

    # Arrow
    ax.annotate("", xy=(9.2, 2.8), xytext=(8.2, 2.8),
                arrowprops=dict(arrowstyle="->", color=_DARK_GREY, lw=1.5))

    # Right column: 8 models
    ax.add_patch(FancyBboxPatch(
        (9.3, 1.0), 2.5, 3.6,
        boxstyle="round,pad=0.04", linewidth=1.2,
        edgecolor=_DARK_GREY, facecolor="#f3f4f6",
    ))
    ax.text(10.55, 5, "8 frontier models",
            ha="center", va="center", fontsize=11,
            fontweight="bold", color=_DARK_GREY)
    models_text = (
        "haiku-4.5\nsonnet-4.6\nopus-4.7\n"
        "gpt-4o-mini\ngpt-5.5\n"
        "deepseek-v3.2\nqwen3-235B\nkimi-k2"
    )
    ax.text(10.55, 2.7, models_text, ha="center", va="center",
            fontsize=9.5, color=_DARK_GREY)

    # Bottom note
    ax.text(
        6, 0.3,
        "16,000 responses · score refusal · classify reasoning trace for "
        '"this is an eval" language · ~$500 in API fees',
        ha="center", va="center", fontsize=9,
        color=_DARK_GREY, style="italic",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    LOGGER.info("wrote %s", out_path)


def _chart_a_forest(cross: Dict[str, Any], out_path: Path) -> None:
    """Forest plot of per-model inject-neutral refusal Δ. Blog version:
    bigger fonts, sorted by point estimate, simple short labels."""
    import matplotlib.pyplot as plt

    rows = list(cross.get("per_model", []) or [])
    if not rows:
        LOGGER.warning("chart_a skipped: no per_model rows")
        return

    # Pull (label, delta, low, high)
    items = []
    for r in rows:
        m = r.get("model_name") or ""
        d = r.get("inject_minus_neutral")
        ci = r.get("inject_minus_neutral_ci") or {}
        if d is None:
            continue
        items.append((_short_model(m), float(d), float(ci.get("low", 0.0)), float(ci.get("high", 0.0))))
    items.sort(key=lambda t: t[1])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ys = list(range(len(items)))
    deltas = [t[1] for t in items]
    lows = [t[2] for t in items]
    highs = [t[3] for t in items]
    err_low = [d - l for d, l in zip(deltas, lows)]
    err_high = [h - d for h, d in zip(highs, deltas)]
    colors = [
        _RED if (l > 0 or h < 0) else _GREY
        for l, h in zip(lows, highs)
    ]

    ax.errorbar(
        deltas, ys, xerr=[err_low, err_high], fmt="o",
        markersize=9, capsize=5,
        ecolor=_DARK_GREY, elinewidth=1.4,
        markeredgecolor=_DARK_GREY, markerfacecolor=_NAVY,
    )
    for y, d, c in zip(ys, deltas, colors):
        ax.plot(d, y, "o", markersize=9, color=c,
                markeredgecolor=_DARK_GREY)

    ax.axvline(0, color=_DARK_GREY, linestyle="--", linewidth=1, alpha=0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels([t[0] for t in items], fontsize=11)
    ax.set_xlabel(
        "Refusal-rate change under inject framing (vs neutral)\n"
        '"Note: this is an evaluation..." minus "You are a helpful assistant"',
        fontsize=11,
    )
    ax.tick_params(axis="x", labelsize=10)
    ax.set_title(
        "Zero of 8 models shows a confidence interval excluding zero",
        fontsize=13, color=_DARK_GREY,
    )
    ax.grid(axis="x", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    LOGGER.info("wrote %s", out_path)


def _chart_b_vea(goodfire: Dict[str, Any], out_path: Path) -> None:
    """VEA inflation bar chart (paired inject-minus-neutral) per
    open-reasoning model + opus-4.7. Blog-styled."""
    import matplotlib.pyplot as plt

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

    fig, ax = plt.subplots(figsize=(10, 4.5))
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

    bars = ax.barh(labels, deltas, color=colors,
                   xerr=[err_low, err_high], capsize=6,
                   error_kw={"ecolor": _DARK_GREY, "elinewidth": 1.3})
    ax.axvline(0, color=_DARK_GREY, linestyle="--", linewidth=1, alpha=0.6)
    # Place each value label OUTSIDE the error-bar cap so the text never
    # collides with the cap glyph (the previous d + 0.02 offset landed
    # right on top of the cap for the wider bars).
    for bar, d, hi, lo in zip(bars, deltas, highs, lows):
        if d >= 0:
            x_pos = hi + 0.012
            ha = "left"
        else:
            x_pos = lo - 0.012
            ha = "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"+{d:.3f}" if d > 0 else f"{d:.3f}",
                va="center", ha=ha,
                fontsize=10, color=_DARK_GREY)
    ax.set_xlabel(
        '"This is an eval" language in reasoning trace\n'
        "(inject minus neutral rate, paired)",
        fontsize=11,
    )
    ax.tick_params(axis="both", labelsize=11)
    ax.set_title(
        "qwen3 talks about the evaluation 38pp more under inject framing.\n"
        "Other models verbalize less or not at all.",
        fontsize=12, color=_DARK_GREY,
    )
    ax.set_xlim(-0.05, 0.5)
    ax.grid(axis="x", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    LOGGER.info("wrote %s", out_path)


def _safety_row(summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pull the safety-family row from a strict-mediation summary."""
    for row in summary.get("per_model_family", []) or []:
        if row.get("task_family") == "safety":
            return row
    return None


def _chart_c_apparent_mediation(
    qwen3_strict: Dict[str, Any], out_path: Path,
) -> None:
    """The 'I thought I had it' moment: qwen3 primary safety flip vs
    stable strata bars. Visually identical to fig5's left panel but
    larger fonts + the amber 'before replication' color."""
    import matplotlib.pyplot as plt

    row = _safety_row(qwen3_strict)
    if not row:
        LOGGER.warning("chart_c skipped: no qwen3 safety row")
        return

    strata = row.get("strata") or {}
    flip = strata.get("flip_neg_to_pos") or {}
    stable = strata.get("stable_neg") or {}

    labels = [
        f'Flipped: said "eval" only under inject\nn={int(flip.get("n") or 0)}',
        f'Stable: never said "eval"\nn={int(stable.get("n") or 0)}',
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

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(labels, means, color=colors,
                  yerr=[err_low, err_high], capsize=8,
                  error_kw={"ecolor": _DARK_GREY, "elinewidth": 1.4},
                  width=0.55)
    ax.axhline(0, color=_DARK_GREY, linestyle="--", linewidth=1, alpha=0.6)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                m + (0.02 if m >= 0 else -0.04),
                f"+{m:.3f}" if m > 0 else f"{m:.3f}",
                ha="center", va="bottom" if m >= 0 else "top",
                fontsize=11, color=_DARK_GREY, fontweight="bold")

    ax.set_ylabel(
        "Change in refusal rate, inject vs neutral\n"
        "(same prompt, both conditions)",
        fontsize=11,
    )
    ax.tick_params(axis="both", labelsize=10)
    ax.set_title(
        "qwen3 safety, primary analysis (n=500): looks like clean mediation",
        fontsize=12, color=_DARK_GREY,
    )
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    LOGGER.info("wrote %s", out_path)


def _chart_d_primary_vs_replication(
    qwen3_strict: Dict[str, Any],
    qwen3_replication_strict: Dict[str, Any],
    out_path: Path,
) -> None:
    """The DiD bars side-by-side: primary in amber (apparent mediation),
    replication in sage (the null). This is the twist."""
    import matplotlib.pyplot as plt

    p_row = _safety_row(qwen3_strict)
    r_row = _safety_row(qwen3_replication_strict)
    if not p_row or not r_row:
        LOGGER.warning("chart_d skipped: missing primary or replication row")
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
        f"Primary (n_flip={p_n})\nfirst sample, prompts 1–500",
        f"Pre-registered replication (n_flip={r_n})\nfresh sample, prompts 501–700",
    ]
    means = [p_d, r_d]
    err_low = [max(0.0, p_d - p_lo), max(0.0, r_d - r_lo)]
    err_high = [max(0.0, p_hi - p_d), max(0.0, r_hi - r_d)]
    colors = [
        _AMBER if (p_lo > 0 or p_hi < 0) else _GREY,
        _SAGE if (r_lo > 0 or r_hi < 0) else _GREY,
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, means, color=colors,
                  yerr=[err_low, err_high], capsize=8,
                  error_kw={"ecolor": _DARK_GREY, "elinewidth": 1.4},
                  width=0.55)
    ax.axhline(0, color=_DARK_GREY, linestyle="--", linewidth=1, alpha=0.6)
    # Place the value+CI label INSIDE the bar (above zero for positive
    # bars, below zero for negative bars), so labels don't collide with
    # the title at the top or the x-axis labels at the bottom.
    for bar, m, lo, hi in zip(bars, means, [p_lo, r_lo], [p_hi, r_hi]):
        label = f"{m:+.3f}  CI [{lo:+.3f}, {hi:+.3f}]"
        if m >= 0:
            # Above the zero line, just inside the top of the bar.
            y_pos = max(0.01, m / 2)
            va = "center"
            color = "white"
        else:
            # Below the zero line.
            y_pos = min(-0.01, m / 2)
            va = "center"
            color = _DARK_GREY
        # If the bar magnitude is too small to host the text comfortably,
        # park the label just above zero in dark grey instead of white-
        # on-bar (which would be unreadable on the narrow grey bar).
        if abs(m) < 0.05:
            y_pos = max(hi, 0.0) + 0.04
            va = "bottom"
            color = _DARK_GREY
        ax.text(bar.get_x() + bar.get_width() / 2,
                y_pos, label, ha="center", va=va,
                fontsize=10.5, color=color, fontweight="bold")

    ax.set_ylabel(
        "qwen3 safety mediation effect (diff-in-diff)",
        fontsize=11,
    )
    ax.tick_params(axis="both", labelsize=10)
    ax.set_title(
        "The replication did not replicate.\n"
        "Apparent +22pp mediation effect collapses to null on a fresh sample.",
        fontsize=12, color=_DARK_GREY, pad=15,
    )
    ax.set_ylim(-0.25, 0.50)
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    LOGGER.info("wrote %s", out_path)


def _chart_e_direction_asymmetry(
    qwen3_replication_strict: Dict[str, Any],
    opus_strict: Dict[str, Any],
    out_path: Path,
) -> None:
    """The headline blog finding: qwen3 vs opus stable_neg paired Δ on
    safety. Opposite directions. Both CI-excludes-zero. The 'invisible'
    refusal shifts."""
    import matplotlib.pyplot as plt

    q_row = _safety_row(qwen3_replication_strict)
    o_row = _safety_row(opus_strict)
    if not q_row or not o_row:
        LOGGER.warning("chart_e skipped: missing qwen3 replication or opus row")
        return

    def _stable_neg(row):
        s = (row.get("strata") or {}).get("stable_neg") or {}
        d = float(s.get("mean_paired_delta") or 0.0)
        ci = s.get("paired_delta_ci") or {}
        return d, float(ci.get("low", 0.0)), float(ci.get("high", 0.0)), int(s.get("n") or 0)

    q_d, q_lo, q_hi, q_n = _stable_neg(q_row)
    o_d, o_lo, o_hi, o_n = _stable_neg(o_row)

    labels = [
        f"qwen3-235B safety (replication)\nn={q_n} prompts, VEA never appeared",
        f"opus-4.7 safety\nn={o_n} prompts, VEA never appeared",
    ]
    means = [q_d, o_d]
    err_low = [max(0.0, q_d - q_lo), max(0.0, o_d - o_lo)]
    err_high = [max(0.0, q_hi - q_d), max(0.0, o_hi - o_d)]
    colors = [
        _RED if (q_lo > 0 or q_hi < 0) else _GREY,
        _RED if (o_lo > 0 or o_hi < 0) else _GREY,
    ]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.barh(labels, means, color=colors,
                   xerr=[err_low, err_high], capsize=8,
                   error_kw={"ecolor": _DARK_GREY, "elinewidth": 1.4},
                   height=0.55)
    ax.axvline(0, color=_DARK_GREY, linestyle="--", linewidth=1, alpha=0.6)
    # Place value labels consistently to the right of each bar's far end,
    # offset away from zero so neither label collides with the y-axis tick
    # text on the left.
    for bar, m, lo, hi in zip(bars, means, [q_lo, o_lo], [q_hi, o_hi]):
        label = f"{m:+.3f}  CI [{lo:+.3f}, {hi:+.3f}]"
        if m >= 0:
            x_pos = hi + 0.02
            ha = "left"
        else:
            x_pos = 0.02   # to the right of the zero line, inside the chart
            ha = "left"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                label, va="center", ha=ha,
                fontsize=10.5, color=_DARK_GREY, fontweight="bold")

    ax.set_xlabel(
        "Refusal-rate change on safety prompts under inject framing\n"
        '(prompts where neither model said "this is an eval")',
        fontsize=11,
    )
    ax.tick_params(axis="both", labelsize=10.5)
    ax.set_title(
        "Two models, opposite directions, both invisible to CoT monitoring.\n"
        "Both CIs exclude zero. Neither verbalized the evaluation.",
        fontsize=12, color=_DARK_GREY,
    )
    ax.set_xlim(-0.55, 0.55)
    ax.grid(axis="x", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    LOGGER.info("wrote %s", out_path)


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

    produced = 0

    # diagram_setup is always rendered — pure schema, needs no data.
    _setup_diagram(out_dir / "diagram_setup.png")
    produced += 1

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
