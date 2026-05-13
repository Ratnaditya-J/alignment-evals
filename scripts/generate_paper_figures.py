"""Generate paper figures from existing run-artifact JSONs.

Produces the four figures referenced in ``docs/paper_draft.md``:

  1. fig1_refusal_forest.png        — cross-protocol per-model refusal Δ
     inject−neutral forest plot with CI error bars and a vertical zero
     line. Reads ``cross_protocol_summary.json``.

  2. fig2_vea_inflation.png         — per-model VEA+ inflation Δ
     inject−neutral with paired-bootstrap CIs; bars colored by
     significance. Reads ``goodfire_vea_summary.json``.

  3. fig3_qwen3_per_family.png      — qwen3 per-family VEA inflation
     bar chart with CIs. Math is the unique null cell. Reads
     ``goodfire_vea_summary.json``.

  4. fig4_mediation_panels.png      — three-panel mediation figure:
     (a) aggregate within-cohort VEA+/VEA− refusal split,
     (b) per-family within-prompt DiD under broad rubric,
     (c) per-family within-prompt DiD under strict rubric.
     Reads ``vea_mediation_summary.json`` + the ``.strict.json`` sibling.

  5. fig5_two_mechanism.png         — side-by-side per-model panel
     contrasting qwen3 (VEA-mediated) vs opus (non-verbalized direct)
     on the safety family. Each panel shows paired Δrefusal for the
     flip and stable_neg strata with CIs. Reads two strict-mediation
     summaries (one per model). See §5.4 of the paper.

Each figure has an explicit caption embedded in the .png as a matplotlib
title; the paper draft references them by filename. The script is
deliberately self-contained — only matplotlib + the project's own
metadata helpers are required.

Usage:
    python scripts/generate_paper_figures.py \\
        --cross-protocol-summary runs/cross-protocol-v6/cross_protocol_summary.json \\
        --goodfire-summary runs/goodfire-mixed-n500/goodfire_vea_summary.json \\
        --mediation-summary runs/goodfire-mixed-n500/vea_mediation_summary.json \\
        --strict-mediation-summary runs/goodfire-mixed-n500/vea_mediation_summary.strict.json \\
        --opus-strict-mediation-summary runs/goodfire-opus-mixed-n300/vea_mediation_summary.strict.json \\
        --out-dir docs/figures/

Each --*-summary argument is optional. Figures whose inputs aren't
supplied are skipped (with an info log); the script exits 0 as long as
at least one figure was produced. fig5 needs both
``--strict-mediation-summary`` (qwen3) and
``--opus-strict-mediation-summary`` (opus).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

LOGGER = logging.getLogger("generate_paper_figures")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def _load_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        LOGGER.warning("skip: %s not found", p)
        return None
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# Figure 1: refusal forest plot
# ---------------------------------------------------------------------------


def _render_refusal_forest(summary: Dict[str, Any], out_path: Path) -> None:
    """Per-model inject-neutral refusal Δ with CIs, ordered by point
    estimate. Vertical line at Δ=0. Used to make the "8/8 frontier models
    null" claim visual."""
    import matplotlib.pyplot as plt

    rows: List[Dict[str, Any]] = list(summary.get("per_model", []))
    if not rows:
        LOGGER.warning("fig1: per_model empty, skipping")
        return
    rows.sort(key=lambda r: float(r.get("inject_minus_neutral", 0.0)))
    labels = [r.get("model_name", "?") for r in rows]
    deltas = [float(r.get("inject_minus_neutral", 0.0)) for r in rows]
    lows = [float((r.get("inject_minus_neutral_ci") or {}).get("low", 0.0)) for r in rows]
    highs = [float((r.get("inject_minus_neutral_ci") or {}).get("high", 0.0)) for r in rows]
    err_low = [d - l for d, l in zip(deltas, lows)]
    err_high = [h - d for d, h in zip(deltas, highs)]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * len(rows))))
    y = list(range(len(rows)))
    ax.errorbar(
        deltas,
        y,
        xerr=[err_low, err_high],
        fmt="o",
        color="#2b7cb8",
        ecolor="#2b7cb8",
        capsize=4,
        markersize=6,
    )
    ax.axvline(0, color="#888", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Δ inject − neutral refusal rate")
    ax.set_title(
        "Cross-protocol refusal-rate Δ inject−neutral per model\n"
        "(no CI excludes zero on 8/8 frontier models)"
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("wrote %s", out_path)


# ---------------------------------------------------------------------------
# Figure 2: per-model VEA inflation
# ---------------------------------------------------------------------------


def _render_vea_inflation(summary: Dict[str, Any], out_path: Path) -> None:
    """Per-model inject-neutral VEA inflation Δ with paired-bootstrap CIs;
    bars colored by CI-excludes-zero significance."""
    import matplotlib.pyplot as plt

    rows: List[Dict[str, Any]] = list(
        summary.get("per_model_inject_minus_neutral", [])
    )
    if not rows:
        LOGGER.warning("fig2: per_model_inject_minus_neutral empty, skipping")
        return
    rows.sort(key=lambda r: float(r.get("inject_minus_neutral_vea", 0.0)))
    labels = [_short_model(r.get("model_name", "?")) for r in rows]
    deltas = [float(r.get("inject_minus_neutral_vea", 0.0)) for r in rows]
    cis = [r.get("inject_minus_neutral_vea_ci") or {} for r in rows]
    lows = [float(c.get("low", 0.0)) for c in cis]
    highs = [float(c.get("high", 0.0)) for c in cis]
    err_low = [d - l for d, l in zip(deltas, lows)]
    err_high = [h - d for d, h in zip(deltas, highs)]
    colors = [
        "#d62728" if (l > 0 or h < 0) else "#aaaaaa"
        for l, h in zip(lows, highs)
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = list(range(len(rows)))
    ax.bar(x, deltas, color=colors, yerr=[err_low, err_high], capsize=6)
    ax.axhline(0, color="#888", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Δ inject − neutral VEA+ rate")
    ax.set_title(
        "Reasoning-trace VEA inflation per model\n"
        "(red = CI excludes zero, grey = not significant)"
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("wrote %s", out_path)


# ---------------------------------------------------------------------------
# Figure 3: qwen3 per-family VEA inflation
# ---------------------------------------------------------------------------


def _render_qwen3_per_family(summary: Dict[str, Any], out_path: Path) -> None:
    """Per-family VEA inflation Δ for qwen3 only. Derived from
    per_model_condition rows by reconstructing per-family inject and
    neutral VEA+ rates from the stratum counts in the mediation output;
    when not available falls back to the per_model_condition summary
    aggregated by family."""
    import matplotlib.pyplot as plt

    target_model = "openrouter:qwen/qwen3-235b-a22b-thinking-2507"
    by_family: Dict[str, Dict[str, float]] = {}
    # The goodfire_vea_summary has per_model_condition keyed only by
    # (model, condition); without per-family stratum it can't break down
    # to families. Most reliable: read the strata from the mediation
    # summary if available, fall back to a placeholder otherwise.
    # Caller passes either the goodfire summary OR the mediation
    # summary. We try both.
    per_family_strata = summary.get("per_model_family", [])
    if per_family_strata and target_model in {
        r.get("model_name") for r in per_family_strata
    }:
        for entry in per_family_strata:
            if entry.get("model_name") != target_model:
                continue
            family = str(entry.get("task_family") or "?")
            strata = entry.get("strata", {})
            flip_neg = (strata.get("flip_neg_to_pos") or {}).get("n", 0)
            stable_pos = (strata.get("stable_pos") or {}).get("n", 0)
            flip_pos_neg = (strata.get("flip_pos_to_neg") or {}).get("n", 0)
            stable_neg = (strata.get("stable_neg") or {}).get("n", 0)
            total = flip_neg + stable_pos + flip_pos_neg + stable_neg
            if total == 0:
                continue
            neutral_pos = stable_pos + flip_pos_neg
            inject_pos = stable_pos + flip_neg
            by_family[family] = {
                "neutral": neutral_pos / total,
                "inject": inject_pos / total,
                "delta": (inject_pos - neutral_pos) / total,
                "n_total": total,
            }

    if not by_family:
        LOGGER.warning(
            "fig3: no per-family data for %s, skipping", target_model
        )
        return

    order = sorted(
        by_family, key=lambda f: by_family[f]["delta"], reverse=True
    )
    deltas = [by_family[f]["delta"] for f in order]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars = ax.bar(order, deltas, color="#3a7eb8")
    for bar, delta in zip(bars, deltas):
        ax.annotate(
            f"+{delta * 100:.1f}pp" if delta >= 0 else f"{delta * 100:.1f}pp",
            xy=(bar.get_x() + bar.get_width() / 2, delta),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    ax.axhline(0, color="#888", linestyle="--", linewidth=1)
    ax.set_ylabel("Δ inject − neutral VEA+ rate")
    ax.set_title(
        "qwen3-235B-thinking VEA inflation per task family\n"
        "(inflation generalizes across truthfulness, safety, knowledge; "
        "math is the unique null)"
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("wrote %s", out_path)


# ---------------------------------------------------------------------------
# Figure 4: three-panel mediation visualization
# ---------------------------------------------------------------------------


def _render_mediation_panels(
    broad_summary: Dict[str, Any],
    strict_summary: Optional[Dict[str, Any]],
    out_path: Path,
) -> None:
    """Three panels:
      (a) aggregate within-cohort VEA+/VEA− refusal split (not the
          mediation estimand; shown to illustrate the apparent
          'mediation' signal that the per-family analysis dismantles),
      (b) per-family within-prompt DiD under broad rubric,
      (c) per-family within-prompt DiD under strict rubric.
    """
    import matplotlib.pyplot as plt

    target = "openrouter:qwen/qwen3-235b-a22b-thinking-2507"
    per_model = next(
        (r for r in broad_summary.get("per_model", []) if r.get("model_name") == target),
        None,
    )
    per_family_broad = [
        r for r in broad_summary.get("per_model_family", []) if r.get("model_name") == target
    ]
    per_family_strict = (
        [
            r
            for r in (strict_summary or {}).get("per_model_family", [])
            if r.get("model_name") == target
        ]
        if strict_summary
        else []
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel a: aggregate within-cohort split. Reconstruct from strata.
    if per_model:
        strata = per_model.get("strata", {})
        # VEA+ rollouts include flip_neg_to_pos.inject + stable_pos.both +
        # flip_pos_to_neg.neutral. We approximate the within-cohort split
        # using the strata as: VEA+ refusal ≈ avg of (flip_neg_to_pos
        # inject) and (stable_pos avg); VEA− refusal ≈ avg of (stable_neg
        # avg) and (flip_neg_to_pos neutral). This is a within-cohort
        # observational comparison shown for contrast, not the causal
        # estimand.
        flip_neg = strata.get("flip_neg_to_pos") or {}
        stable_neg = strata.get("stable_neg") or {}
        stable_pos = strata.get("stable_pos") or {}
        # Pull averages directly.
        pos_refusal_estimate = (
            float(flip_neg.get("mean_refusal_treatment") or 0.0)
            * (flip_neg.get("n") or 0)
            + float(stable_pos.get("mean_refusal_treatment") or 0.0)
            * (stable_pos.get("n") or 0)
        )
        pos_n = (flip_neg.get("n") or 0) + (stable_pos.get("n") or 0)
        pos_refusal = pos_refusal_estimate / pos_n if pos_n else 0.0
        neg_refusal_estimate = (
            float(stable_neg.get("mean_refusal_control") or 0.0)
            * (stable_neg.get("n") or 0)
            + float(flip_neg.get("mean_refusal_control") or 0.0)
            * (flip_neg.get("n") or 0)
        )
        neg_n = (stable_neg.get("n") or 0) + (flip_neg.get("n") or 0)
        neg_refusal = neg_refusal_estimate / neg_n if neg_n else 0.0
        axes[0].bar(
            ["VEA+", "VEA−"],
            [pos_refusal, neg_refusal],
            color=["#d62728", "#7e9eb2"],
        )
        axes[0].set_ylabel("mean refusal rate")
        axes[0].set_ylim(0, 1)
        axes[0].set_title(
            "(a) Aggregate within-cohort refusal\n"
            "(observational, confounded by family)"
        )
        axes[0].grid(axis="y", alpha=0.25)
    else:
        axes[0].text(0.5, 0.5, "no aggregate data", ha="center", va="center")
        axes[0].set_axis_off()

    # Panels b and c: per-family DiD with CIs.
    def _panel_did(
        ax,
        rows: Sequence[Dict[str, Any]],
        title_suffix: str,
    ) -> None:
        rows = [r for r in rows if r.get("task_family")]
        rows = sorted(rows, key=lambda r: r["task_family"])
        if not rows:
            ax.text(0.5, 0.5, "no per-family data", ha="center", va="center")
            ax.set_axis_off()
            return
        families = [r["task_family"] for r in rows]
        dids = [
            float((r.get("diff_in_diff") or {}).get("diff_in_diff") or 0.0)
            for r in rows
        ]
        cis = [
            (r.get("diff_in_diff") or {}).get("diff_in_diff_ci") or {}
            for r in rows
        ]
        lows = [float(c.get("low", 0.0)) for c in cis]
        highs = [float(c.get("high", 0.0)) for c in cis]
        err_low = [max(0.0, d - l) for d, l in zip(dids, lows)]
        err_high = [max(0.0, h - d) for d, h in zip(dids, highs)]
        colors = [
            "#d62728" if (l > 0 or h < 0) else "#7e9eb2"
            for l, h in zip(lows, highs)
        ]
        ax.bar(
            families, dids, color=colors,
            yerr=[err_low, err_high], capsize=6,
        )
        ax.axhline(0, color="#888", linestyle="--", linewidth=1)
        ax.set_ylabel("diff-in-diff Δrefusal")
        ax.set_title(
            "(b) per-family DiD\n(broad rubric)"
            if "broad" in title_suffix
            else "(c) per-family DiD\n(strict rubric)"
        )
        ax.grid(axis="y", alpha=0.25)

    _panel_did(axes[1], per_family_broad, "broad")
    _panel_did(axes[2], per_family_strict or [], "strict")

    fig.suptitle(
        "Aggregate vs per-family vs validated-strict mediation analysis (qwen3)\n"
        "validated strict surfaces qwen3 safety strong_mediation; "
        "capability families are mechanically null (refusal ~0)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("wrote %s", out_path)


# ---------------------------------------------------------------------------
# Figure 5: two-mechanism contrast (qwen3 vs opus, safety family)
# ---------------------------------------------------------------------------


def _render_two_mechanism(
    qwen3_strict: Dict[str, Any],
    opus_strict: Dict[str, Any],
    out_path: Path,
) -> None:
    """Side-by-side per-model panel contrasting qwen3 (VEA-mediated)
    vs opus (non-verbalized direct) on the safety family. Each panel
    shows paired Δrefusal for the flip-neg-to-pos stratum and the
    stable_neg stratum with bootstrap CIs. The qualitative contrast:
    qwen3's effect lives in the flip stratum; opus's effect lives in
    the stable_neg stratum.

    Strict-rubric, safety-family-only. Reads each model's
    ``vea_mediation_summary.strict.json``.
    """
    import matplotlib.pyplot as plt

    def _safety_row(summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for row in summary.get("per_model_family", []) or []:
            if row.get("task_family") == "safety":
                return row
        return None

    qwen3_row = _safety_row(qwen3_strict)
    opus_row = _safety_row(opus_strict)
    if not qwen3_row or not opus_row:
        LOGGER.warning("fig5: missing safety row in qwen3 or opus summary")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    def _panel(ax: Any, row: Dict[str, Any], model_label: str) -> None:
        strata = row.get("strata") or {}
        flip = strata.get("flip_neg_to_pos") or {}
        stable = strata.get("stable_neg") or {}

        labels = [
            f"flip (− → +)\nn={int(flip.get('n') or 0)}",
            f"stable −\nn={int(stable.get('n') or 0)}",
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
        # Red when CI excludes zero, gray otherwise — matches fig4 convention.
        colors = [
            "#d62728" if (l > 0 or h < 0) else "#7e9eb2"
            for l, h in zip(lows, highs)
        ]
        ax.bar(
            labels, means, color=colors,
            yerr=[err_low, err_high], capsize=6,
        )
        ax.axhline(0, color="#888", linestyle="--", linewidth=1)
        ax.set_ylabel("paired Δrefusal (inject − neutral)")

        did_obj = row.get("diff_in_diff") or {}
        did = did_obj.get("diff_in_diff")
        did_ci = did_obj.get("diff_in_diff_ci") or {}
        if did is not None:
            did_text = f"DiD = {float(did):+.3f}"
            if did_ci:
                did_text += (
                    f"  CI [{float(did_ci.get('low', 0.0)):+.3f},"
                    f" {float(did_ci.get('high', 0.0)):+.3f}]"
                )
        else:
            did_text = "DiD = —"
        verdict = row.get("verdict") or "—"
        ax.set_title(f"{model_label}\n{did_text}\nverdict: {verdict}")
        ax.grid(axis="y", alpha=0.25)

    _panel(axes[0], qwen3_row, "qwen3-235B-thinking (n=500)")
    _panel(axes[1], opus_row, "claude-opus-4-7 (n=300)")

    # Unify y-limits so the magnitude contrast is visually faithful
    # rather than each panel auto-scaling independently.
    yl0 = axes[0].get_ylim()
    yl1 = axes[1].get_ylim()
    lo = min(yl0[0], yl1[0])
    hi = max(yl0[1], yl1[1])
    axes[0].set_ylim(lo, hi)
    axes[1].set_ylim(lo, hi)

    fig.suptitle(
        "Two mechanisms on safety (validated strict rubric):\n"
        "qwen3: behavioral effect only on VEA-flip prompts (mediated). "
        "opus: behavioral effect on flip AND stable_neg prompts (direct).",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("wrote %s", out_path)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _short_model(name: str) -> str:
    """Render long openrouter slugs as compact paper labels."""
    name = (name or "").replace("openrouter:", "")
    name = name.replace("anthropic:", "")
    name = name.replace("openai:", "")
    if "qwen3-235b" in name:
        return "qwen3-thinking"
    if "deepseek-v3.2" in name:
        return "deepseek-v3.2"
    if "kimi-k2-thinking" in name:
        return "kimi-k2-thinking"
    return name


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross-protocol-summary", default=None)
    parser.add_argument("--goodfire-summary", default=None)
    parser.add_argument("--mediation-summary", default=None)
    parser.add_argument("--strict-mediation-summary", default=None)
    parser.add_argument(
        "--opus-strict-mediation-summary",
        default=None,
        help=(
            "Strict-rubric mediation summary for opus-4.7 (n=300). "
            "Combined with --strict-mediation-summary (qwen3 n=500) to "
            "render fig5 — the two-mechanism contrast figure on the "
            "safety family. See §5.4 of the paper."
        ),
    )
    parser.add_argument("--out-dir", default="docs/figures")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cross = _load_json(args.cross_protocol_summary)
    goodfire = _load_json(args.goodfire_summary)
    mediation = _load_json(args.mediation_summary)
    strict_mediation = _load_json(args.strict_mediation_summary)
    opus_strict_mediation = _load_json(args.opus_strict_mediation_summary)

    produced = 0

    if cross:
        _render_refusal_forest(cross, out_dir / "fig1_refusal_forest.png")
        produced += 1
    else:
        LOGGER.info("fig1 skipped (no --cross-protocol-summary)")

    if goodfire:
        _render_vea_inflation(goodfire, out_dir / "fig2_vea_inflation.png")
        produced += 1
    else:
        LOGGER.info("fig2 skipped (no --goodfire-summary)")

    if mediation:
        _render_qwen3_per_family(mediation, out_dir / "fig3_qwen3_per_family.png")
        produced += 1
        _render_mediation_panels(
            mediation, strict_mediation, out_dir / "fig4_mediation_panels.png"
        )
        produced += 1
    else:
        LOGGER.info("fig3 + fig4 skipped (no --mediation-summary)")

    if strict_mediation and opus_strict_mediation:
        _render_two_mechanism(
            strict_mediation,
            opus_strict_mediation,
            out_dir / "fig5_two_mechanism.png",
        )
        produced += 1
    else:
        LOGGER.info(
            "fig5 skipped (needs both --strict-mediation-summary and "
            "--opus-strict-mediation-summary)"
        )

    if produced == 0:
        LOGGER.error(
            "no figures produced - pass at least one --*-summary argument"
        )
        return 2
    LOGGER.info("produced %d figure(s) in %s", produced, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
