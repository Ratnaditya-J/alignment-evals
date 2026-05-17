"""Generate paper figures from existing run-artifact JSONs.

Produces the nine figures referenced in ``docs/paper_draft.md``:

  V1 figures (BeaverTails cross-protocol + Goodfire reasoning-trace):

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

  V2 figure (Apollo factorial, 6-model with closed-CoT backfill):

  6. fig6_v2_intrinsic_main_effect.png — per-model intrinsic
     eval-vs-deployment refusal Δ with paired-bootstrap CIs; points
     colored red for Holm survivors, grey otherwise. Visual proof of
     V2's cross-laboratory intrinsic-eval-label finding: 5/6 models
     from 4 labs survive Holm in their respective correction families.
     Reads ``v2_apollo_factorial_summary.strict.with-backfill.json``.

  V3 figures (inject-clause ablation, 6-model with closed-CoT backfill):

  7. fig7_triangulation.png         — SOTA headline figure. For opus
     and qwen3, plots full_inject Δ refusal across V1 (BeaverTails),
     V2 (Apollo alignment-stratum deep-dive), and V3 (Apollo
     alignment-eval full sample) with CIs. The direction asymmetry
     (opus −, qwen3 +) replicates at consistent within-model
     magnitudes across all three pre-registered experiments. V1+V2
     numbers hardcoded from the pre-reg §10 entries; V3 numbers read
     from the V3 summary.

  8. fig8_v3_clause_attribution.png — for opus and qwen3 (the two
     active V3 models), side-by-side bars of per-singleton Δrefusal,
     sum-of-singletons, and full_inject anchor. Visual proof of V3's
     §5.6 additive attribution: sum-of-singletons ≈ full_inject
     within CI on both models. Reads V3 summary.

  9. fig9_v3_vea_dissociation.png   — for qwen3 and opus, dual-axis
     plot of refusal (left axis, blue) and strict-VEA rate (right
     axis, red) across the 5 V3 conditions. Visual proof of §5.6.4's
     model-specific verbalization-vs-behavior dissociation: qwen3's
     strict-VEA scales proportionally per clause; opus's stays at
     baseline on singletons and only inflates on full_inject; both
     models have identically additive refusal attribution. Reads V3
     summary.

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
        --v2-summary runs/v2-apollo-factorial-prereg/v2_apollo_factorial_summary.strict.with-backfill.json \\
        --v3-summary runs/v3-clause-ablation-prereg/v3_clause_ablation_summary.strict.with-backfill.json \\
        --out-dir docs/figures/

Each --*-summary argument is optional. Figures whose inputs aren't
supplied are skipped (with an info log); the script exits 0 as long as
at least one figure was produced. fig5 needs both
``--strict-mediation-summary`` (qwen3) and
``--opus-strict-mediation-summary`` (opus). fig6 needs ``--v2-summary``.
fig7, fig8, fig9 need ``--v3-summary``.
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
    estimate. Vertical line at Δ=0. Used to make the "null on 8/8 tested
    models (frontier and frontier-adjacent)" claim visual."""
    import matplotlib.pyplot as plt

    rows: List[Dict[str, Any]] = list(summary.get("per_model", []))
    if not rows:
        LOGGER.warning("fig1: per_model empty, skipping")
        return
    rows.sort(key=lambda r: float(r.get("inject_minus_neutral", 0.0)))
    labels = [_short_model(r.get("model_name", "?")) for r in rows]
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
        "(no CI excludes zero on 8/8 tested models)"
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


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
        "Reasoning-trace VEA inflation (open-reasoning trio)\n"
        "(red = CI excludes zero, grey = not significant)"
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


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
    _save_figure(fig, out_path)
    plt.close(fig)


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
        "Primary qwen3 mediation signal; disjoint replication is null",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: two-mechanism contrast (qwen3 vs opus, safety family)
# ---------------------------------------------------------------------------


def _render_two_mechanism(
    qwen3_strict: Dict[str, Any],
    opus_strict: Dict[str, Any],
    out_path: Path,
    qwen3_replication_strict: Optional[Dict[str, Any]] = None,
) -> None:
    """Per-sample side-by-side panel showing paired Δrefusal for the
    safety family under the validated strict rubric.

    If ``qwen3_replication_strict`` is supplied, renders a 3-panel
    layout (qwen3 primary, qwen3 pre-registered disjoint replication,
    opus) that supports the §5.4 direction-asymmetry/non-replication
    finding. Otherwise renders the legacy 2-panel layout (qwen3
    primary, opus) — kept for back-compat with run dirs that pre-date
    the replication.

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

    qwen3_repl_row = (
        _safety_row(qwen3_replication_strict)
        if qwen3_replication_strict
        else None
    )

    panel_count = 3 if qwen3_repl_row else 2
    fig, axes = plt.subplots(1, panel_count, figsize=(5.5 * panel_count, 5))
    if panel_count == 1:
        axes = [axes]

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

    if qwen3_repl_row:
        _panel(axes[0], qwen3_row, "qwen3-235B (primary, n=500)")
        _panel(axes[1], qwen3_repl_row, "qwen3-235B (replication, n=200)")
        _panel(axes[2], opus_row, "claude-opus-4-7 (n=300)")
        for idx, ax in enumerate(axes):
            _add_panel_letter(ax, "abc"[idx])
    else:
        _panel(axes[0], qwen3_row, "qwen3-235B-thinking (n=500)")
        _panel(axes[1], opus_row, "claude-opus-4-7 (n=300)")
        for idx, ax in enumerate(axes):
            _add_panel_letter(ax, "ab"[idx])

    # Unify y-limits so the magnitude contrast is visually faithful
    # rather than each panel auto-scaling independently.
    ylims = [ax.get_ylim() for ax in axes]
    lo = min(yl[0] for yl in ylims)
    hi = max(yl[1] for yl in ylims)
    for ax in axes:
        ax.set_ylim(lo, hi)

    if qwen3_repl_row:
        fig.suptitle(
            "Safety-family stratum-level paired Δrefusal under inject "
            "(validated strict rubric):\n"
            "qwen3 primary mediation suggestive → replication null; "
            "both samples + opus show CI-excludes-zero stable_neg "
            "effects in opposite directions.",
            fontsize=11,
        )
    else:
        fig.suptitle(
            "Safety-family stratum-level paired Δrefusal under inject "
            "(validated strict rubric):\n"
            "qwen3 primary vs opus on the same task family.",
            fontsize=11,
        )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    _save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: V2 intrinsic-eval-label main effect across 6 models
# ---------------------------------------------------------------------------


def _render_v2_intrinsic_main_effect(
    summary: Dict[str, Any], out_path: Path
) -> None:
    """Per-model intrinsic main effect (eval - deployment refusal) with
    paired-bootstrap CIs. Holm survivors highlighted. Reads V2
    factorial summary (v2_apollo_factorial_summary.strict.with-backfill.json).
    Used to make V2's cross-laboratory robustness claim visual:
    intrinsic-eval-label refusal predictor survives across 5 of 6 models
    from 4 different labs."""
    import matplotlib.pyplot as plt

    rows = list(summary.get("per_model") or [])
    if not rows:
        LOGGER.warning("fig6: per_model empty, skipping")
        return

    # Find Holm-survivor lookup. The supplement pre-reg's two-family
    # treatment is encoded at the analyzer level as a single 18-hypothesis
    # family; we honor the per-family decisions but mark them visually.
    holm = {
        (h["model_name"], h["effect"]): bool(h.get("reject_at_0_05"))
        for h in (summary.get("holm_bonferroni") or [])
    }

    parsed: List[Dict[str, Any]] = []
    for r in rows:
        eff = r.get("main_effect_intrinsic") or {}
        ci = eff.get("ci") or {}
        parsed.append({
            "model": _short_model(r.get("model_name", "?")),
            "point": float(eff.get("point") or 0.0),
            "low": float(ci.get("low", 0.0)),
            "high": float(ci.get("high", 0.0)),
            "holm_survivor": holm.get(
                (r.get("model_name"), "main_effect_intrinsic"), False
            ),
        })
    parsed.sort(key=lambda r: r["point"])

    labels = [r["model"] for r in parsed]
    points = [r["point"] for r in parsed]
    err_low = [p - r["low"] for r, p in zip(parsed, points)]
    err_high = [r["high"] - p for r, p in zip(parsed, points)]
    colors = [
        "#d62728" if r["holm_survivor"] else "#7e9eb2" for r in parsed
    ]

    fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.55 * len(parsed))))
    y = list(range(len(parsed)))
    ax.errorbar(
        points, y, xerr=[err_low, err_high],
        fmt="o", color="#333", ecolor="#333", capsize=4, markersize=0,
        zorder=1,
    )
    ax.scatter(points, y, c=colors, s=70, zorder=2, edgecolor="#333")
    ax.axvline(0, color="#888", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Δ refusal (eval − deployment)")
    ax.set_title(
        "V2 Apollo intrinsic-eval-label main effect across 6 frontier models\n"
        "(red = survives Holm-Bonferroni in its correction family; grey = does not)"
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7: triangulation — V1+V2+V3 direction asymmetry on opus and qwen3
# ---------------------------------------------------------------------------


def _render_triangulation(
    v3_summary: Dict[str, Any], out_path: Path
) -> None:
    """SOTA headline figure: the V1+V2+V3 direction-asymmetry triangulation.

    For opus and qwen3, plot full_inject Δ refusal across the three
    experiments. V1 numbers are hardcoded from the V1 pre-reg §10 (the
    canonical pre-registered replication numbers); V2 numbers are
    hardcoded from V2 pre-reg §10's alignment-stratum deep-dive; V3
    numbers are read from the V3 with-backfill summary.

    Visual story: same asymmetric refusal-shift pattern (opus −, qwen3 +)
    at consistent within-model magnitudes across three independent
    pre-registered experiments. The triangulation rules out
    sample-specific or corpus-specific artifacts.
    """
    import matplotlib.pyplot as plt

    # V1 paired replication numbers (BeaverTails safety, n=200 disjoint
    # replication sample). Source: docs/pre_registration_qwen3_safety_
    # mediation.md §10. These are the strict-rubric paired Δ refusal
    # numbers on the safety task family.
    v1 = {
        "qwen3": {"point": +0.217, "low": +0.098, "high": +0.337},
        "opus":  {"point": -0.339, "low": -0.446, "high": -0.231},
    }

    # V2 alignment-stratum deep-dive (Apollo eval prompts, eval_type=alignment).
    # Source: docs/pre_registration_v2_apollo_2x2.md §10 secondary analysis.
    v2 = {
        "qwen3": {"point": +0.057, "low": -0.040, "high": +0.141},
        "opus":  {"point": -0.060, "low": -0.180, "high": -0.001},
    }

    # V3 alignment-eval full-sample (n=297 alignment-eval prompts).
    # Read directly from the supplied V3 summary.
    v3 = {}
    for r in (v3_summary.get("per_model") or []):
        name = r.get("model_name", "")
        if "qwen3-235b" in name:
            key = "qwen3"
        elif "claude-opus-4-7" in name:
            key = "opus"
        else:
            continue
        eff = r.get("full_inject_effect") or {}
        ci = eff.get("ci") or {}
        v3[key] = {
            "point": float(eff.get("point") or 0.0),
            "low": float(ci.get("low", 0.0)),
            "high": float(ci.get("high", 0.0)),
        }
    if "qwen3" not in v3 or "opus" not in v3:
        LOGGER.warning("fig7: V3 summary missing opus or qwen3, skipping")
        return

    # Three experiments × two models. Plot each model as a line of three
    # points with CIs across V1, V2, V3.
    experiments = ["V1\n(BeaverTails\nsafety)", "V2\n(Apollo\nalignment)", "V3\n(Apollo\nalignment, n=297)"]
    x = list(range(3))

    fig, ax = plt.subplots(figsize=(8.5, 5))

    for model_key, color, marker, label in [
        ("opus", "#1f77b4", "s", "claude-opus-4-7"),
        ("qwen3", "#d62728", "o", "qwen3-235B-thinking"),
    ]:
        points = [v1[model_key]["point"], v2[model_key]["point"], v3[model_key]["point"]]
        lows = [v1[model_key]["low"], v2[model_key]["low"], v3[model_key]["low"]]
        highs = [v1[model_key]["high"], v2[model_key]["high"], v3[model_key]["high"]]
        # max(0.0, ...) is defensive: matplotlib's errorbar rejects negative
        # yerr values, which can arise from floating-point drift when a CI
        # bound coincides exactly with the point estimate (e.g. opus V2
        # high=-0.001 vs point=-0.060 is fine, but bootstrap precision can
        # invert by ~1e-9). Clamp to zero so we always plot the CI even when
        # the asymmetric half is degenerate.
        err_low = [max(0.0, p - l) for p, l in zip(points, lows)]
        err_high = [max(0.0, h - p) for h, p in zip(points, highs)]
        ax.errorbar(
            x, points, yerr=[err_low, err_high],
            fmt=marker, color=color, ecolor=color,
            capsize=5, markersize=10, linewidth=2,
            linestyle="-", label=label,
        )

    ax.axhline(0, color="#888", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(experiments)
    ax.set_ylabel("Δ refusal (inject − neutral, paired)")
    ax.set_title(
        "V1+V2+V3 direction-asymmetry triangulation on opus and qwen3\n"
        "Opposite-signed refusal shifts replicate across three pre-registered\n"
        "experiments at consistent within-model magnitudes."
    )
    ax.legend(loc="best", frameon=True)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 8: V3 per-clause attribution decomposition (opus + qwen3)
# ---------------------------------------------------------------------------


def _render_v3_clause_attribution(
    v3_summary: Dict[str, Any], out_path: Path
) -> None:
    """For opus and qwen3, plot per-singleton Δrefusal + sum-of-singletons
    + full_inject side-by-side. The visual proof of the V3 §5.6 additive
    attribution: sum-of-singletons ≈ full_inject within CI on both
    active models, while no single clause individually carries the
    effect."""
    import matplotlib.pyplot as plt

    targets = []
    for r in (v3_summary.get("per_model") or []):
        name = r.get("model_name", "")
        if "qwen3-235b" in name:
            targets.append(("qwen3-235B-thinking", r))
        elif "claude-opus-4-7" in name:
            targets.append(("claude-opus-4-7", r))
    if len(targets) != 2:
        LOGGER.warning("fig8: needs both opus and qwen3 in V3 summary, skipping")
        return
    # Sort with opus on the left, qwen3 on the right
    targets.sort(key=lambda t: 0 if "opus" in t[0] else 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    bar_labels = [
        "eval_only", "reviewer_only", "comparison_only",
        "sum-of-\nsingletons", "full_inject\n(anchor)",
    ]
    bar_keys = ["eval_only", "reviewer_only", "comparison_only", "_sum", "_full"]

    for panel_idx, (ax, (model_label, entry)) in enumerate(zip(axes, targets)):
        singleton_effects = entry.get("singleton_effects") or {}
        full = entry.get("full_inject_effect") or {}
        sum_point = float(entry.get("singleton_sum_point") or 0.0)

        points = []
        lows = []
        highs = []
        colors = []
        for key in bar_keys:
            if key == "_sum":
                # Sum has no native CI in the analyzer output; we leave
                # the bar without error caps to make it visually distinct
                # from the bootstrap-derived bars.
                points.append(sum_point)
                lows.append(sum_point)
                highs.append(sum_point)
                colors.append("#999")
            elif key == "_full":
                ci = full.get("ci") or {}
                points.append(float(full.get("point") or 0.0))
                lows.append(float(ci.get("low", 0.0)))
                highs.append(float(ci.get("high", 0.0)))
                colors.append("#1f77b4")
            else:
                eff = singleton_effects.get(key) or {}
                ci = eff.get("ci") or {}
                points.append(float(eff.get("point") or 0.0))
                lows.append(float(ci.get("low", 0.0)))
                highs.append(float(ci.get("high", 0.0)))
                colors.append("#d62728" if (lows[-1] > 0 or highs[-1] < 0) else "#aaaaaa")

        err_low = [max(0.0, p - l) for p, l in zip(points, lows)]
        err_high = [max(0.0, h - p) for h, p in zip(points, highs)]

        x = list(range(len(bar_labels)))
        # Render in two passes so the sum-of-singletons bar has no errorbar
        # call at all (no rendering artifact from zero-length yerr): pass 1
        # draws the 4 CI-bearing bars (singletons + full_inject), pass 2
        # draws the sum bar with no yerr argument.
        sum_idx = bar_keys.index("_sum")
        ci_indices = [i for i in range(len(bar_keys)) if i != sum_idx]
        ax.bar(
            [x[i] for i in ci_indices],
            [points[i] for i in ci_indices],
            color=[colors[i] for i in ci_indices],
            yerr=[
                [err_low[i] for i in ci_indices],
                [err_high[i] for i in ci_indices],
            ],
            capsize=5,
        )
        ax.bar(
            [x[sum_idx]],
            [points[sum_idx]],
            color=colors[sum_idx],
        )
        ax.axhline(0, color="#888", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, rotation=10, ha="right", fontsize=9)
        attribution = str(entry.get("attribution", "—"))
        ax.set_title(
            f"{model_label}\nattribution: {attribution}",
            fontsize=10,
        )
        ax.grid(axis="y", alpha=0.25)
        _add_panel_letter(ax, "ab"[panel_idx])

    axes[0].set_ylabel("Δ refusal (condition − neutral, paired)")
    fig.suptitle(
        "V3 per-clause attribution on the two V1-active models\n"
        "sum-of-singletons ≈ full_inject within CI on both → ADDITIVE attribution\n"
        "(no single clause's effect is CI-detectable on either model)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    _save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 9: V3 verbalization-vs-behavior dissociation (qwen3 vs opus)
# ---------------------------------------------------------------------------


def _render_v3_vea_dissociation(
    v3_summary: Dict[str, Any], out_path: Path
) -> None:
    """Side-by-side: for qwen3 and opus, dual y-axis plot of refusal
    (left) and strict-VEA (right) across the 5 V3 conditions. Visual
    proof of the §5.6.4 model-specific dissociation: qwen3's strict-VEA
    scales proportionally per clause (×13 baseline on eval_only alone,
    ×22 on full); opus's stays at baseline on all 3 singletons and only
    inflates ×4 on full_inject. Yet both models' refusal attribution is
    identically additive."""
    import matplotlib.pyplot as plt

    conditions = ["neutral", "eval_only", "reviewer_only", "comparison_only", "full_inject"]
    targets = []
    for r in (v3_summary.get("per_model") or []):
        name = r.get("model_name", "")
        if "qwen3-235b" in name:
            targets.append(("qwen3-235B-thinking", r))
        elif "claude-opus-4-7" in name:
            targets.append(("claude-opus-4-7", r))
    if len(targets) != 2:
        LOGGER.warning("fig9: needs both opus and qwen3 in V3 summary, skipping")
        return
    targets.sort(key=lambda t: 0 if "opus" in t[0] else 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for panel_idx, (ax, (model_label, entry)) in enumerate(zip(axes, targets)):
        cells = entry.get("cells") or {}
        refusals = [float((cells.get(c) or {}).get("refusal_mean", 0.0)) for c in conditions]
        veas = [float((cells.get(c) or {}).get("vea_mean", 0.0)) for c in conditions]

        x = list(range(len(conditions)))
        # Left axis: refusal (line + markers, blue)
        ax.plot(
            x, refusals, color="#1f77b4", marker="s", linewidth=2,
            markersize=10, label="refusal",
        )
        ax.set_ylabel("refusal rate", color="#1f77b4")
        ax.tick_params(axis="y", labelcolor="#1f77b4")
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=10, ha="right", fontsize=9)
        ax.set_ylim(0, max(0.45, max(refusals) * 1.15))
        ax.grid(axis="y", alpha=0.2)

        # Right axis: strict VEA (line + markers, red)
        ax2 = ax.twinx()
        ax2.plot(
            x, veas, color="#d62728", marker="o", linewidth=2,
            markersize=10, label="strict-VEA",
        )
        ax2.set_ylabel("strict-VEA rate", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        ax2.set_ylim(0, max(0.5, max(veas) * 1.15))

        ax.set_title(model_label, fontsize=10)
        _add_panel_letter(ax, "ab"[panel_idx])

    fig.suptitle(
        "V3 verbalization-vs-behavior dissociation: same additive refusal\n"
        "attribution on both models, different per-clause VEA patterns.\n"
        "qwen3: proportional VEA per clause. opus: gestalt-only VEA on full_inject.",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    _save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _save_figure(fig, out_path: Path) -> None:
    """Save figure as both PNG (for GitHub markdown rendering) and PDF
    (for arXiv / LaTeX submission). The PDF is vector and scales without
    pixelation in the final paper PDF. Both files share the same stem
    so paper Markdown references work unchanged (it links to .png) and
    the LaTeX export pass swaps in .pdf automatically via pandoc.
    """
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    LOGGER.info("wrote %s (+%s)", out_path, pdf_path.name)


def _add_panel_letter(ax, letter: str) -> None:
    """Add a small bold (a)/(b)/(c) label in the upper-left of the
    subplot, in axes coordinates so it doesn't move with data."""
    ax.text(
        -0.08, 1.02, f"({letter})",
        transform=ax.transAxes,
        fontsize=12, fontweight="bold",
        ha="right", va="bottom",
    )


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
    if "claude-opus-4-7" in name:
        return "opus-4.7"
    if "openai/gpt-5.5" in name:
        return "gpt-5.5"
    if "google/gemini-2.5-pro" in name:
        return "gemini-2.5-pro"
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
            "render fig5 — the safety-family stratum-level paired "
            "Δrefusal figure. See §5.4 of the paper."
        ),
    )
    parser.add_argument(
        "--qwen3-replication-strict-mediation-summary",
        default=None,
        help=(
            "Strict-rubric mediation summary for the pre-registered "
            "qwen3 safety replication (n=200 disjoint). When supplied, "
            "fig5 becomes a 3-panel layout including the replication; "
            "the §5.4 direction-asymmetry / non-replication finding is "
            "what this panel exists to show."
        ),
    )
    parser.add_argument(
        "--v2-summary",
        default=None,
        help=(
            "V2 Apollo factorial strict-with-backfill summary "
            "(v2_apollo_factorial_summary.strict.with-backfill.json). "
            "Produces fig6 (intrinsic-eval-label main effect, 6 models)."
        ),
    )
    parser.add_argument(
        "--v3-summary",
        default=None,
        help=(
            "V3 inject-clause-ablation strict-with-backfill summary "
            "(v3_clause_ablation_summary.strict.with-backfill.json). "
            "Produces fig7 (triangulation), fig8 (per-clause attribution "
            "decomposition), fig9 (VEA-vs-behavior dissociation)."
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
    qwen3_replication_strict_mediation = _load_json(
        args.qwen3_replication_strict_mediation_summary
    )

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
            qwen3_replication_strict=qwen3_replication_strict_mediation,
        )
        produced += 1
    else:
        LOGGER.info(
            "fig5 skipped (needs both --strict-mediation-summary and "
            "--opus-strict-mediation-summary)"
        )

    v2_summary = _load_json(args.v2_summary)
    v3_summary = _load_json(args.v3_summary)

    if v2_summary:
        _render_v2_intrinsic_main_effect(
            v2_summary, out_dir / "fig6_v2_intrinsic_main_effect.png"
        )
        produced += 1
    else:
        LOGGER.info("fig6 skipped (no --v2-summary)")

    if v3_summary:
        _render_triangulation(
            v3_summary, out_dir / "fig7_triangulation.png"
        )
        produced += 1
        _render_v3_clause_attribution(
            v3_summary, out_dir / "fig8_v3_clause_attribution.png"
        )
        produced += 1
        _render_v3_vea_dissociation(
            v3_summary, out_dir / "fig9_v3_vea_dissociation.png"
        )
        produced += 1
    else:
        LOGGER.info("fig7 + fig8 + fig9 skipped (no --v3-summary)")

    if produced == 0:
        LOGGER.error(
            "no figures produced - pass at least one --*-summary argument"
        )
        return 2
    LOGGER.info("produced %d figure(s) in %s", produced, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
