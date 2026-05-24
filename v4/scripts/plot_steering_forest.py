"""Forest plot of Stage-4 steering deltas: probe direction vs controls (Qwen3-32B L55).

Reads the two judged jsonl files, computes paired Δ vs the α=0 baseline (pairing
within each file to avoid judge-nondeterminism across batches), and renders a
3-panel forest plot (refusal / sandbagging / hedging) at α=+20 for both judges.

Output: docs/figures/v4_steering_forest.png  (aggregate deltas only; no prompt text).

Usage:
    python v4/scripts/plot_steering_forest.py
"""
from __future__ import annotations
import json
import math
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
S4 = ROOT / "v4" / "runs" / "stage4"


def load(name):
    with open(S4 / name) as f:
        return [json.loads(line) for line in f]


def num(v):
    if v is True:
        return 1.0
    if v is False:
        return 0.0
    return None if v is None else float(v)


def baseline(recs):
    return {r["prompt_id"]: r["judges"] for r in recs
            if r.get("layer") == 55 and r.get("alpha") == 0.0
            and r.get("condition") == "mean_diff_zscored"}


def cell(recs, cond, layer, alpha):
    return {r["prompt_id"]: r["judges"] for r in recs
            if r.get("condition") == cond and r.get("layer") == layer
            and r.get("alpha") == alpha}


def delta(steer, base, judge, metric):
    ds = []
    for pid, jd in steer.items():
        if pid not in base:
            continue
        sv = num(jd.get(judge, {}).get(metric))
        bv = num(base[pid].get(judge, {}).get(metric))
        if sv is None or bv is None:
            continue
        ds.append(sv - bv)
    if len(ds) < 2:
        return float("nan"), float("nan")
    return st.mean(ds), st.stdev(ds) / math.sqrt(len(ds))


def main():
    controls = load("qwen3_L55_controls_judged.jsonl")
    pvr = load("qwen3_L55_probe_vs_random_judged.jsonl")
    b_pvr, b_ctl = baseline(pvr), baseline(controls)

    # (label, recs, baseline, condition, layer)
    conds = [
        ("probe (mean_diff)", pvr, b_pvr, "mean_diff_zscored", 55),
        ("probe-coef",        controls, b_ctl, "probe_coef_zscored", 55),
        ("random",            pvr, b_pvr, "random", 55),
        ("orthogonal",        controls, b_ctl, "orthogonal", 55),
        ("wrong-layer L25",   controls, b_ctl, "mean_diff_zscored", 25),
    ]
    metrics = [("refusal", "Δ refusal"), ("sandbagging", "Δ sandbagging"), ("hedging", "Δ hedging")]
    judges = [("opus", "Judge A"), ("gpt5", "Judge B")]
    colors = {"opus": "#1f77b4", "gpt5": "#d62728"}
    offset = {"opus": 0.12, "gpt5": -0.12}
    ALPHA = 20.0

    n_c = len(conds)
    ypos = list(range(n_c))[::-1]  # probe at top
    probe_y = ypos[0]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, (m, mlabel) in zip(axes, metrics):
        ax.axhspan(probe_y - 0.45, probe_y + 0.45, color="0.92", zorder=0)
        ax.axvline(0, ls="--", color="0.4", lw=1, zorder=1)
        for (label, recs, base, cond, layer), y in zip(conds, ypos):
            for jk, _ in judges:
                mu, se = delta(cell(recs, cond, layer, ALPHA), base, jk, m)
                ax.errorbar(mu, y + offset[jk], xerr=2 * se, fmt="o", ms=4,
                            capsize=2, color=colors[jk], zorder=3)
        ax.set_title(mlabel, fontsize=10)
        ax.set_xlabel("Δ vs α=0  (±2 SE)", fontsize=8)
        ax.grid(axis="x", ls=":", alpha=0.4)

    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([c[0] for c in conds], fontsize=9)
    fig.suptitle(
        "Qwen3-32B L55 steering at α=+20: probe direction vs controls\n"
        "No condition exceeds the others or replicates across both judges",
        fontsize=11)
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=colors[j], label=n)
               for j, n in judges]
    axes[2].legend(handles=handles, fontsize=8, loc="lower right")
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    out = ROOT / "docs" / "figures" / "v4_steering_forest.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
