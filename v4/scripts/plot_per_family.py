"""Per-family eval/deploy probe-projection histograms.

Projects controlled-holdout activations onto the trained probe coefficient w,
then plots the 1-D probe-score histogram (eval vs deploy) separately for each
framing family, with the per-family AUROC. Shows that probe separability
varies sharply across families. Output is an aggregate plot (no prompt text).

Usage:
    python v4/scripts/plot_per_family.py \\
        --acts v4/runs/gpt_oss_stage2/gpt_oss_120b_controlled_activations.npz \\
        --probes v4/runs/gpt_oss_stage2/frozen_probes_gpt_oss_120b.npz \\
        --layer 34 --model "gpt-oss-120b L34" \\
        --output docs/figures/v4_per_family.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", type=Path, required=True)
    ap.add_argument("--probes", type=Path, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--model", required=True, help="e.g. 'gpt-oss-120b L34'")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    acts = np.load(args.acts, allow_pickle=True)
    probes = np.load(args.probes)
    X = acts[f"X_L{args.layer}"]
    y = acts["y"]
    fam = np.array([f.decode() if isinstance(f, (bytes, bytearray)) else str(f)
                    for f in acts["families"]])
    w = probes[f"coef_L{args.layer}"]
    b = float(probes[f"intercept_L{args.layer}"]) if f"intercept_L{args.layer}" in probes.files else 0.0
    scores = X @ w + b

    families = sorted(set(fam))
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    smin, smax = scores.min(), scores.max()
    bins = np.linspace(smin, smax, 30)
    for ax, fm in zip(axes.ravel(), families):
        m = fam == fm
        auroc = roc_auc_score(y[m], scores[m])
        ax.hist(scores[m & (y == 0)], bins=bins, color="tab:blue", alpha=0.6,
                label=f"deploy (n={int((m & (y == 0)).sum())})")
        ax.hist(scores[m & (y == 1)], bins=bins, color="tab:red", alpha=0.6,
                label=f"eval (n={int((m & (y == 1)).sum())})")
        ax.axvline(0, color="black", ls="--", lw=1)
        ax.set_title(f"Family {fm} (AUROC {auroc:.3f})", fontsize=10)
        ax.set_xlabel("probe score")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"{args.model} eval/deploy probe projection by framing family",
        fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=120, bbox_inches="tight")
    for fm in families:
        m = fam == fm
        print(f"Family {fm}: AUROC {roc_auc_score(y[m], scores[m]):.3f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
