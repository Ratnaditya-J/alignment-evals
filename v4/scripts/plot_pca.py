"""2-D PCA scatter of controlled-holdout activations, colored by eval/deploy
label (left) and by framing family (right).

The point of the figure: the top two principal components track the framing
family, not the eval/deploy label, so eval/deploy is a smaller shift off the
dominant variance axes. Output is an aggregate scatter (no prompt text).

Usage:
    python v4/scripts/plot_pca.py \\
        --acts v4/runs/qwen3_stage2/controlled_activations_v2.npz \\
        --layer 55 --model "Qwen3-32B L55" \\
        --output docs/figures/v4_pca.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", type=Path, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--model", required=True, help="e.g. 'Qwen3-32B L55'")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    d = np.load(args.acts, allow_pickle=True)
    X = d[f"X_L{args.layer}"]
    y = d["y"]
    fam = np.array([f.decode() if isinstance(f, (bytes, bytearray)) else str(f)
                    for f in d["families"]])

    # Standardize per-dimension before PCA. Without this, a few very
    # high-variance ("rogue") dimensions dominate the top components and can
    # make eval/deploy appear separable on PC1; z-scoring gives each dimension
    # equal weight so the dominant structure (framing family) is what shows.
    Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    pca = PCA(n_components=2, random_state=0).fit(Xs)
    Z = pca.transform(Xs)
    ev = pca.explained_variance_ratio_ * 100.0

    fig, (ax_lab, ax_fam) = plt.subplots(1, 2, figsize=(12, 5))

    for lab, name, col in [(0, "deploy", "tab:blue"), (1, "eval", "tab:red")]:
        m = y == lab
        ax_lab.scatter(Z[m, 0], Z[m, 1], s=14, alpha=0.6, c=col,
                       edgecolors="none", label=name)
    ax_lab.set_title("Colored by eval/deploy label")
    ax_lab.legend(loc="best", fontsize=9)

    cmap = plt.get_cmap("tab10")
    for i, fm in enumerate(sorted(set(fam))):
        m = fam == fm
        ax_fam.scatter(Z[m, 0], Z[m, 1], s=14, alpha=0.6, color=cmap(i),
                       edgecolors="none", label=f"family {fm}")
    ax_fam.set_title("Colored by framing family")
    ax_fam.legend(loc="best", fontsize=9)

    for ax in (ax_lab, ax_fam):
        ax.set_xlabel(f"PC1 ({ev[0]:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({ev[1]:.1f}% variance)")
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"{args.model} holdout activations (2-D PCA): "
        "top components track framing family, not eval/deploy",
        fontsize=12)
    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=120, bbox_inches="tight")
    print(f"PC1 {ev[0]:.1f}%, PC2 {ev[1]:.1f}% (sum {ev[:2].sum():.1f}%)")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
