"""
Generate side-by-side probe-projection visualizations for two model+layer
pairs, illustrating the cross-architecture replication of V4b.

For each (model_tag, layer) pair, projects controlled-holdout activations
onto the trained probe's coefficient w, then plots the 1-D histogram of
probe scores (eval vs deploy). This is the paper-worthy figure — the
sigmoid-squashed probabilities (which look saturated under
cross-distribution scoring) are NOT plotted.

Usage:
    python v4/scripts/plot_probe_projection.py \\
        --left-acts v4/runs/qwen3_stage2/controlled_activations_v2.npz \\
        --left-probes v4/runs/qwen3_stage2/frozen_probes_qwen3.npz \\
        --left-layer 55 \\
        --left-title "Qwen3-32B L55 (AUROC 0.917)" \\
        --right-acts v4/runs/gpt_oss_stage2/gpt_oss_120b_controlled_activations.npz \\
        --right-probes v4/runs/gpt_oss_stage2/frozen_probes_gpt_oss_120b.npz \\
        --right-layer 34 \\
        --right-title "gpt-oss-120b L34 (AUROC 0.896)" \\
        --output v4/runs/v4b_cross_arch_probe_projection.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score


def load_and_project(acts_path: Path, probes_path: Path, layer: int):
    acts = np.load(acts_path)
    probes = np.load(probes_path)
    X = acts[f"X_L{layer}"]
    y = acts["y"]
    w = probes[f"coef_L{layer}"]
    b = float(probes[f"intercept_L{layer}"])
    scores = X @ w + b
    auroc = float(roc_auc_score(y, scores))
    return X, y, scores, auroc


def plot_panel(ax, scores, y, title: str, auroc: float) -> None:
    bins = np.linspace(scores.min(), scores.max(), 40)
    ax.hist(scores[y == 0], bins=bins, color="tab:blue", alpha=0.6,
            label=f"deploy (n={int((y == 0).sum())})")
    ax.hist(scores[y == 1], bins=bins, color="tab:red", alpha=0.6,
            label=f"eval (n={int((y == 1).sum())})")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, label="decision boundary")
    ax.set_xlabel("probe score (activation · w + b)")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-acts", type=Path, required=True)
    parser.add_argument("--left-probes", type=Path, required=True)
    parser.add_argument("--left-layer", type=int, required=True)
    parser.add_argument("--left-title", required=True)
    parser.add_argument("--right-acts", type=Path, required=True)
    parser.add_argument("--right-probes", type=Path, required=True)
    parser.add_argument("--right-layer", type=int, required=True)
    parser.add_argument("--right-title", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    _, y_L, scores_L, auroc_L = load_and_project(
        args.left_acts, args.left_probes, args.left_layer)
    _, y_R, scores_R, auroc_R = load_and_project(
        args.right_acts, args.right_probes, args.right_layer)

    print(f"Left  ({args.left_title}): AUROC {auroc_L:.4f}, n={len(y_L)}")
    print(f"Right ({args.right_title}): AUROC {auroc_R:.4f}, n={len(y_R)}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_panel(axes[0], scores_L, y_L, args.left_title, auroc_L)
    plot_panel(axes[1], scores_R, y_R, args.right_title, auroc_R)

    fig.suptitle("Cross-architecture eval/deploy linear-probe projection",
                 fontsize=13)
    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=120, bbox_inches="tight")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
