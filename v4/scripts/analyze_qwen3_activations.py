"""
V4b offline analysis on the Stage 2 activations npz.

Runs locally on your laptop (no GPU needed). Loads the saved Qwen3-32B
activations from `qwen3_activations.npz`, then does several analyses that
strengthen the V4b paper section without spending another GPU dollar:

  1. Per-pair scores at the peak layer (L55) -- which pairs are easy/hard?
  2. Probe robustness sweep: vary L2 regularization (C in [0.01, 0.1, 1.0, 10, 100])
     and try a small MLP probe. Does L55 stay peak?
  3. Cross-layer probe direction similarity -- are L40-L55 probes pointing the
     same way (distributed feature) or are they distinct (privileged location)?
  4. Probe-direction analysis: which TRAINING pairs are nearest to the probe
     direction at L55? Gives qualitative interpretability for the methods section.

Usage:
    python v4/scripts/analyze_qwen3_activations.py \\
        --npz v4/runs/qwen3_stage2/qwen3_activations.npz \\
        --output v4/runs/qwen3_stage2/analysis.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

LAYERS = [10, 20, 30, 40, 50, 55, 60, 62, 63]

OUR_PAIR_IDS = [f"ed_{i:02d}" for i in range(1, 11)]


def _per_pair_records_from_test(test_scores: np.ndarray, test_y: np.ndarray) -> list[dict]:
    """Test set is ordered: ed_01/eval, ed_01/deploy, ed_02/eval, ..., ed_10/deploy."""
    records = []
    for i, pid in enumerate(OUR_PAIR_IDS):
        eval_score = float(test_scores[2*i])
        deploy_score = float(test_scores[2*i + 1])
        eval_y = int(test_y[2*i])
        deploy_y = int(test_y[2*i + 1])
        assert eval_y == 1 and deploy_y == 0, f"Unexpected test_y at pair {pid}"
        records.append({
            "pair_id": pid,
            "eval_score": eval_score,
            "deploy_score": deploy_score,
            "spread": eval_score - deploy_score,
            "correctly_ranked": eval_score > deploy_score,
        })
    return records


def analyze(npz_path: Path, output_path: Path) -> None:
    print(f"Loading activations from {npz_path}")
    data = np.load(npz_path)
    train_y = data["train_y"]
    test_y = data["test_y"]
    print(f"  train_y: shape={train_y.shape}, counts={np.bincount(train_y)}")
    print(f"  test_y: shape={test_y.shape}, counts={np.bincount(test_y)}")

    results: dict = {"layers": LAYERS, "per_layer": {}, "cross_layer": {}, "robustness": {}}

    # 1. Per-pair scores at the peak layer L55 (and L62 for comparison)
    print("\n=== 1. Per-pair test-set scores at L55 vs L62 ===")
    for layer in (55, 62):
        Xtr, Xte = data[f"train_X_L{layer}"], data[f"test_X_L{layer}"]
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, train_y)
        test_scores = clf.predict_proba(Xte)[:, 1]
        test_auroc = float(roc_auc_score(test_y, test_scores))
        per_pair = _per_pair_records_from_test(test_scores, test_y)
        n_correct = sum(1 for r in per_pair if r["correctly_ranked"])
        results["per_layer"][f"L{layer}"] = {
            "test_auroc": test_auroc,
            "n_pairs_correctly_ranked": n_correct,
            "per_pair": per_pair,
        }
        print(f"\n  L{layer}: test AUROC = {test_auroc:.3f}, "
              f"{n_correct}/{len(per_pair)} pairs correctly ranked")
        sorted_pairs = sorted(per_pair, key=lambda r: r["spread"])
        print(f"  Hardest pairs (lowest spread):")
        for r in sorted_pairs[:3]:
            print(f"    {r['pair_id']}: eval={r['eval_score']:.3f}  "
                  f"deploy={r['deploy_score']:.3f}  spread={r['spread']:+.3f}")
        print(f"  Easiest pairs (highest spread):")
        for r in sorted_pairs[-3:]:
            print(f"    {r['pair_id']}: eval={r['eval_score']:.3f}  "
                  f"deploy={r['deploy_score']:.3f}  spread={r['spread']:+.3f}")

    # 2. Probe robustness at L55
    print("\n=== 2. Probe robustness at L55 ===")
    Xtr55, Xte55 = data["train_X_L55"], data["test_X_L55"]
    robustness = {}
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(max_iter=2000, C=C)
        clf.fit(Xtr55, train_y)
        scores = clf.predict_proba(Xte55)[:, 1]
        auroc = float(roc_auc_score(test_y, scores))
        robustness[f"LR_C={C}"] = auroc
        print(f"  LR C={C:<6}: test AUROC = {auroc:.3f}")
    # MLP probe
    mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, random_state=42)
    mlp.fit(Xtr55, train_y)
    mlp_scores = mlp.predict_proba(Xte55)[:, 1]
    mlp_auroc = float(roc_auc_score(test_y, mlp_scores))
    robustness["MLP(256)"] = mlp_auroc
    print(f"  MLP(256 hidden): test AUROC = {mlp_auroc:.3f}")
    results["robustness"]["L55"] = robustness

    # 3. Cross-layer probe direction similarity
    print("\n=== 3. Cross-layer probe direction similarity ===")
    directions = {}
    for layer in LAYERS:
        Xtr = data[f"train_X_L{layer}"]
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, train_y)
        w = clf.coef_[0]  # (d_model,)
        directions[layer] = w / np.linalg.norm(w)
    cos_matrix = np.zeros((len(LAYERS), len(LAYERS)))
    for i, li in enumerate(LAYERS):
        for j, lj in enumerate(LAYERS):
            cos_matrix[i, j] = float(np.dot(directions[li], directions[lj]))
    print(f"  Probe-direction cosine matrix (rows/cols = layers {LAYERS}):")
    print(f"        " + " ".join(f"L{l:>3}" for l in LAYERS))
    for i, li in enumerate(LAYERS):
        row = " ".join(f"{cos_matrix[i, j]:+.2f}" for j in range(len(LAYERS)))
        print(f"  L{li:<3}  {row}")
    results["cross_layer"]["cosine_matrix"] = cos_matrix.tolist()
    results["cross_layer"]["layers"] = LAYERS

    # 4. Probe-direction analysis: which training points are most aligned with the L55 probe?
    print("\n=== 4. Top training points aligned with L55 probe direction ===")
    Xtr55 = data["train_X_L55"]
    w_l55 = directions[55]
    train_projections = Xtr55 @ w_l55  # (n_train,)
    # Top 5 most "hypothetical-like" (highest projection) and most "real-like" (lowest)
    top_hyp_idx = np.argsort(train_projections)[-5:][::-1]
    top_real_idx = np.argsort(train_projections)[:5]
    print(f"  Top 5 'hypothetical-like' training indices (highest projection):")
    for idx in top_hyp_idx:
        label = "hypothetical" if train_y[idx] == 1 else "real"
        pair_idx = idx // 2  # 2 versions per pair
        print(f"    idx={idx}  proj={train_projections[idx]:+.3f}  label={label}  "
              f"(pair #{pair_idx})")
    print(f"  Top 5 'real-like' training indices (lowest projection):")
    for idx in top_real_idx:
        label = "hypothetical" if train_y[idx] == 1 else "real"
        pair_idx = idx // 2
        print(f"    idx={idx}  proj={train_projections[idx]:+.3f}  label={label}  "
              f"(pair #{pair_idx})")
    results["probe_direction_l55"] = {
        "top_hypothetical_like_indices": [int(i) for i in top_hyp_idx],
        "top_real_like_indices": [int(i) for i in top_real_idx],
        "top_hypothetical_projections": [float(train_projections[i]) for i in top_hyp_idx],
        "top_real_projections": [float(train_projections[i]) for i in top_real_idx],
    }

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved analysis to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path,
                        default=Path("v4/runs/qwen3_stage2/qwen3_activations.npz"))
    parser.add_argument("--output", type=Path,
                        default=Path("v4/runs/qwen3_stage2/analysis.json"))
    args = parser.parse_args()
    analyze(args.npz, args.output)


if __name__ == "__main__":
    main()
