"""
V4b Step A: train + freeze per-layer probes from existing Qwen3 activations.

Reads `qwen3_activations.npz` (Microsoft triggers training activations at
9 layers), trains a logistic regression per layer, and saves the probe
coefficients to disk in a format that's portable across sklearn versions
(numpy + JSON metadata).

The frozen probes are then independent artifacts — we can score any new
activation (e.g., SAD prompts in Step E) without re-training.

Saved files:
  frozen_probes_qwen3.npz     -- coef + intercept per layer (small numpy arrays)
  frozen_probes_qwen3_meta.json -- training config, hashes, CV AUROCs

Usage:
    python v4/scripts/freeze_qwen3_probes.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

LAYERS = [10, 20, 30, 40, 50, 55, 60, 62, 63]
LR_C = 1.0  # baseline; we know L55 stays peak across C in {0.01..100}
LR_MAX_ITER = 2000


def _sha256_bytes(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path,
                        default=Path("v4/runs/qwen3_stage2/qwen3_activations.npz"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("v4/runs/qwen3_stage2"))
    parser.add_argument("--C", type=float, default=LR_C)
    args = parser.parse_args()

    print(f"Loading activations from {args.npz}")
    data = np.load(args.npz)
    train_y = data["train_y"]
    test_y = data["test_y"]
    print(f"  train_y: shape={train_y.shape}, counts={np.bincount(train_y)}")
    print(f"  test_y: shape={test_y.shape}, counts={np.bincount(test_y)}")

    coefs = {}
    intercepts = {}
    metadata = {
        "training_data_source": "Microsoft triggers.json (real=0, hypothetical=1)",
        "training_npz_path": str(args.npz),
        "training_y_sha256": _sha256_bytes(train_y),
        "n_train": int(len(train_y)),
        "lr_C": float(args.C),
        "lr_max_iter": int(LR_MAX_ITER),
        "layers": LAYERS,
        "per_layer": {},
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\nTraining + freezing one probe per layer (C={args.C})...")
    print(f"{'Layer':>6} {'d_model':>9} {'CV mean':>10} {'CV std':>10} {'Test':>10}")
    print("-" * 55)
    for layer in LAYERS:
        Xtr = data[f"train_X_L{layer}"]
        Xte = data[f"test_X_L{layer}"]

        cv_aurocs = []
        for tr_i, va_i in skf.split(Xtr, train_y):
            clf_cv = LogisticRegression(max_iter=LR_MAX_ITER, C=args.C)
            clf_cv.fit(Xtr[tr_i], train_y[tr_i])
            va_scores = clf_cv.predict_proba(Xtr[va_i])[:, 1]
            cv_aurocs.append(roc_auc_score(train_y[va_i], va_scores))

        final_clf = LogisticRegression(max_iter=LR_MAX_ITER, C=args.C)
        final_clf.fit(Xtr, train_y)
        coefs[f"coef_L{layer}"] = final_clf.coef_[0].astype(np.float32)
        intercepts[f"intercept_L{layer}"] = float(final_clf.intercept_[0])

        test_scores = final_clf.predict_proba(Xte)[:, 1]
        test_auroc = float(roc_auc_score(test_y, test_scores))

        cv_mean = float(np.mean(cv_aurocs))
        cv_std = float(np.std(cv_aurocs))
        print(f"L{layer:>4}  {Xtr.shape[1]:>9} {cv_mean:>10.3f} {cv_std:>10.3f} {test_auroc:>10.3f}")

        metadata["per_layer"][f"L{layer}"] = {
            "d_model": int(Xtr.shape[1]),
            "cv_mean_auroc": cv_mean,
            "cv_std_auroc": cv_std,
            "test_auroc_on_ed_01_to_10": test_auroc,
            "coef_sha256": _sha256_bytes(coefs[f"coef_L{layer}"]),
            "intercept": intercepts[f"intercept_L{layer}"],
        }

    out_npz = args.out_dir / "frozen_probes_qwen3.npz"
    out_meta = args.out_dir / "frozen_probes_qwen3_meta.json"
    np.savez_compressed(
        out_npz,
        **coefs,
        **{k: np.array(v, dtype=np.float32) for k, v in intercepts.items()},
    )
    with open(out_meta, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved frozen probes: {out_npz}")
    print(f"Saved metadata: {out_meta}")
    print(f"\nTo score new activations:")
    print(f"  probes = np.load('{out_npz}')")
    print(f"  coef_L55 = probes['coef_L55']")
    print(f"  intercept_L55 = float(probes['intercept_L55'])")
    print(f"  logit = X @ coef_L55 + intercept_L55")
    print(f"  prob = 1 / (1 + np.exp(-logit))")


if __name__ == "__main__":
    main()
