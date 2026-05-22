"""
Train + freeze per-layer logistic-regression probes from extracted activations.

Generic version: works with any model's *_triggers_activations.npz produced by
extract_acts_any_model_pod.py. Replaces the Qwen3-specific freeze_qwen3_probes.py.

Saves coefficients + intercepts as a portable .npz with metadata, so we can
score new activations later without re-loading sklearn.

Usage:
    python train_probes_any_model.py \\
        --acts v4/runs/qwen3_stage2/qwen3_32b_triggers_activations.npz \\
        --model-tag qwen3_32b \\
        --output-dir v4/runs/qwen3_stage2
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


def _hash_bytes(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acts", type=Path, required=True,
                        help="Path to *_triggers_activations.npz")
    parser.add_argument("--model-tag", required=True,
                        help="Short tag, e.g. 'qwen3_32b' or 'gpt_oss_120b'")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=2000)
    args = parser.parse_args()

    print(f"Loading activations from {args.acts}")
    data = np.load(args.acts)
    y = data["y"]
    print(f"  n={len(y)}, labels {np.bincount(y).tolist()}")

    # Discover layers from npz keys
    layers = sorted(int(k[3:]) for k in data.files if k.startswith("X_L"))
    print(f"  Layers in npz: {layers}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    coefs, intercepts = {}, {}
    per_layer_meta: dict[str, dict] = {}

    print(f"\nTraining LR probes per layer (C={args.C})")
    print(f"{'Layer':>6} {'d_model':>9} {'CV mean':>10} {'CV std':>10}")
    print("-" * 40)
    for layer in layers:
        X = data[f"X_L{layer}"]
        cv_aurocs = []
        for tr, va in skf.split(X, y):
            clf_cv = LogisticRegression(max_iter=args.max_iter, C=args.C)
            clf_cv.fit(X[tr], y[tr])
            s = clf_cv.predict_proba(X[va])[:, 1]
            cv_aurocs.append(roc_auc_score(y[va], s))
        final_clf = LogisticRegression(max_iter=args.max_iter, C=args.C)
        final_clf.fit(X, y)
        coefs[f"coef_L{layer}"] = final_clf.coef_[0].astype(np.float32)
        intercepts[f"intercept_L{layer}"] = float(final_clf.intercept_[0])
        cv_mean = float(np.mean(cv_aurocs))
        cv_std = float(np.std(cv_aurocs))
        print(f"L{layer:>4}  {X.shape[1]:>9}  {cv_mean:>10.3f} {cv_std:>10.3f}")
        per_layer_meta[f"L{layer}"] = {
            "d_model": int(X.shape[1]),
            "cv_mean_auroc": cv_mean,
            "cv_std_auroc": cv_std,
            "coef_sha256": _hash_bytes(coefs[f"coef_L{layer}"]),
            "intercept": intercepts[f"intercept_L{layer}"],
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_npz = args.output_dir / f"frozen_probes_{args.model_tag}.npz"
    out_meta = args.output_dir / f"frozen_probes_{args.model_tag}_meta.json"
    np.savez_compressed(out_npz, **coefs,
                        **{k: np.array(v, dtype=np.float32) for k, v in intercepts.items()})
    with open(out_meta, "w") as f:
        json.dump({
            "model_tag": args.model_tag,
            "training_acts_path": str(args.acts),
            "training_y_sha256": _hash_bytes(y),
            "n_train": int(len(y)),
            "lr_C": float(args.C),
            "lr_max_iter": int(args.max_iter),
            "layers": layers,
            "per_layer": per_layer_meta,
        }, f, indent=2)
    print(f"\nSaved probes: {out_npz}")
    print(f"Saved metadata: {out_meta}")


if __name__ == "__main__":
    main()
