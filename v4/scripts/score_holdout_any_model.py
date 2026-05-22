"""
Score holdout activations against frozen per-layer probes, with text baselines.

Generic version: works for ANY model's (frozen probes, holdout activations)
pair produced by train_probes_any_model.py + extract_acts_any_model_pod.py.

Reports per layer:
  - Probe AUROC overall
  - Probe AUROC per family (if controlled holdout)
  - Probe AUROC vs MS-trained TF-IDF baseline
  - Delta probe - TF-IDF

Verdict logic based on delta vs MS-trained TF-IDF baseline (same as Qwen3 V4b).

Usage:
    python score_holdout_any_model.py \\
        --probes v4/runs/qwen3_stage2/frozen_probes_qwen3_32b.npz \\
        --probes-meta v4/runs/qwen3_stage2/frozen_probes_qwen3_32b_meta.json \\
        --holdout-acts v4/runs/qwen3_stage2/qwen3_32b_controlled_activations.npz \\
        --holdout-yaml v4/prompts/controlled_holdout_v2.yaml \\
        --triggers /tmp/Test_Awareness_Steering/triggers.json \\
        --model-tag qwen3_32b \\
        --output v4/runs/qwen3_stage2/validation_qwen3_32b.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--probes-meta", type=Path, required=True)
    parser.add_argument("--holdout-acts", type=Path, required=True)
    parser.add_argument("--holdout-yaml", type=Path, required=True)
    parser.add_argument("--triggers", type=Path, required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    print(f"Loading frozen probes from {args.probes}")
    probes = np.load(args.probes)
    with open(args.probes_meta) as f:
        probes_meta = json.load(f)
    layers = probes_meta["layers"]
    print(f"  layers: {layers}")

    print(f"Loading holdout activations from {args.holdout_acts}")
    holdout = np.load(args.holdout_acts)
    y = holdout["y"]
    lengths = holdout["lengths"]
    families = np.array([f.decode("utf-8") for f in holdout["families"]])
    pair_ids = np.array([p.decode("utf-8") for p in holdout["pair_ids"]])
    arms = np.array([a.decode("utf-8") for a in holdout["arms"]])
    print(f"  n={len(y)}, families={sorted(set(families))}, "
          f"by label: {int((y==1).sum())} eval, {int((y==0).sum())} deploy")

    # Reload prompt texts from YAML for TF-IDF baseline
    with open(args.holdout_yaml) as f:
        holdout_data = yaml.safe_load(f)
    pair_lookup = {p["pair_id"]: p for p in holdout_data["pairs"]}
    texts = [pair_lookup[pid][arm]["text"] for pid, arm in zip(pair_ids, arms)]

    # Baseline 1: length-only
    length_only_auroc = float(roc_auc_score(y, -lengths.astype(np.float64)))
    print(f"\nBaseline 1 (length-only): AUROC = {length_only_auroc:.3f}")

    # Baseline 2: MS-trained TF-IDF
    with open(args.triggers) as f:
        triggers = json.load(f)
    ms_text, ms_y = [], []
    for pair in triggers:
        if "real" in pair:
            ms_text.append(pair["real"]); ms_y.append(0)
        if "hypothetical" in pair:
            ms_text.append(pair["hypothetical"]); ms_y.append(1)
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, lowercase=True)
    Xms = vec.fit_transform(ms_text)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xms, np.array(ms_y))
    tfidf_scores = clf.predict_proba(vec.transform(texts))[:, 1]
    tfidf_auroc = float(roc_auc_score(y, tfidf_scores))
    print(f"Baseline 2 (MS-trained TF-IDF -> holdout): AUROC = {tfidf_auroc:.3f}")
    tfidf_by_family = {
        fid: float(roc_auc_score(y[families == fid], tfidf_scores[families == fid]))
        for fid in sorted(set(families))
    }

    # Per-layer probe scoring
    print(f"\n{'Layer':>6} {'Overall':>8} " + " ".join(f"{f:>6}" for f in sorted(set(families)))
          + f" {'CV(train)':>11} {'vs TFIDF':>10}")
    print("-" * 80)
    results: dict = {
        "model_tag": args.model_tag,
        "layers": layers,
        "baselines": {
            "length_only_auroc": length_only_auroc,
            "ms_tfidf_auroc": tfidf_auroc,
            "ms_tfidf_by_family": tfidf_by_family,
        },
        "per_layer": {},
    }
    for layer in layers:
        coef = probes[f"coef_L{layer}"]
        intercept = float(probes[f"intercept_L{layer}"])
        X = holdout[f"X_L{layer}"]
        probs = sigmoid(X @ coef + intercept)
        overall = float(roc_auc_score(y, probs))
        per_fam = {
            fid: float(roc_auc_score(y[families == fid], probs[families == fid]))
            for fid in sorted(set(families))
        }
        cv = probes_meta["per_layer"][f"L{layer}"]["cv_mean_auroc"]
        delta = overall - tfidf_auroc
        print(f"L{layer:>4}  {overall:>8.3f}  " +
              " ".join(f"{per_fam[f]:>5.3f}" for f in sorted(per_fam)) +
              f"  {cv:>11.3f} {delta:>+10.3f}")
        results["per_layer"][f"L{layer}"] = {
            "auroc_overall": overall,
            "auroc_by_family": per_fam,
            "cv_meta": cv,
            "delta_vs_ms_tfidf": delta,
        }

    peak_layer = max(layers, key=lambda l: results["per_layer"][f"L{l}"]["auroc_overall"])
    peak_auroc = results["per_layer"][f"L{peak_layer}"]["auroc_overall"]
    peak_delta = results["per_layer"][f"L{peak_layer}"]["delta_vs_ms_tfidf"]
    print(f"\nPeak probe AUROC: L{peak_layer} = {peak_auroc:.3f}")
    print(f"MS-trained TF-IDF baseline: {tfidf_auroc:.3f}")
    print(f"Peak delta: {peak_delta:+.3f}")

    n = len(y)
    if peak_auroc >= 0.95:
        verdict = (f"STRONG: L{peak_layer} probe at {peak_auroc:.3f} clearly beats "
                   f"TF-IDF baseline {tfidf_auroc:.3f} by {peak_delta:+.3f} (n={n}). "
                   f"Activation-level cross-distribution signal confirmed.")
    elif peak_auroc >= 0.90:
        verdict = (f"MODERATE: L{peak_layer} probe at {peak_auroc:.3f} beats "
                   f"TF-IDF baseline {tfidf_auroc:.3f} by {peak_delta:+.3f} (n={n}). "
                   f"Probable activation-level signal.")
    elif peak_auroc >= tfidf_auroc + 0.03:
        verdict = (f"SUGGESTIVE: L{peak_layer} probe at {peak_auroc:.3f} edges out "
                   f"TF-IDF baseline {tfidf_auroc:.3f} by {peak_delta:+.3f} (n={n}). "
                   f"Within noise; not definitive.")
    elif peak_auroc >= tfidf_auroc - 0.03:
        verdict = (f"NO ADDED VALUE: L{peak_layer} probe at {peak_auroc:.3f} ~= "
                   f"TF-IDF baseline {tfidf_auroc:.3f}. Probe captures same signal "
                   f"as text classifier.")
    else:
        verdict = (f"WORSE THAN BASELINE: L{peak_layer} probe at {peak_auroc:.3f} < "
                   f"TF-IDF baseline {tfidf_auroc:.3f}. No activation signal.")
    print(f"\nVerdict: {verdict}")

    results["peak_layer"] = peak_layer
    results["peak_auroc"] = peak_auroc
    results["peak_delta_vs_tfidf"] = peak_delta
    results["verdict"] = verdict

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
