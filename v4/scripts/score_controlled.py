"""
V4b Step E2 (local): score controlled-holdout activations with frozen probes.

Loads:
  - frozen_probes_qwen3.npz       (Step A artifact)
  - controlled_activations.npz    (Step D2 artifact)
  - /tmp/Test_Awareness_Steering/triggers.json (for MS-trained TF-IDF baseline)

For each layer:
  - probe score = sigmoid(coef . X + intercept)
  - AUROC overall (n=80)
  - AUROC per family (A/B/C/D, n=20 each)

Baselines (Microsoft-trained, same training as the probe):
  - Length-only AUROC (expected 0.500 by design)
  - MS-trained TF-IDF AUROC (expected ~0.834 per sanity check)

The decisive question: does probe AUROC > MS-trained TF-IDF AUROC by a
meaningful margin? If yes -> activations capture cross-distribution signal
beyond what text features carry.

For n=80, SE of AUROC at 0.834 is ~0.04, so probe needs to be >= 0.90 to
clearly beat 0.834 (~2 SE). 0.85-0.90 is suggestive. <=0.85 is no evidence.

Saves: v4/runs/qwen3_stage2/controlled_validation_report.json

Usage:
    python v4/scripts/score_controlled.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import yaml

LAYERS = [10, 20, 30, 40, 50, 55, 60, 62, 63]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", type=Path,
                        default=Path("v4/runs/qwen3_stage2/frozen_probes_qwen3.npz"))
    parser.add_argument("--probes-meta", type=Path,
                        default=Path("v4/runs/qwen3_stage2/frozen_probes_qwen3_meta.json"))
    parser.add_argument("--controlled-acts", type=Path,
                        default=Path("v4/runs/qwen3_stage2/controlled_activations.npz"))
    parser.add_argument("--holdout-yaml", type=Path,
                        default=Path("v4/prompts/controlled_holdout_v1.yaml"))
    parser.add_argument("--triggers", type=Path,
                        default=Path("/tmp/Test_Awareness_Steering/triggers.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("v4/runs/qwen3_stage2/controlled_validation_report.json"))
    args = parser.parse_args()

    print(f"Loading frozen probes from {args.probes}")
    probes = np.load(args.probes)
    with open(args.probes_meta) as f:
        probes_meta = json.load(f)

    print(f"Loading controlled activations from {args.controlled_acts}")
    ctrl = np.load(args.controlled_acts)
    y = ctrl["y"]
    lengths = ctrl["lengths"]
    families = np.array([f.decode("utf-8") for f in ctrl["families"]])
    pair_ids = np.array([p.decode("utf-8") for p in ctrl["pair_ids"]])
    print(f"  n={len(y)}, families={sorted(set(families))}, "
          f"by label: {int((y==1).sum())} eval, {int((y==0).sum())} deploy")

    # Reload prompt texts for the TF-IDF baseline (sanity-check already computed this,
    # but we recompute here for self-contained reporting)
    print(f"Loading prompt texts from {args.holdout_yaml}")
    with open(args.holdout_yaml) as f:
        holdout_data = yaml.safe_load(f)
    pair_lookup = {p["pair_id"]: p for p in holdout_data["pairs"]}
    texts = []
    for pid, arm in zip(pair_ids, ctrl["arms"]):
        arm_str = arm.decode("utf-8")
        pair = pair_lookup[pid]
        texts.append(pair[arm_str]["text"])

    # Baseline 1: length-only
    length_only_auroc = float(roc_auc_score(y, -lengths.astype(np.float64)))
    print(f"\nBaseline 1 (length-only): AUROC = {length_only_auroc:.3f}")

    # Baseline 2: Microsoft-trained TF-IDF
    print(f"\nBaseline 2: MS-trained TF-IDF -> controlled")
    with open(args.triggers) as f:
        triggers = json.load(f)
    ms_text: list[str] = []
    ms_y: list[int] = []
    for pair in triggers:
        if "real" in pair:
            ms_text.append(pair["real"])
            ms_y.append(0)
        if "hypothetical" in pair:
            ms_text.append(pair["hypothetical"])
            ms_y.append(1)
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, lowercase=True)
    Xms = vec.fit_transform(ms_text)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xms, np.array(ms_y))
    Xctrl = vec.transform(texts)
    tfidf_scores = clf.predict_proba(Xctrl)[:, 1]
    tfidf_auroc = float(roc_auc_score(y, tfidf_scores))
    print(f"  MS-trained TF-IDF on controlled: AUROC = {tfidf_auroc:.3f}")
    # Per-family TF-IDF
    tfidf_by_family = {}
    for fid in sorted(set(families)):
        mask = families == fid
        try:
            tfidf_by_family[fid] = float(roc_auc_score(y[mask], tfidf_scores[mask]))
        except ValueError:
            tfidf_by_family[fid] = float("nan")
    print(f"  Per-family TF-IDF: " + ", ".join(f"{f}={tfidf_by_family[f]:.3f}" for f in sorted(tfidf_by_family)))

    # Per-layer probe scoring
    print(f"\n{'Layer':>6} {'Overall':>8} {'A':>6} {'B':>6} {'C':>6} {'D':>6} {'CV(MS)':>8} {'vs TFIDF':>10}")
    print("-" * 70)
    results: dict = {
        "layers": LAYERS,
        "baselines": {
            "length_only_auroc": length_only_auroc,
            "ms_tfidf_auroc": tfidf_auroc,
            "ms_tfidf_by_family": tfidf_by_family,
        },
        "per_layer": {},
    }
    for layer in LAYERS:
        coef = probes[f"coef_L{layer}"]
        intercept = float(probes[f"intercept_L{layer}"])
        X = ctrl[f"X_L{layer}"]
        probs = sigmoid(X @ coef + intercept)

        overall_auroc = float(roc_auc_score(y, probs))
        per_family = {}
        for fid in sorted(set(families)):
            mask = families == fid
            try:
                per_family[fid] = float(roc_auc_score(y[mask], probs[mask]))
            except ValueError:
                per_family[fid] = float("nan")
        cv_meta = probes_meta["per_layer"][f"L{layer}"]["cv_mean_auroc"]
        delta = overall_auroc - tfidf_auroc

        print(f"L{layer:>4}  {overall_auroc:>8.3f}  "
              + " ".join(f"{per_family[f]:>5.3f}" for f in sorted(per_family))
              + f"  {cv_meta:>8.3f}  {delta:>+10.3f}")

        results["per_layer"][f"L{layer}"] = {
            "auroc_overall": overall_auroc,
            "auroc_by_family": per_family,
            "cv_ms_meta": cv_meta,
            "delta_vs_ms_tfidf": delta,
        }

    # Peak and verdict
    peak_layer = max(LAYERS, key=lambda l: results["per_layer"][f"L{l}"]["auroc_overall"])
    peak_auroc = results["per_layer"][f"L{peak_layer}"]["auroc_overall"]
    peak_delta = results["per_layer"][f"L{peak_layer}"]["delta_vs_ms_tfidf"]
    print(f"\nPeak probe AUROC on controlled: L{peak_layer} = {peak_auroc:.3f}")
    print(f"MS-trained TF-IDF baseline:     {tfidf_auroc:.3f}")
    print(f"Peak delta (probe - TF-IDF):    {peak_delta:+.3f}")

    # Verdict (with n=80 noise considerations)
    # SE of AUROC at p=0.83 with n=80 is roughly 0.04; 2-SE = 0.08
    if peak_auroc >= 0.95:
        verdict = (f"STRONG: L{peak_layer} probe at {peak_auroc:.3f} clearly beats "
                   f"TF-IDF baseline {tfidf_auroc:.3f} by {peak_delta:+.3f}. "
                   f"Activation-level cross-distribution signal confirmed.")
    elif peak_auroc >= 0.90:
        verdict = (f"MODERATE: L{peak_layer} probe at {peak_auroc:.3f} beats "
                   f"TF-IDF baseline {tfidf_auroc:.3f} by {peak_delta:+.3f}. "
                   f"Probable activation-level signal, but margin not bulletproof at n=80.")
    elif peak_auroc >= tfidf_auroc + 0.03:
        verdict = (f"SUGGESTIVE: L{peak_layer} probe at {peak_auroc:.3f} edges out "
                   f"TF-IDF baseline {tfidf_auroc:.3f} by {peak_delta:+.3f}. "
                   f"At n=80 the margin is within noise; not definitive.")
    elif peak_auroc >= tfidf_auroc - 0.03:
        verdict = (f"NO ADDED VALUE: L{peak_layer} probe at {peak_auroc:.3f} ~= "
                   f"TF-IDF baseline {tfidf_auroc:.3f}. Probe captures the same signal "
                   f"a text classifier captures; no activation-specific contribution.")
    else:
        verdict = (f"WORSE THAN BASELINE: L{peak_layer} probe at {peak_auroc:.3f} < "
                   f"TF-IDF baseline {tfidf_auroc:.3f}. Probe loses information that "
                   f"text features capture. L55 finding does not survive validation.")
    print(f"\nVerdict: {verdict}")

    # L55-specific report
    l55 = results["per_layer"]["L55"]
    print(f"\nL55 specifically:")
    print(f"  Overall:        {l55['auroc_overall']:.3f} (vs TF-IDF {tfidf_auroc:.3f}, delta {l55['delta_vs_ms_tfidf']:+.3f})")
    for fid in sorted(set(families)):
        delta_f = l55['auroc_by_family'][fid] - tfidf_by_family[fid]
        print(f"  Family {fid}:       {l55['auroc_by_family'][fid]:.3f} (vs TF-IDF {tfidf_by_family[fid]:.3f}, delta {delta_f:+.3f})")

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
