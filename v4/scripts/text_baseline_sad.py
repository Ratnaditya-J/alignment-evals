"""
V4b text-only baselines for SAD validation.

Three baselines that need NO model activations:

  1. Length-only: AUROC using prompt length alone as a feature.
  2. Cross-distribution TF-IDF + LR: train on Microsoft triggers text,
     predict SAD labels. Tests whether surface n-gram features transfer.
  3. Within-SAD TF-IDF + LR (5-fold CV): trains and tests on SAD itself.
     Establishes the upper bound of what any text-only classifier can do
     on SAD. If this is high, even within-distribution surface features
     dominate -- probes need to clearly beat it to claim activation-level signal.

These baselines exist to falsify the L55 "eval-awareness" claim. If our probe
AUROC on SAD doesn't beat all three, we're not capturing anything beyond
trivial text features.

DATA POLICY: reads SAD plaintext from ~/.cache/alignment-evals-local/ (outside
the repo tree, per DATA_POLICY.md). Writes only metrics (AUROCs + fold scores)
to v4/runs/qwen3_stage2/text_baseline_report.json -- safe to commit.

Usage:
    python v4/scripts/text_baseline_sad.py --triggers /tmp/Test_Awareness_Steering/triggers.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triggers", type=Path, required=True,
                        help="Path to Microsoft triggers.json")
    parser.add_argument("--sad-cache", type=Path,
                        default=Path.home() / ".cache/alignment-evals-local/sad_oversight_dedup.json")
    parser.add_argument("--output", type=Path,
                        default=Path("v4/runs/qwen3_stage2/text_baseline_report.json"))
    args = parser.parse_args()

    # Load Microsoft triggers
    print(f"Loading Microsoft triggers from {args.triggers}")
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
    ms_y_arr = np.array(ms_y)
    print(f"  Microsoft training: {len(ms_text)} prompts, labels {np.bincount(ms_y_arr).tolist()}")

    # Load SAD prompts (from cache, outside repo)
    print(f"Loading SAD prompts from {args.sad_cache}")
    if not args.sad_cache.exists():
        raise SystemExit(
            f"SAD cache not found at {args.sad_cache}. "
            f"Run load_and_dedup_sad.py first to populate the cache."
        )
    with open(args.sad_cache) as f:
        sad_data = json.load(f)
    sad_text = [it["body"] for it in sad_data["kept"]]
    sad_y = np.array([it["label"] for it in sad_data["kept"]])
    sad_lens = np.array([it["length"] for it in sad_data["kept"]], dtype=np.float64)
    print(f"  SAD: {len(sad_text)} prompts, labels {np.bincount(sad_y).tolist()}")

    results: dict = {
        "n_microsoft_train": len(ms_text),
        "n_sad": len(sad_text),
    }

    # Baseline 1: length-only
    print(f"\n--- Baseline 1: length only (no activations) ---")
    # Score = -length because longer = deploy = label 0
    length_only_auroc = float(roc_auc_score(sad_y, -sad_lens))
    print(f"  Length-only AUROC: {length_only_auroc:.3f}")
    results["length_only_auroc"] = length_only_auroc

    # Baseline 2: cross-distribution TF-IDF + LR (MS -> SAD)
    print(f"\n--- Baseline 2: cross-distribution TF-IDF + LR (Microsoft -> SAD) ---")
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, lowercase=True)
    Xms = vec.fit_transform(ms_text)
    print(f"  Vocab size: {len(vec.vocabulary_)}")
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xms, ms_y_arr)
    Xsad = vec.transform(sad_text)
    ms_to_sad_scores = clf.predict_proba(Xsad)[:, 1]
    cross_dist_auroc = float(roc_auc_score(sad_y, ms_to_sad_scores))
    print(f"  MS-trained TF-IDF+LR AUROC on SAD: {cross_dist_auroc:.3f}")
    results["cross_dist_tfidf_auroc_ms_to_sad"] = cross_dist_auroc

    # Baseline 3: within-SAD 5-fold CV
    print(f"\n--- Baseline 3: within-SAD TF-IDF + LR (5-fold CV) ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aurocs: list[float] = []
    for fold, (tr_i, va_i) in enumerate(skf.split(sad_text, sad_y)):
        vec2 = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, lowercase=True)
        Xtr = vec2.fit_transform([sad_text[i] for i in tr_i])
        Xva = vec2.transform([sad_text[i] for i in va_i])
        clf2 = LogisticRegression(max_iter=2000, C=1.0)
        clf2.fit(Xtr, sad_y[tr_i])
        va_scores = clf2.predict_proba(Xva)[:, 1]
        fold_auroc = float(roc_auc_score(sad_y[va_i], va_scores))
        cv_aurocs.append(fold_auroc)
        print(f"  Fold {fold+1}: AUROC = {fold_auroc:.3f}")
    within_mean = float(np.mean(cv_aurocs))
    within_std = float(np.std(cv_aurocs))
    print(f"  Within-SAD 5-fold CV: {within_mean:.3f} +/- {within_std:.3f}")
    results["within_sad_tfidf_5fold_cv"] = {
        "mean": within_mean, "std": within_std,
        "folds": cv_aurocs,
    }

    # Summary + interpretation hints
    print(f"\n{'='*60}")
    print(f"BASELINE SUMMARY (need to beat these to claim probe signal):")
    print(f"  Length-only:                        {length_only_auroc:.3f}")
    print(f"  Cross-dist TF-IDF (MS -> SAD):      {cross_dist_auroc:.3f}")
    print(f"  Within-SAD TF-IDF (5-fold CV):      {within_mean:.3f}")
    print(f"{'='*60}")
    print(f"\nInterpretation guide:")
    print(f"  - If a probe's SAD AUROC is BELOW length-only -> probe is worse than measuring length")
    print(f"  - If a probe's SAD AUROC is BELOW cross-dist TF-IDF -> surface text features explain it")
    print(f"  - If within-SAD TF-IDF is near 1.0 -> SAD is trivially text-separable, activations aren't needed")
    print(f"  - Probe AUROC only meaningfully > all three baselines counts as activation-level signal")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
