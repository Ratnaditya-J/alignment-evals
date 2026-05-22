"""
V4b sanity check on controlled_holdout_v1: confirm dataset isn't trivially
text-separable BEFORE paying for GPU activation extraction.

Computes (no GPU, no model):
  1. Length-only AUROC -- should be near 0.5 by construction
  2. Within-set TF-IDF 5-fold CV AUROC -- should be moderate (NOT > 0.95)
  3. Per-family length-only AUROC -- per-family sanity

Decision:
  - Length-only < 0.55 AND TF-IDF CV < 0.85 -> dataset is good, proceed to GPU
  - Length-only >= 0.55 -> redesign framings (length confound creeping in)
  - TF-IDF CV >= 0.95 -> dataset is text-separable, probes won't add anything
    (need to make framings more semantically minimal or harder to distinguish)

Usage:
    python v4/scripts/sanity_check_controlled.py
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("v4/prompts/controlled_holdout_v1.yaml"))
    parser.add_argument("--output", type=Path,
                        default=Path("v4/runs/qwen3_stage2/controlled_sanity_report.json"))
    args = parser.parse_args()

    with open(args.input) as f:
        data = yaml.safe_load(f)
    pairs = data["pairs"]

    texts, labels, lengths, families, qids = [], [], [], [], []
    for p in pairs:
        for arm in ("a", "b"):
            texts.append(p[arm]["text"])
            labels.append(1 if p[arm]["label"] == "eval" else 0)
            lengths.append(len(p[arm]["text"]))
            families.append(p["family"])
            qids.append(p["question_id"])

    y = np.array(labels)
    lens = np.array(lengths, dtype=np.float64)
    fams = np.array(families)
    print(f"Loaded {len(texts)} prompts from {len(pairs)} pairs")
    print(f"  labels: {np.bincount(y).tolist()}")
    print(f"  length stats: min={int(lens.min())}, max={int(lens.max())}, "
          f"mean={float(lens.mean()):.1f}, std={float(lens.std()):.1f}")

    # 1. Length-only AUROC
    length_auroc = float(roc_auc_score(y, -lens))
    print(f"\n=== 1. Length-only AUROC (should be near 0.5 by design) ===")
    print(f"  Overall: {length_auroc:.3f}")
    for fid in sorted(set(families)):
        mask = fams == fid
        fam_auroc = float(roc_auc_score(y[mask], -lens[mask]))
        print(f"  Family {fid}: {fam_auroc:.3f} (n={int(mask.sum())})")

    # 2. Within-set TF-IDF 5-fold CV (memorization check, expected to be high)
    print(f"\n=== 2. Within-set TF-IDF + LR 5-fold CV (in-distribution, expected high) ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aurocs: list[float] = []
    for fold, (tr, va) in enumerate(skf.split(texts, y)):
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, lowercase=True)
        Xtr = vec.fit_transform([texts[i] for i in tr])
        Xva = vec.transform([texts[i] for i in va])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, y[tr])
        va_scores = clf.predict_proba(Xva)[:, 1]
        fold_auroc = float(roc_auc_score(y[va], va_scores))
        cv_aurocs.append(fold_auroc)
        print(f"  Fold {fold+1}: {fold_auroc:.3f}")
    tfidf_mean = float(np.mean(cv_aurocs))
    tfidf_std = float(np.std(cv_aurocs))
    print(f"  Mean: {tfidf_mean:.3f} +/- {tfidf_std:.3f}")
    print(f"  (high value expected -- this only tests memorizability within seen framings)")

    # 3. Leave-one-family-out TF-IDF (within-controlled, cross-family generalization)
    print(f"\n=== 3. Leave-one-family-out TF-IDF + LR (within-controlled, cross-family) ===")
    unique_families = sorted(set(families))
    lofo_aurocs: dict[str, float] = {}
    for held_out in unique_families:
        train_idx = [i for i, f in enumerate(families) if f != held_out]
        test_idx = [i for i, f in enumerate(families) if f == held_out]
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, lowercase=True)
        Xtr = vec.fit_transform([texts[i] for i in train_idx])
        Xte = vec.transform([texts[i] for i in test_idx])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, y[train_idx])
        te_scores = clf.predict_proba(Xte)[:, 1]
        try:
            auroc = float(roc_auc_score(y[test_idx], te_scores))
        except ValueError:
            auroc = float("nan")
        lofo_aurocs[held_out] = auroc
        print(f"  Hold out Family {held_out}: AUROC={auroc:.3f}")
    lofo_mean = float(np.nanmean(list(lofo_aurocs.values())))
    lofo_std = float(np.nanstd(list(lofo_aurocs.values())))
    print(f"  Mean LOFO TF-IDF AUROC: {lofo_mean:.3f} +/- {lofo_std:.3f}")
    print(f"  (NOT a gate -- eval/deploy distinction is inherently lexical; this is informational)")

    # 4. Microsoft-trained TF-IDF baseline (the actually-meaningful baseline)
    triggers_path = Path("/tmp/Test_Awareness_Steering/triggers.json")
    ms_to_ctrl_auroc: float | None = None
    if triggers_path.exists():
        print(f"\n=== 4. Microsoft-trained TF-IDF on controlled (the meaningful baseline) ===")
        print(f"  Loading Microsoft triggers from {triggers_path}")
        with open(triggers_path) as f:
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
        print(f"  Microsoft training: {len(ms_text)} prompts")
        vec_ms = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, lowercase=True)
        Xms = vec_ms.fit_transform(ms_text)
        clf_ms = LogisticRegression(max_iter=2000, C=1.0)
        clf_ms.fit(Xms, np.array(ms_y))
        Xctrl = vec_ms.transform(texts)
        ctrl_scores = clf_ms.predict_proba(Xctrl)[:, 1]
        ms_to_ctrl_auroc = float(roc_auc_score(y, ctrl_scores))
        print(f"  MS-trained TF-IDF -> controlled AUROC: {ms_to_ctrl_auroc:.3f}")
        print(f"  ^ This is the baseline the L55 probe (also MS-trained) needs to beat.")
    else:
        print(f"\n=== 4. Microsoft-trained TF-IDF baseline ===")
        print(f"  Skipped -- triggers.json not at {triggers_path}")

    # 5. Decision
    print(f"\n{'='*60}")
    print(f"DECISION:")
    length_ok = length_auroc < 0.55
    print(f"  Length-only AUROC {length_auroc:.3f} < 0.55: {'PASS' if length_ok else 'FAIL'}")
    if ms_to_ctrl_auroc is not None:
        print(f"  MS-trained TF-IDF -> controlled AUROC: {ms_to_ctrl_auroc:.3f}")
        print(f"    (probe must beat {ms_to_ctrl_auroc:.3f} on controlled to claim activation signal)")
    print(f"  Within-controlled LOFO TF-IDF: {lofo_mean:.3f} (informational, not a gate)")
    print(f"  Within-controlled 5-fold TF-IDF: {tfidf_mean:.3f} (informational, not a gate)")

    if length_ok:
        baseline_str = f"{ms_to_ctrl_auroc:.3f}" if ms_to_ctrl_auroc is not None else "TBD"
        verdict = (f"PROCEED with GPU extraction. The dataset has perfectly controlled length. "
                   f"The decisive comparison will be: probe-on-controlled AUROC vs "
                   f"MS-trained-TFIDF-on-controlled AUROC ({baseline_str}). "
                   f"If probe > TF-IDF by meaningful margin, activation captures "
                   f"cross-distribution signal beyond surface text.")
    else:
        verdict = "REDESIGN: length confound still present, framings need rebalancing"
    print(f"\n  Verdict: {verdict}")
    print(f"{'='*60}")

    report = {
        "n_prompts": len(texts),
        "n_pairs": len(pairs),
        "length_only_auroc_overall": length_auroc,
        "length_only_auroc_by_family": {
            fid: float(roc_auc_score(y[fams == fid], -lens[fams == fid]))
            for fid in sorted(set(families))
        },
        "tfidf_5fold_cv": {
            "mean": tfidf_mean, "std": tfidf_std, "folds": cv_aurocs,
        },
        "tfidf_leave_one_family_out": {
            "per_family": lofo_aurocs,
            "mean": lofo_mean, "std": lofo_std,
        },
        "ms_trained_tfidf_on_controlled_auroc": ms_to_ctrl_auroc,
        "length_only_passes_threshold": length_ok,
        "verdict": verdict,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
