"""Refusal-scorer validation workflow.

This is the human-labeling companion to ``rescore_cross_protocol.py``. Use it
to produce a stratified sample of rollouts, label them by hand, and then
compute the agreement / precision / recall / F1 of both the regex scorer and
(optionally) the LLM judge against the human gold labels.

A SOTA-grade benchmark needs to ground its refusal rates in something other
than "the regex matched". Once you have human labels for ~100 records, the
``--score`` mode emits a clear validation table you can drop straight into
the report:

    | Sample size | Agreement | Precision | Recall | F1 |

It also surfaces disagreement so you can iterate on the scorer.

Two modes:

1. **Sample** (default). Reads ``rollouts.jsonl`` (or the enriched
   ``rollouts_scored.jsonl`` produced by ``rescore_cross_protocol.py``),
   stratifies across (model, condition, regex_predicted_refusal) so every
   bucket gets some records, and writes a CSV ready for human labeling.

       python scripts/validate_refusal_scorer.py runs/cross-protocol-v3/ \\
           --n-per-bucket 5 --output labelling.csv

2. **Score**. Reads a filled CSV and emits agreement / precision / recall /
   F1 vs human gold labels. ``human_refusal`` must be ``yes`` / ``no`` /
   ``unsure``; ``unsure`` rows are excluded from the metrics so partial
   labelling still works.

       python scripts/validate_refusal_scorer.py runs/cross-protocol-v3/ \\
           --score --input labelling.csv

The CSV columns are:
    response_id,model_name,condition,task_family,example_id,rollout_index,
    prompt,response,refusal_regex,refusal_llm,llm_refusal_type,
    human_refusal,human_notes
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval_awareness import (  # noqa: E402
    RefusalScorer,
    ScoringContext,
    classify_model,
    task_family_from_example_id,
)

LOGGER = logging.getLogger("validate_refusal_scorer")

CSV_FIELDS = (
    "response_id",
    "model_name",
    "condition",
    "task_family",
    "example_id",
    "rollout_index",
    "prompt",
    "response",
    "refusal_regex",
    "refusal_llm",
    "llm_refusal_type",
    "human_refusal",
    "human_notes",
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _pick_rollouts_path(run_dir: Path) -> Path:
    """Prefer the enriched rollouts_scored.jsonl when present."""
    scored = run_dir / "rollouts_scored.jsonl"
    if scored.exists():
        return scored
    return run_dir / "rollouts.jsonl"


def _load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(
            f"missing {path}; run rescore_cross_protocol.py first or pass an "
            "explicit --rollouts path"
        )
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _ensure_regex_score(record: Dict[str, Any]) -> None:
    if "refusal_regex" not in record and "refusal" in record:
        record["refusal_regex"] = float(record["refusal"])
    if "refusal_regex" not in record:
        scorer = RefusalScorer()
        score = scorer.score(
            record.get("response", "") or "", ScoringContext(task_id="validate")
        )
        record["refusal_regex"] = score.score


def _ensure_task_family(record: Dict[str, Any]) -> None:
    if record.get("task_family") in (None, "", "unknown"):
        record["task_family"] = task_family_from_example_id(
            str(record.get("example_id", ""))
        )


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------


def _stratified_sample(
    records: Sequence[Dict[str, Any]],
    n_per_bucket: int,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Sample ``n_per_bucket`` records from each (model, condition, regex+/−).

    This makes sure every cell of the report appears in the human label set,
    so the resulting metrics aren't dominated by the largest cell.
    """
    rng = random.Random(seed)
    buckets: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        regex = float(record.get("refusal_regex", 0.0))
        key = (
            str(record.get("model_name", "?")),
            str(record.get("condition", "?")),
            "regex+" if regex >= 0.5 else "regex-",
        )
        buckets[key].append(record)

    sample: List[Dict[str, Any]] = []
    for key in sorted(buckets):
        items = buckets[key]
        rng.shuffle(items)
        sample.extend(items[:n_per_bucket])
    return sample


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------


def _format_for_csv(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)


def _write_labelling_csv(records: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for index, record in enumerate(records):
            row = {
                "response_id": f"row_{index:05d}",
                "model_name": _format_for_csv(record.get("model_name")),
                "condition": _format_for_csv(record.get("condition")),
                "task_family": _format_for_csv(record.get("task_family")),
                "example_id": _format_for_csv(record.get("example_id")),
                "rollout_index": _format_for_csv(record.get("rollout_index")),
                "prompt": _format_for_csv(record.get("prompt")),
                "response": _format_for_csv(record.get("response")),
                "refusal_regex": _format_for_csv(record.get("refusal_regex")),
                "refusal_llm": _format_for_csv(record.get("refusal_llm")),
                "llm_refusal_type": _format_for_csv(record.get("llm_refusal_type")),
                "human_refusal": "",
                "human_notes": "",
            }
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Scoring (mode 2): compare human labels against regex / LLM
# ---------------------------------------------------------------------------


def _coerce_label(value: str) -> Optional[bool]:
    """Map human-typed labels to bool. Returns None for unlabeled / unsure."""
    norm = (value or "").strip().lower()
    if norm in {"y", "yes", "true", "1", "refuse", "refused", "refusal"}:
        return True
    if norm in {"n", "no", "false", "0", "comply", "complied", "answer"}:
        return False
    return None


def _coerce_score(value: str) -> Optional[bool]:
    if value is None or value == "":
        return None
    try:
        return float(value) >= 0.5
    except (TypeError, ValueError):
        return None


def _metrics(pairs: Sequence[Tuple[bool, bool]]) -> Dict[str, Any]:
    tp = sum(1 for predicted, gold in pairs if predicted and gold)
    fp = sum(1 for predicted, gold in pairs if predicted and not gold)
    fn = sum(1 for predicted, gold in pairs if not predicted and gold)
    tn = sum(1 for predicted, gold in pairs if not predicted and not gold)
    total = tp + fp + fn + tn
    agreement = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )
    return {
        "n": total,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "agreement": round(agreement, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _score_labelling_csv(path: Path) -> Dict[str, Any]:
    """Read a filled labeling CSV and emit metrics for regex + LLM vs human."""
    if not path.exists():
        raise SystemExit(f"missing labeling CSV: {path}")
    regex_pairs: List[Tuple[bool, bool]] = []
    llm_pairs: List[Tuple[bool, bool]] = []
    labeled = 0
    unsure = 0
    per_model_regex: Dict[str, List[Tuple[bool, bool]]] = defaultdict(list)
    per_model_llm: Dict[str, List[Tuple[bool, bool]]] = defaultdict(list)
    disagreements: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            human_raw = row.get("human_refusal", "") or ""
            gold = _coerce_label(human_raw)
            if gold is None:
                if human_raw:
                    unsure += 1
                continue
            labeled += 1
            model = row.get("model_name", "?")
            regex_pred = _coerce_score(row.get("refusal_regex", ""))
            if regex_pred is not None:
                regex_pairs.append((regex_pred, gold))
                per_model_regex[model].append((regex_pred, gold))
                if regex_pred != gold:
                    disagreements.append(
                        {
                            "response_id": row.get("response_id"),
                            "model_name": model,
                            "condition": row.get("condition"),
                            "regex_predicted_refusal": regex_pred,
                            "human_refusal": gold,
                            "notes": row.get("human_notes", ""),
                        }
                    )
            llm_pred = _coerce_score(row.get("refusal_llm", ""))
            if llm_pred is not None:
                llm_pairs.append((llm_pred, gold))
                per_model_llm[model].append((llm_pred, gold))
    return {
        "labeled": labeled,
        "unlabeled_or_unsure": unsure,
        "regex_vs_human": _metrics(regex_pairs),
        "llm_vs_human": _metrics(llm_pairs) if llm_pairs else None,
        "regex_vs_human_per_model": {
            model: _metrics(pairs) for model, pairs in sorted(per_model_regex.items())
        },
        "llm_vs_human_per_model": {
            model: _metrics(pairs) for model, pairs in sorted(per_model_llm.items())
        }
        if llm_pairs
        else None,
        "scorer_disagreements": disagreements[:50],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Iterable[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Run directory containing rollouts(.jsonl)")
    parser.add_argument(
        "--rollouts",
        default=None,
        help=(
            "Explicit path to the rollouts file. Default picks "
            "rollouts_scored.jsonl when present, else rollouts.jsonl."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the labeling CSV (sample mode). Default: <run_dir>/labelling.csv",
    )
    parser.add_argument(
        "--n-per-bucket",
        type=int,
        default=5,
        help="Records per (model, condition, regex+/−) bucket. Default 5.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Sampling seed.",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="Score-mode: read a filled labeling CSV and emit metrics.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to the filled labeling CSV when --score is used.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    run_dir = Path(args.run_dir)

    if args.score:
        csv_path = Path(args.input) if args.input else run_dir / "labelling.csv"
        LOGGER.info("scoring labels from %s", csv_path)
        result = _score_labelling_csv(csv_path)
        out_path = csv_path.with_suffix(".metrics.json")
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        LOGGER.info("wrote %s", out_path)
        regex = result["regex_vs_human"]
        llm = result.get("llm_vs_human")
        print()
        print(
            f"Labeled records: {result['labeled']} "
            f"(skipped {result['unlabeled_or_unsure']} unlabeled/unsure)"
        )
        print(
            f"Regex vs human: agreement={regex['agreement']:.3f}, "
            f"precision={regex['precision']:.3f}, recall={regex['recall']:.3f}, "
            f"F1={regex['f1']:.3f}"
        )
        if llm:
            print(
                f"LLM judge vs human: agreement={llm['agreement']:.3f}, "
                f"precision={llm['precision']:.3f}, recall={llm['recall']:.3f}, "
                f"F1={llm['f1']:.3f}"
            )
        print(f"Detailed metrics at: {out_path}")
        return

    rollouts_path = Path(args.rollouts) if args.rollouts else _pick_rollouts_path(run_dir)
    LOGGER.info("loading rollouts from %s", rollouts_path)
    records = _load_records(rollouts_path)
    LOGGER.info("loaded %d records", len(records))

    for record in records:
        _ensure_regex_score(record)
        _ensure_task_family(record)

    sample = _stratified_sample(
        records, n_per_bucket=args.n_per_bucket, seed=args.seed
    )
    LOGGER.info(
        "stratified sample: %d records across %d buckets",
        len(sample),
        # quick estimate
        max(1, len(sample) // max(1, args.n_per_bucket)),
    )

    out_path = Path(args.output) if args.output else run_dir / "labelling.csv"
    _write_labelling_csv(sample, out_path)
    LOGGER.info("wrote %s", out_path)

    # Print a per-model breakdown so the user knows what they're labeling.
    by_model: Dict[str, int] = defaultdict(int)
    for record in sample:
        by_model[str(record.get("model_name", "?"))] += 1
    print()
    print(f"Wrote labeling CSV with {len(sample)} rows to: {out_path}")
    print("Per-model sample counts:")
    for model, count in sorted(by_model.items()):
        tier = classify_model(model)
        print(
            f"  - {model} ({tier.vendor}, "
            f"{'reasoning' if tier.is_reasoning else 'non-reasoning'}): {count}"
        )
    print()
    print("To label: open the CSV, fill in human_refusal (yes/no/unsure) and "
          "human_notes columns, then run:")
    print(f"  python scripts/validate_refusal_scorer.py {run_dir} --score "
          f"--input {out_path}")


if __name__ == "__main__":
    main()
