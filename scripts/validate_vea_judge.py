"""VEA-judge validation workflow.

The human-labeling companion to ``run_goodfire_vea.py``. Use it to
produce a stratified sample of VEA classifications, label them by hand,
and then compute precision / recall / F1 of the LLM VEA judge against
human gold labels - both overall AND per ``vea_type`` subtype.

A SOTA-grade benchmark needs the VEA judge grounded in something other
than "another LLM said so." Once you have ~200-500 human labels, the
``--score`` mode emits a validation table you can drop straight into
the methods section:

    | Subtype | n | precision | recall | F1 |

Two modes:

1. **Sample** (default). Reads ``goodfire_vea_classifications.jsonl`` +
   ``rollouts.jsonl`` from a Goodfire run dir, joins them, stratifies
   so every ``vea_type`` subtype is represented (oversamples positives
   across strict + soft subtypes; samples some negatives too), and
   writes a CSV ready for human labeling.

       python scripts/validate_vea_judge.py runs/goodfire-mixed-n500/ \\
           --n-per-subtype 30 --output vea_labels.csv

2. **Score**. Reads a filled CSV and emits judge-vs-human metrics
   overall, per-subtype, and split by strict / broad rubric.
   ``human_is_vea`` must be ``yes`` / ``no`` / ``unsure``; ``unsure``
   rows are excluded from the metrics so partial labelling still works.

       python scripts/validate_vea_judge.py runs/goodfire-mixed-n500/ \\
           --score --input vea_labels.csv

CSV columns:
    label_id, model_name, condition, task_family, example_id,
    rollout_index, prompt_snippet, response_snippet,
    reasoning_trace_snippet, judge_is_vea, judge_vea_type,
    judge_confidence, judge_quote, human_is_vea, human_notes
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

LOGGER = logging.getLogger("validate_vea_judge")

CSV_FIELDS = (
    "label_id",
    "model_name",
    "condition",
    "task_family",
    "example_id",
    "rollout_index",
    "prompt_snippet",
    "response_snippet",
    "reasoning_trace_snippet",
    "judge_is_vea",
    "judge_vea_type",
    "judge_confidence",
    "judge_quote",
    "human_is_vea",
    "human_notes",
)

# Subtypes that count as VEA+ under strict eval-recognition rubrics.
# Mirrors scripts/analyze_vea_mediation.py:_STRICT_VEA_SUBTYPES.
_STRICT_SUBTYPES: frozenset = frozenset(
    {"explicit_eval", "benchmark", "grader_awareness"}
)
_SOFT_SUBTYPES: frozenset = frozenset(
    {"safety_check", "test_scenario", "artificial_scenario"}
)
_NEGATIVE_SUBTYPES: frozenset = frozenset({"none", "no_trace"})

# Snippet lengths to keep the CSV readable in a spreadsheet.
_PROMPT_SNIPPET_CHARS = 240
_RESPONSE_SNIPPET_CHARS = 320
_TRACE_SNIPPET_CHARS = 1200


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    out: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _join_rollouts_with_classifications(
    rollouts: Sequence[Dict[str, Any]],
    classifications: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Inner join on (model_name, condition, example_id, rollout_index).

    Returns one record per rollout that has a matching classification, with
    judge fields surfaced and the original rollout fields preserved.
    """
    key = lambda r: (
        str(r.get("model_name", "")),
        str(r.get("condition", "")),
        str(r.get("example_id", "")),
        int(r.get("rollout_index", 0) or 0),
    )
    cls_by_key = {key(c): c for c in classifications}
    joined: List[Dict[str, Any]] = []
    misses = 0
    for r in rollouts:
        c = cls_by_key.get(key(r))
        if c is None:
            misses += 1
            continue
        merged = dict(r)
        merged["judge_is_vea"] = bool(c.get("is_vea"))
        merged["judge_vea_type"] = c.get("vea_type")
        merged["judge_confidence"] = c.get("confidence")
        merged["judge_quote"] = c.get("quoted_text") or c.get("quote") or ""
        joined.append(merged)
    if misses:
        LOGGER.warning(
            "%d rollouts had no matching VEA classification (skipped)", misses
        )
    return joined


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------


def _stratified_sample(
    records: Sequence[Dict[str, Any]],
    n_per_subtype: int,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Stratify by judge ``vea_type`` so every subtype gets ``n_per_subtype``
    records (or all records, whichever is smaller).

    Strict subtypes (explicit_eval / benchmark / grader_awareness) and soft
    subtypes (safety_check / test_scenario / artificial_scenario) and the
    negative bucket (none / no_trace) are all represented. Without this
    stratification the human-labeling sample would be dominated by ``none``
    (the modal class) and the labels wouldn't constrain the judge's
    precision on the rare-but-load-bearing positive subtypes.
    """
    rng = random.Random(seed)
    by_subtype: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        subtype = str(record.get("judge_vea_type") or "unknown").strip().lower()
        by_subtype[subtype].append(record)

    sample: List[Dict[str, Any]] = []
    for subtype in sorted(by_subtype):
        items = by_subtype[subtype]
        rng.shuffle(items)
        sample.extend(items[:n_per_subtype])
    return sample


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------


def _truncate(text: Any, n: int) -> str:
    if text is None:
        return ""
    s = str(text)
    if len(s) <= n:
        return s
    return s[:n] + "..."


def _write_labelling_csv(
    records: Sequence[Dict[str, Any]], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_ALL
        )
        writer.writeheader()
        for index, record in enumerate(records):
            row = {
                "label_id": f"vea_{index:05d}",
                "model_name": str(record.get("model_name") or ""),
                "condition": str(record.get("condition") or ""),
                "task_family": str(record.get("task_family") or ""),
                "example_id": str(record.get("example_id") or ""),
                "rollout_index": str(record.get("rollout_index") or 0),
                "prompt_snippet": _truncate(
                    record.get("prompt"), _PROMPT_SNIPPET_CHARS
                ),
                "response_snippet": _truncate(
                    record.get("response"), _RESPONSE_SNIPPET_CHARS
                ),
                "reasoning_trace_snippet": _truncate(
                    record.get("reasoning_trace"), _TRACE_SNIPPET_CHARS
                ),
                "judge_is_vea": "true" if record.get("judge_is_vea") else "false",
                "judge_vea_type": str(record.get("judge_vea_type") or ""),
                "judge_confidence": str(record.get("judge_confidence") or ""),
                "judge_quote": _truncate(record.get("judge_quote"), 240),
                "human_is_vea": "",
                "human_notes": "",
            }
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Scoring (mode 2): judge vs human
# ---------------------------------------------------------------------------


def _coerce_human_label(value: str) -> Optional[bool]:
    """Map human-typed labels to bool. Returns None for unlabeled / unsure.

    Accepted positive: y/yes/true/1/vea/vea+/aware
    Accepted negative: n/no/false/0/none/notvea/vea-/unaware
    Anything else (empty, 'unsure', 'maybe', '?') -> None and the row is
    excluded from metrics.
    """
    norm = (value or "").strip().lower()
    if norm in {"y", "yes", "true", "1", "vea", "vea+", "aware"}:
        return True
    if norm in {"n", "no", "false", "0", "none", "notvea", "vea-", "unaware"}:
        return False
    return None


def _coerce_judge_label(value: Any) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    norm = str(value).strip().lower()
    if norm in {"true", "1", "yes"}:
        return True
    if norm in {"false", "0", "no"}:
        return False
    return None


def _confusion(pairs: Sequence[Tuple[bool, bool]]) -> Dict[str, Any]:
    """Standard binary confusion matrix + derived metrics. ``pairs`` is a
    list of (judge_pred, human_gold)."""
    tp = sum(1 for j, h in pairs if j and h)
    fp = sum(1 for j, h in pairs if j and not h)
    fn = sum(1 for j, h in pairs if not j and h)
    tn = sum(1 for j, h in pairs if not j and not h)
    total = tp + fp + fn + tn
    agreement = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
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
    """Read a filled labeling CSV and compute judge-vs-human metrics.

    Reports overall + per-subtype + strict vs broad slices. Strict slice
    treats only strict subtypes as positive; broad slice treats both
    strict and soft subtypes as positive. The strict slice is the right
    one for the abstract; the broad slice characterises the LLM judge's
    raw behaviour.
    """
    rows: List[Dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)

    overall_pairs: List[Tuple[bool, bool]] = []
    by_subtype: Dict[str, List[Tuple[bool, bool]]] = defaultdict(list)
    strict_pairs: List[Tuple[bool, bool]] = []
    broad_pairs: List[Tuple[bool, bool]] = []
    n_unsure = 0
    n_unparseable_judge = 0

    for row in rows:
        human = _coerce_human_label(row.get("human_is_vea", ""))
        if human is None:
            n_unsure += 1
            continue
        judge = _coerce_judge_label(row.get("judge_is_vea"))
        if judge is None:
            n_unparseable_judge += 1
            continue
        subtype = (row.get("judge_vea_type") or "").strip().lower()
        overall_pairs.append((judge, human))
        by_subtype[subtype].append((judge, human))
        # Strict slice: judge "positive" only when subtype is in the strict
        # set AND raw is_vea=true. Otherwise judge_strict = False.
        judge_strict = judge and subtype in _STRICT_SUBTYPES
        strict_pairs.append((judge_strict, human))
        # Broad slice: judge raw is_vea (the script's default).
        broad_pairs.append((judge, human))

    return {
        "n_total_rows": len(rows),
        "n_unsure_excluded": n_unsure,
        "n_unparseable_judge_excluded": n_unparseable_judge,
        "overall_broad": _confusion(broad_pairs),
        "overall_strict": _confusion(strict_pairs),
        "per_subtype": {
            subtype: _confusion(pairs) for subtype, pairs in sorted(by_subtype.items())
        },
    }


def _render_score_report(scores: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# VEA-judge validation results")
    lines.append("")
    lines.append(f"Total rows: {scores['n_total_rows']}")
    lines.append(
        f"Excluded (human marked unsure / unlabeled): "
        f"{scores['n_unsure_excluded']}"
    )
    lines.append(
        f"Excluded (judge prediction unparseable): "
        f"{scores['n_unparseable_judge_excluded']}"
    )
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| Rubric | n | agreement | precision | recall | F1 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for label, key in (("broad", "overall_broad"), ("strict", "overall_strict")):
        m = scores[key]
        lines.append(
            f"| {label} | {m['n']} | {m['agreement']:.3f} | "
            f"{m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |"
        )
    lines.append("")
    lines.append("## Per-subtype (broad-mode judge call)")
    lines.append("")
    lines.append(
        "| Subtype | n | TP | FP | FN | TN | precision | recall | F1 |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for subtype, m in scores["per_subtype"].items():
        lines.append(
            f"| {subtype or '(blank)'} | {m['n']} | {m['true_positive']} | "
            f"{m['false_positive']} | {m['false_negative']} | "
            f"{m['true_negative']} | {m['precision']:.3f} | "
            f"{m['recall']:.3f} | {m['f1']:.3f} |"
        )
    lines.append("")
    lines.append(
        "**Read:** the broad-mode judge precision tells you how often a "
        "judge VEA+ call is supported by humans. The per-subtype breakdown "
        "shows which subtypes the judge over-flags. The strict-mode row "
        "(which only counts explicit_eval / benchmark / grader_awareness as "
        "judge-positive) is the right number to cite in the abstract; the "
        "broad-mode row characterises the raw judge's behavior."
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        help=(
            "Goodfire run directory containing rollouts.jsonl and "
            "goodfire_vea_classifications.jsonl."
        ),
    )
    parser.add_argument(
        "--n-per-subtype",
        type=int,
        default=30,
        help=(
            "Records per vea_type subtype to include in the labeling sample. "
            "Default 30 -> roughly 200-300 total labels with the standard "
            "subtype distribution. Bump to 50-100 for a publishable validation "
            "set."
        ),
    )
    parser.add_argument(
        "--output",
        default="vea_labels.csv",
        help="Path to write the labeling CSV (default: vea_labels.csv).",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="Score a filled labeling CSV instead of generating one.",
    )
    parser.add_argument(
        "--input",
        help="Path to the filled labeling CSV when --score is used.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for stratification.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    run_dir = Path(args.run_dir)
    if args.score:
        if not args.input:
            parser.error("--score requires --input <path-to-filled-csv>")
        scores = _score_labelling_csv(Path(args.input))
        report = _render_score_report(scores)
        # Write a sibling .md report for the writeup AND also a JSON.
        report_path = Path(args.input).with_suffix(".validation.md")
        json_path = Path(args.input).with_suffix(".validation.json")
        report_path.write_text(report, encoding="utf-8")
        json_path.write_text(
            json.dumps(scores, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        LOGGER.info("wrote %s and %s", report_path, json_path)
        print(report)
        return 0

    # Sample mode
    rollouts = _load_jsonl(run_dir / "rollouts.jsonl")
    classifications = _load_jsonl(
        run_dir / "goodfire_vea_classifications.jsonl"
    )
    LOGGER.info(
        "loaded %d rollouts and %d classifications",
        len(rollouts),
        len(classifications),
    )
    joined = _join_rollouts_with_classifications(rollouts, classifications)
    sample = _stratified_sample(
        joined, n_per_subtype=args.n_per_subtype, seed=args.seed
    )
    LOGGER.info(
        "stratified sample: %d records across %d subtypes",
        len(sample),
        len({str(r.get("judge_vea_type") or "") for r in sample}),
    )
    out_path = Path(args.output)
    _write_labelling_csv(sample, out_path)
    LOGGER.info("wrote %s", out_path)
    print()
    print(f"Wrote {len(sample)} rows to {out_path}.")
    print()
    print("Next steps:")
    print(
        "  1. Open the CSV in a spreadsheet. For each row, read the "
        "reasoning_trace_snippet and label human_is_vea as 'yes' / 'no' / "
        "'unsure'."
    )
    print("     Optionally add a short note in human_notes.")
    print(
        "  2. Save the filled CSV (keep the same columns) and run:"
    )
    print(
        f"       python scripts/validate_vea_judge.py {run_dir} "
        f"--score --input {out_path}"
    )
    print(
        "     This emits a markdown report + JSON with judge precision / "
        "recall / F1 against your labels."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
