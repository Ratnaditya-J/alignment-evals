"""Re-aggregate a cross-protocol run from its saved rollouts.jsonl.

Use this when you ran ``run_cross_protocol_comparison.py`` with
``--save-rollouts`` and the refusal scorer (or any other scoring logic)
has since been improved. It re-applies the current ``RefusalScorer`` to
every saved response and rewrites ``cross_protocol_summary.json`` and
``cross_protocol_report.md`` in place. **No new API calls are made.**

It also reports per-model statistics on empty/missing responses, which
flag measurement issues that need a fresh run rather than a re-score
(e.g. reasoning models running out of token budget mid-thought).

Usage:
    python scripts/rescore_cross_protocol.py runs/cross-protocol-v2/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Import re-uses to avoid duplicating bootstrap / report logic.
from scripts.run_cross_protocol_comparison import (  # noqa: E402
    ArxivRunResult,
    GoodfireRunResult,
    _bootstrap_delta,
    _render_comparison_md,
)
from src.eval_awareness import RefusalScorer, ScoringContext  # noqa: E402

LOGGER = logging.getLogger("rescore")


def _rescore(rollouts: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    scorer = RefusalScorer()
    rescored: List[Dict[str, Any]] = []
    for record in rollouts:
        response = record.get("response", "") or ""
        new_score = scorer.score(response, ScoringContext(task_id="rescore")).score
        rescored.append({**record, "refusal": new_score})
    return rescored


def _aggregate_arxiv(records: List[Dict[str, Any]]) -> List[ArxivRunResult]:
    by_model: Dict[str, Dict[str, List[float]]] = {}
    for record in records:
        model_name = record["model_name"]
        condition = record["condition"]
        by_model.setdefault(
            model_name, {"baseline": [], "inject": [], "remove": [], "neutral": []}
        )[condition].append(record["refusal"])
    results: List[ArxivRunResult] = []
    for model_name, refusals in sorted(by_model.items()):
        baseline = refusals.get("baseline", [])
        injected = refusals.get("inject", [])
        removed = refusals.get("remove", [])
        neutral = refusals.get("neutral", [])
        baseline_rate = sum(baseline) / max(1, len(baseline))
        injected_rate = sum(injected) / max(1, len(injected))
        removed_rate = sum(removed) / max(1, len(removed))
        neutral_rate = sum(neutral) / max(1, len(neutral))
        results.append(
            ArxivRunResult(
                model_name=model_name,
                baseline_refusal_rate=round(baseline_rate, 4),
                injected_refusal_rate=round(injected_rate, 4),
                removed_refusal_rate=round(removed_rate, 4),
                neutral_refusal_rate=round(neutral_rate, 4),
                inject_minus_baseline=round(injected_rate - baseline_rate, 4),
                inject_minus_baseline_ci=_bootstrap_delta(baseline, injected),
                remove_minus_baseline=round(removed_rate - baseline_rate, 4),
                remove_minus_baseline_ci=_bootstrap_delta(baseline, removed),
                inject_minus_neutral=round(injected_rate - neutral_rate, 4),
                inject_minus_neutral_ci=_bootstrap_delta(neutral, injected),
                remove_minus_neutral=round(removed_rate - neutral_rate, 4),
                remove_minus_neutral_ci=_bootstrap_delta(neutral, removed),
                neutral_minus_baseline=round(neutral_rate - baseline_rate, 4),
                neutral_minus_baseline_ci=_bootstrap_delta(baseline, neutral),
            )
        )
    return results


def _empty_report(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    report: Dict[str, Dict[str, int]] = {}
    for record in records:
        model = record["model_name"]
        cond = record["condition"]
        slot = report.setdefault(model, {"empty": 0, "total": 0})
        slot["total"] += 1
        if not (record.get("response") or "").strip():
            slot["empty"] += 1
    return report


def main(argv: Iterable[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Run directory containing rollouts.jsonl")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help=(
            "Overwrite cross_protocol_summary.json and cross_protocol_report.md "
            "in run_dir. Without this flag the rescored files are written next "
            "to the originals with a .rescored suffix."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    run_dir = Path(args.run_dir)
    rollouts_path = run_dir / "rollouts.jsonl"
    summary_path = run_dir / "cross_protocol_summary.json"
    if not rollouts_path.exists():
        raise SystemExit(
            f"missing {rollouts_path}; rerun with --save-rollouts to enable rescoring"
        )

    LOGGER.info("loading rollouts from %s", rollouts_path)
    with rollouts_path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    LOGGER.info("loaded %d records", len(records))

    empties = _empty_report(records)
    LOGGER.info("empty-response stats per model:")
    for model, stats in sorted(empties.items()):
        pct = stats["empty"] / max(1, stats["total"])
        flag = "  ⚠️ MOST RESPONSES EMPTY" if pct >= 0.5 else ""
        LOGGER.info(
            "  %s: %d/%d empty (%.1f%%)%s",
            model,
            stats["empty"],
            stats["total"],
            100 * pct,
            flag,
        )

    rescored = _rescore(records)
    arxiv = _aggregate_arxiv(rescored)

    # Read original summary to preserve metadata and goodfire results.
    original = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    payload = {
        **original,
        "arxiv_protocol": [r.__dict__ for r in arxiv],
        "rescored": True,
        "rescored_records": len(records),
        "empty_response_stats": empties,
    }
    goodfire_dicts = original.get("goodfire_protocol", []) or []
    goodfire = [
        GoodfireRunResult(
            **{k: v for k, v in g.items() if k in GoodfireRunResult.__annotations__}
        )
        for g in goodfire_dicts
    ]

    suffix = "" if args.in_place else ".rescored"
    out_summary = run_dir / f"cross_protocol_summary{suffix}.json"
    out_report = run_dir / f"cross_protocol_report{suffix}.md"
    out_summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    out_report.write_text(_render_comparison_md(goodfire, arxiv), encoding="utf-8")
    LOGGER.info("wrote %s and %s", out_summary, out_report)

    print()
    print(
        f"Rescored {len(records)} records ({sum(s['empty'] for s in empties.values())} empty)."
    )
    print(f"Updated:")
    print(f"  - {out_summary}")
    print(f"  - {out_report}")


if __name__ == "__main__":
    main()
