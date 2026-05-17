"""V2 deep-dive: per-stratum interaction effects on one model.

Default target is opus-4-7. Reads ``rollouts.jsonl`` +
``goodfire_vea_classifications.jsonl`` from a V2 run directory,
filters to the requested model, and runs the same factorial analysis
(``scripts.analyze_v2_factorial.analyze``) on each stratum.

Strata produced:

  * by ``apollo_eval_type`` (``capability`` / ``alignment``). Each eval
    stratum is paired against the FULL deployment-cells of the same
    model, so every stratum has all 4 factorial cells.
  * by ``apollo_scaffold_type`` (``agentic`` / ``chat``).
  * Top-k Apollo categories ranked by interaction magnitude
    (descriptive only — small per-category n).

Uses ``min_cell_n = 10`` (vs the pre-reg's 30) because per-stratum
samples are smaller. **Outputs are exploratory, not confirmatory.**
The pre-reg's cross-model headline still stands; this script answers
"is the opus interaction concentrated in a subset of prompts?"

Usage::

    python scripts/analyze_v2_opus_deepdive.py \\
      --run-dir runs/v2-apollo-factorial-prereg \\
      --strict-vea
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_v2_factorial import (  # noqa: E402
    _DEFAULT_BOOTSTRAP_SAMPLES,
    _load_jsonl,
    analyze,
)

LOGGER = logging.getLogger("v2_opus_deepdive")

_DEFAULT_MIN_CELL_N = 10  # exploratory; pre-reg uses 30 for confirmatory
_DEFAULT_MODEL_FILTER = "anthropic:claude-opus-4-7"


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------


def _filter_to_model(
    rollouts: Sequence[Dict[str, Any]], model_name: str
) -> List[Dict[str, Any]]:
    return [r for r in rollouts if r.get("model_name") == model_name]


def _split_eval_deployment(
    rollouts: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    eval_rollouts = [r for r in rollouts if str(r.get("apollo_is_eval", "")).lower() == "true"]
    deploy_rollouts = [r for r in rollouts if str(r.get("apollo_is_eval", "")).lower() == "false"]
    return eval_rollouts, deploy_rollouts


def _by_field(
    rollouts: Sequence[Dict[str, Any]], field: str
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rollouts:
        out[str(r.get(field, "") or "(missing)")].append(r)
    return dict(out)


def _strata_paired_with_deployment(
    rollouts: Sequence[Dict[str, Any]],
    eval_stratum_field: str,
    *,
    skip_empty_stratum: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """Build strata: each value of eval_stratum_field gets all matching eval
    rollouts paired with the FULL deployment set. Strata with no eval
    rollouts are skipped (controlled by ``skip_empty_stratum``).
    """
    eval_rollouts, deploy_rollouts = _split_eval_deployment(rollouts)
    eval_by_stratum = _by_field(eval_rollouts, eval_stratum_field)
    strata: Dict[str, List[Dict[str, Any]]] = {}
    for stratum_value, eval_subset in eval_by_stratum.items():
        if skip_empty_stratum and not eval_subset:
            continue
        if stratum_value == "(missing)":
            # eval_type is only set on eval rows; deployment rows have it blank
            continue
        strata[stratum_value] = list(eval_subset) + list(deploy_rollouts)
    return strata


def _strata_full_split(
    rollouts: Sequence[Dict[str, Any]], field: str
) -> Dict[str, List[Dict[str, Any]]]:
    """Split rollouts entirely by a metadata field. Used for symmetric
    binary fields like scaffold_type that exist on both eval and
    deployment rows."""
    return _by_field(rollouts, field)


# ---------------------------------------------------------------------------
# Per-stratum analysis
# ---------------------------------------------------------------------------


def _stratum_summary(
    rollouts: Sequence[Dict[str, Any]],
    classifications: Sequence[Dict[str, Any]],
    *,
    strict_vea: bool,
    bootstrap_samples: int,
    min_cell_n: int,
    seed: int,
) -> Dict[str, Any]:
    """Run analyze() on a stratum subset, return the per-model entry."""
    if not rollouts:
        return {"n_rollouts": 0, "skipped": "empty stratum"}
    cls_keys = {
        (
            str(r.get("model_name", "")),
            str(r.get("condition", "")),
            str(r.get("example_id", "")),
            int(r.get("rollout_index", 0) or 0),
        )
        for r in rollouts
    }
    stratum_cls = [
        c
        for c in classifications
        if (
            str(c.get("model_name", "")),
            str(c.get("condition", "")),
            str(c.get("example_id", "")),
            int(c.get("rollout_index", 0) or 0),
        )
        in cls_keys
    ]
    summary = analyze(
        rollouts,
        stratum_cls or None,
        strict_vea=strict_vea,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        min_cell_n=min_cell_n,
    )
    per_model = summary.get("per_model") or []
    if not per_model:
        return {"n_rollouts": len(rollouts), "skipped": "no per-model entry"}
    entry = per_model[0]
    entry["n_classifications"] = len(stratum_cls)
    return entry


def _per_category_ranking(
    rollouts: Sequence[Dict[str, Any]],
    classifications: Sequence[Dict[str, Any]],
    *,
    strict_vea: bool,
    bootstrap_samples: int,
    min_per_category: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Rank Apollo categories by interaction magnitude. Per-category n is
    typically small so this is descriptive, not inferential."""
    by_cat = _by_field(rollouts, "apollo_category")
    _, deploy_rollouts = _split_eval_deployment(rollouts)
    ranked: List[Dict[str, Any]] = []
    for cat, cat_rollouts in by_cat.items():
        eval_subset = [r for r in cat_rollouts if str(r.get("apollo_is_eval", "")).lower() == "true"]
        if len(eval_subset) < min_per_category:
            continue
        stratum = eval_subset + deploy_rollouts
        result = _stratum_summary(
            stratum, classifications,
            strict_vea=strict_vea, bootstrap_samples=bootstrap_samples,
            min_cell_n=min_per_category, seed=seed,
        )
        if "interaction_effect" not in result:
            continue
        ranked.append(
            {
                "category": cat,
                "n_eval": len(eval_subset),
                "interaction_point": result["interaction_effect"]["point"],
                "interaction_ci": result["interaction_effect"]["ci"],
                "interaction_p": result["interaction_effect"].get("p_value"),
                "verdict": result.get("verdict"),
            }
        )
    ranked.sort(
        key=lambda row: abs(float(row.get("interaction_point") or 0.0)),
        reverse=True,
    )
    return ranked


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _render_report(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"# V2 deep-dive — model: `{summary['model_name']}`")
    lines.append("")
    lines.append(
        f"- vea_strictness: `{summary['vea_strictness']}`"
    )
    lines.append(
        f"- bootstrap_samples: `{summary['bootstrap_samples']}` | min_cell_n: `{summary['min_cell_n']}`"
    )
    lines.append(
        f"- baseline (full corpus, all rollouts): interaction = {_fmt_effect(summary['baseline'])}"
    )
    lines.append("")
    for header, key in (
        ("By `apollo_eval_type`", "by_eval_type"),
        ("By `apollo_scaffold_type`", "by_scaffold_type"),
    ):
        block = summary.get(key, {})
        if not block:
            continue
        lines.append(f"## {header}")
        lines.append("")
        lines.append("| Stratum | cells (n) | interaction | 95% CI | p | verdict |")
        lines.append("| --- | --- | ---: | --- | ---: | --- |")
        for stratum, entry in sorted(block.items()):
            if entry.get("skipped"):
                lines.append(f"| `{stratum}` | — | — | — | — | _{entry['skipped']}_ |")
                continue
            cells = entry.get("cells") or {}
            cell_ns = "/".join(str(cells.get(k, {}).get("n", 0)) for k in (
                "neutral_eval", "inject_eval", "neutral_deployment", "inject_deployment",
            ))
            eff = entry["interaction_effect"]
            ci = eff["ci"]
            lines.append(
                f"| `{stratum}` | {cell_ns} | {float(eff['point']):+.4f} | "
                f"[{float(ci['low']):+.4f}, {float(ci['high']):+.4f}] | "
                f"{float(eff.get('p_value', 1.0)):.4f} | {entry.get('verdict', '?')} |"
            )
        lines.append("")

    ranking = summary.get("per_category_ranking") or []
    if ranking:
        lines.append("## Top categories by |interaction|")
        lines.append("")
        lines.append(
            "Descriptive only — per-category n is small. Each row pairs a category's"
            " eval rollouts against the model's full deployment set."
        )
        lines.append("")
        lines.append("| Category | n_eval | interaction | 95% CI | p | verdict |")
        lines.append("| --- | ---: | ---: | --- | ---: | --- |")
        for row in ranking[:15]:
            ci = row["interaction_ci"]
            lines.append(
                f"| `{row['category']}` | {row['n_eval']} | "
                f"{float(row['interaction_point']):+.4f} | "
                f"[{float(ci['low']):+.4f}, {float(ci['high']):+.4f}] | "
                f"{float(row.get('interaction_p', 1.0)):.4f} | {row.get('verdict', '?')} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _fmt_effect(effect: Dict[str, Any]) -> str:
    if not effect or effect.get("point") is None:
        return "—"
    ci = effect.get("ci", {})
    return (
        f"{float(effect['point']):+.4f} "
        f"[{float(ci.get('low', 0.0)):+.4f}, {float(ci.get('high', 0.0)):+.4f}]"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def deepdive(
    rollouts: Sequence[Dict[str, Any]],
    classifications: Sequence[Dict[str, Any]],
    *,
    model_filter: str,
    strict_vea: bool,
    bootstrap_samples: int,
    seed: int,
    min_cell_n: int,
    min_per_category: int,
) -> Dict[str, Any]:
    model_rollouts = _filter_to_model(rollouts, model_filter)
    if not model_rollouts:
        raise SystemExit(f"no rollouts for model {model_filter!r}")

    baseline = _stratum_summary(
        model_rollouts, classifications,
        strict_vea=strict_vea, bootstrap_samples=bootstrap_samples,
        min_cell_n=min_cell_n, seed=seed,
    )

    by_eval_type = {}
    for stratum, stratum_rollouts in _strata_paired_with_deployment(
        model_rollouts, "apollo_eval_type"
    ).items():
        by_eval_type[stratum] = _stratum_summary(
            stratum_rollouts, classifications,
            strict_vea=strict_vea, bootstrap_samples=bootstrap_samples,
            min_cell_n=min_cell_n, seed=seed,
        )

    by_scaffold = {}
    for stratum, stratum_rollouts in _strata_full_split(
        model_rollouts, "apollo_scaffold_type"
    ).items():
        by_scaffold[stratum] = _stratum_summary(
            stratum_rollouts, classifications,
            strict_vea=strict_vea, bootstrap_samples=bootstrap_samples,
            min_cell_n=min_cell_n, seed=seed,
        )

    per_category = _per_category_ranking(
        model_rollouts, classifications,
        strict_vea=strict_vea, bootstrap_samples=bootstrap_samples,
        min_per_category=min_per_category, seed=seed,
    )

    return {
        "model_name": model_filter,
        "vea_strictness": "strict" if strict_vea else "broad",
        "bootstrap_samples": bootstrap_samples,
        "min_cell_n": min_cell_n,
        "min_per_category": min_per_category,
        "baseline": baseline.get("interaction_effect", {}),
        "baseline_cells": baseline.get("cells", {}),
        "baseline_n_rollouts": baseline.get("n_rollouts"),
        "baseline_verdict": baseline.get("verdict"),
        "by_eval_type": by_eval_type,
        "by_scaffold_type": by_scaffold,
        "per_category_ranking": per_category,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--model-filter",
        default=_DEFAULT_MODEL_FILTER,
        help="Restrict the deep-dive to one model_name (default: opus-4-7).",
    )
    parser.add_argument("--rollouts-path", type=Path, default=None)
    parser.add_argument("--classifications-path", type=Path, default=None)
    parser.add_argument("--strict-vea", action="store_true")
    parser.add_argument(
        "--bootstrap-samples", type=int, default=_DEFAULT_BOOTSTRAP_SAMPLES
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-cell-n",
        type=int,
        default=_DEFAULT_MIN_CELL_N,
        help="Minimum rollouts per cell for stratum verdicts (default 10, exploratory).",
    )
    parser.add_argument(
        "--min-per-category",
        type=int,
        default=8,
        help="Minimum eval rollouts in a category before it's eligible for the ranking.",
    )
    parser.add_argument("--out-suffix", default="")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    rollouts_path = args.rollouts_path or (args.run_dir / "rollouts.jsonl")
    classifications_path = args.classifications_path or (
        args.run_dir / "goodfire_vea_classifications.jsonl"
    )
    if not rollouts_path.exists():
        LOGGER.error("rollouts file not found: %s", rollouts_path)
        return 1
    rollouts = _load_jsonl(rollouts_path)
    classifications = (
        _load_jsonl(classifications_path) if classifications_path.exists() else []
    )
    LOGGER.info(
        "loaded %d rollouts + %d classifications; filtering to %s",
        len(rollouts), len(classifications), args.model_filter,
    )

    summary = deepdive(
        rollouts, classifications,
        model_filter=args.model_filter,
        strict_vea=args.strict_vea,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        min_cell_n=args.min_cell_n,
        min_per_category=args.min_per_category,
    )

    suffix = args.out_suffix
    safe_name = args.model_filter.replace(":", "_").replace("/", "_")
    json_path = args.run_dir / f"deepdive_{safe_name}{suffix}.json"
    md_path = args.run_dir / f"deepdive_{safe_name}{suffix}.md"
    args.run_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    md_path.write_text(_render_report(summary), encoding="utf-8")
    LOGGER.info("wrote %s", json_path)
    LOGGER.info("wrote %s", md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
