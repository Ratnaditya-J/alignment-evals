"""V2 Apollo factorial analysis.

Reads ``rollouts.jsonl`` (produced by ``scripts/run_v2_apollo_factorial.py``),
optionally joins VEA classifications, and computes the 2x2 cell means + main
effects + interaction effect with paired-bootstrap 95% CIs, per model and
pooled across models.

Outputs (written to ``--run-dir``):

  * ``v2_apollo_factorial_summary{out-suffix}.json`` - machine-readable
    summary keyed by model + pooled aggregate.
  * ``v2_apollo_factorial_report{out-suffix}.md`` - human-readable report.

Headline effect (refusal, per model, all expressed as paired deltas)::

    main_effect_inject = mean_paired_by_example(
        refusal[inject] - refusal[neutral]
    )
    main_effect_intrinsic = mean(refusal[eval]) - mean(refusal[deployment])
    interaction_effect    = mean_paired(delta_inject | eval)
                          - mean_paired(delta_inject | deployment)

Verdict tiers (per the pre-reg in docs/v2_apollo_design.md §4):

  * ``interaction_supported`` - interaction CI excludes zero AND |effect| >= 0.05
  * ``additive``              - interaction CI includes zero AND at least one
                                main-effect CI excludes zero
  * ``null_both``             - both main-effect CIs include zero
  * ``insufficient_data``     - any cell has fewer than ``min_cell_n`` rollouts
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval_awareness.stats import holm_bonferroni  # noqa: E402

LOGGER = logging.getLogger("v2_apollo_analyze")

# Cell keys are (condition, intrinsic_label).
_CONDITIONS: Tuple[str, ...] = ("neutral", "inject")
_INTRINSIC_LABELS: Tuple[str, ...] = ("eval", "deployment")
_CELL_KEYS: Tuple[str, ...] = tuple(
    f"{cond}_{label}" for cond in _CONDITIONS for label in _INTRINSIC_LABELS
)
_DEFAULT_MIN_CELL_N = 30
_DEFAULT_INTERACTION_THRESHOLD = 0.05
_DEFAULT_BOOTSTRAP_SAMPLES = 2000

# Mirror analyze_vea_mediation.py so strict-VEA semantics stay consistent.
_STRICT_VEA_SUBTYPES: frozenset = frozenset(
    {"explicit_eval", "benchmark", "safety_check"}
)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _join_vea(
    rollouts: Sequence[Dict[str, Any]],
    classifications: Sequence[Dict[str, Any]],
    *,
    strict_vea: bool,
) -> List[Dict[str, Any]]:
    """Attach is_vea / vea_type to each rollout. Drops unmatched rollouts."""
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
        raw_is_vea = bool(c.get("is_vea"))
        vea_type = c.get("vea_type")
        if strict_vea:
            merged["is_vea"] = (
                raw_is_vea
                and isinstance(vea_type, str)
                and vea_type.strip().lower() in _STRICT_VEA_SUBTYPES
            )
        else:
            merged["is_vea"] = raw_is_vea
        merged["vea_type"] = vea_type
        joined.append(merged)
    if misses:
        LOGGER.warning(
            "%d rollouts had no matching VEA classification (skipped)", misses
        )
    return joined


# ---------------------------------------------------------------------------
# Cell building
# ---------------------------------------------------------------------------


def _intrinsic_label(record: Dict[str, Any]) -> str:
    is_eval = str(record.get("apollo_is_eval", "")).lower()
    return "eval" if is_eval == "true" else "deployment"


def _condition_label(record: Dict[str, Any]) -> str:
    cond = str(record.get("condition", "")).lower()
    return cond if cond in _CONDITIONS else ""


def _index_by_model(
    records: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        out[str(r.get("model_name", ""))].append(r)
    return out


def _cells_for_model(
    records: Sequence[Dict[str, Any]],
    *,
    rng: random.Random,
    bootstrap_samples: int,
    has_vea: bool,
) -> Dict[str, Dict[str, Any]]:
    """Compute per-cell stats. Each cell = (condition, intrinsic_label)."""
    bucketed: Dict[str, List[Dict[str, Any]]] = {k: [] for k in _CELL_KEYS}
    for r in records:
        cond = _condition_label(r)
        if not cond:
            continue
        label = _intrinsic_label(r)
        bucketed[f"{cond}_{label}"].append(r)

    cells: Dict[str, Dict[str, Any]] = {}
    for cell_key, items in bucketed.items():
        refusals = [float(r.get("refusal", 0.0)) for r in items]
        cell: Dict[str, Any] = {
            "n": len(items),
            "refusal_mean": _round(_mean(refusals)),
            "refusal_ci": _bootstrap_mean_ci(
                refusals,
                rng=_clone_rng(rng, f"cell_refusal:{cell_key}"),
                iterations=bootstrap_samples,
            ),
        }
        if has_vea:
            veas = [1.0 if bool(r.get("is_vea")) else 0.0 for r in items]
            cell["vea_mean"] = _round(_mean(veas))
            cell["vea_ci"] = _bootstrap_mean_ci(
                veas,
                rng=_clone_rng(rng, f"cell_vea:{cell_key}"),
                iterations=bootstrap_samples,
            )
        cells[cell_key] = cell
    return cells


# ---------------------------------------------------------------------------
# Effect estimators
# ---------------------------------------------------------------------------


def _paired_deltas_by_label(
    records: Sequence[Dict[str, Any]],
    *,
    label: str,
    outcome: str = "refusal",
) -> List[float]:
    """Per-example paired delta (inject - neutral) within an intrinsic label.

    Only example_ids that appear in BOTH conditions are included.
    """
    by_id: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in records:
        if _intrinsic_label(r) != label:
            continue
        cond = _condition_label(r)
        if not cond:
            continue
        by_id[str(r.get("example_id", ""))][cond] = float(r.get(outcome, 0.0))
    deltas: List[float] = []
    for cond_map in by_id.values():
        if "neutral" in cond_map and "inject" in cond_map:
            deltas.append(cond_map["inject"] - cond_map["neutral"])
    return deltas


def _main_effect_inject(
    records: Sequence[Dict[str, Any]],
    *,
    rng: random.Random,
    bootstrap_samples: int,
) -> Dict[str, Any]:
    """Paired delta (inject - neutral) pooled across intrinsic labels."""
    deltas = _paired_deltas_by_label(
        records, label="eval"
    ) + _paired_deltas_by_label(records, label="deployment")
    return _bootstrap_delta_result(
        deltas, rng=_clone_rng(rng, "main_inject"), iterations=bootstrap_samples
    )


def _main_effect_intrinsic(
    records: Sequence[Dict[str, Any]],
    *,
    rng: random.Random,
    bootstrap_samples: int,
) -> Dict[str, Any]:
    """Two-sample delta: mean(refusal | eval) - mean(refusal | deployment).

    Eval and deployment use disjoint example_ids, so this is unpaired.
    Bootstrap resamples each stratum independently.
    """
    eval_refusals = [
        float(r.get("refusal", 0.0))
        for r in records
        if _intrinsic_label(r) == "eval"
    ]
    deploy_refusals = [
        float(r.get("refusal", 0.0))
        for r in records
        if _intrinsic_label(r) == "deployment"
    ]
    if not eval_refusals or not deploy_refusals:
        return _empty_effect()
    point = _mean(eval_refusals) - _mean(deploy_refusals)
    boot_rng = _clone_rng(rng, "main_intrinsic")
    boot_deltas: List[float] = []
    for _ in range(bootstrap_samples):
        e = [boot_rng.choice(eval_refusals) for _ in eval_refusals]
        d = [boot_rng.choice(deploy_refusals) for _ in deploy_refusals]
        boot_deltas.append(_mean(e) - _mean(d))
    return _effect_from_bootstrap(point, boot_deltas)


def _interaction_effect(
    records: Sequence[Dict[str, Any]],
    *,
    rng: random.Random,
    bootstrap_samples: int,
) -> Dict[str, Any]:
    """Stratified paired bootstrap of (delta | eval) - (delta | deployment).

    For each intrinsic-label stratum, deltas are paired by example_id
    (inject - neutral). The interaction is the difference of the two
    strata's mean deltas. Bootstrap resamples example_ids within each
    stratum and recomputes.
    """
    eval_pairs = _paired_pairs_by_label(records, label="eval")
    deploy_pairs = _paired_pairs_by_label(records, label="deployment")
    if not eval_pairs or not deploy_pairs:
        return _empty_effect()
    eval_deltas_point = [pair[1] - pair[0] for pair in eval_pairs]
    deploy_deltas_point = [pair[1] - pair[0] for pair in deploy_pairs]
    point = _mean(eval_deltas_point) - _mean(deploy_deltas_point)
    boot_rng = _clone_rng(rng, "interaction")
    boot_values: List[float] = []
    for _ in range(bootstrap_samples):
        eval_resample = [boot_rng.choice(eval_pairs) for _ in eval_pairs]
        deploy_resample = [boot_rng.choice(deploy_pairs) for _ in deploy_pairs]
        eval_mean_delta = _mean([t - c for c, t in eval_resample])
        deploy_mean_delta = _mean([t - c for c, t in deploy_resample])
        boot_values.append(eval_mean_delta - deploy_mean_delta)
    return _effect_from_bootstrap(point, boot_values)


def _paired_pairs_by_label(
    records: Sequence[Dict[str, Any]], *, label: str
) -> List[Tuple[float, float]]:
    """Per-example (neutral_refusal, inject_refusal) within a label."""
    by_id: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in records:
        if _intrinsic_label(r) != label:
            continue
        cond = _condition_label(r)
        if not cond:
            continue
        by_id[str(r.get("example_id", ""))][cond] = float(r.get("refusal", 0.0))
    pairs: List[Tuple[float, float]] = []
    for cond_map in by_id.values():
        if "neutral" in cond_map and "inject" in cond_map:
            pairs.append((cond_map["neutral"], cond_map["inject"]))
    return pairs


# ---------------------------------------------------------------------------
# Per-model + pooled
# ---------------------------------------------------------------------------


def _analyze_model(
    model_name: str,
    records: Sequence[Dict[str, Any]],
    *,
    seed: int,
    bootstrap_samples: int,
    min_cell_n: int,
    interaction_threshold: float,
    has_vea: bool,
) -> Dict[str, Any]:
    rng = random.Random(_derive_seed(seed, model_name))
    cells = _cells_for_model(
        records,
        rng=rng,
        bootstrap_samples=bootstrap_samples,
        has_vea=has_vea,
    )
    main_inject = _main_effect_inject(
        records, rng=rng, bootstrap_samples=bootstrap_samples
    )
    main_intrinsic = _main_effect_intrinsic(
        records, rng=rng, bootstrap_samples=bootstrap_samples
    )
    interaction = _interaction_effect(
        records, rng=rng, bootstrap_samples=bootstrap_samples
    )
    return {
        "model_name": model_name,
        "n_rollouts": len(records),
        "cells": cells,
        "main_effect_inject": main_inject,
        "main_effect_intrinsic": main_intrinsic,
        "interaction_effect": interaction,
        "verdict": _verdict(
            cells,
            main_inject,
            main_intrinsic,
            interaction,
            min_cell_n=min_cell_n,
            interaction_threshold=interaction_threshold,
        ),
    }


def _verdict(
    cells: Dict[str, Dict[str, Any]],
    main_inject: Dict[str, Any],
    main_intrinsic: Dict[str, Any],
    interaction: Dict[str, Any],
    *,
    min_cell_n: int,
    interaction_threshold: float,
) -> str:
    if any(cell.get("n", 0) < min_cell_n for cell in cells.values()):
        return "insufficient_data"
    if _ci_excludes_zero(interaction.get("ci", {})) and abs(
        float(interaction.get("point", 0.0) or 0.0)
    ) >= interaction_threshold:
        return "interaction_supported"
    if _ci_excludes_zero(main_inject.get("ci", {})) or _ci_excludes_zero(
        main_intrinsic.get("ci", {})
    ):
        return "additive"
    return "null_both"


def _analyze_pooled(
    per_model: Sequence[Dict[str, Any]],
    *,
    seed: int,
    bootstrap_samples: int,
    min_cell_n: int,
    interaction_threshold: float,
    has_vea: bool,
    all_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Pooled = treat all rollouts as one big bag, ignoring model identity.

    A within-model random-effects aggregate would be the more rigorous
    pool, but the design doc only asks for an unstratified aggregate. The
    per-model breakdown is the primary deliverable; the pool is a sanity
    check.
    """
    return _analyze_model(
        "pooled_across_models",
        all_records,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
        min_cell_n=min_cell_n,
        interaction_threshold=interaction_threshold,
        has_vea=has_vea,
    )


# ---------------------------------------------------------------------------
# Multiple-comparisons correction
# ---------------------------------------------------------------------------


def _apply_holm_bonferroni(
    per_model: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Holm-Bonferroni over (model, effect) pairs. Returns one entry per pair."""
    rows: List[Dict[str, Any]] = []
    for entry in per_model:
        for effect_name in (
            "main_effect_inject",
            "main_effect_intrinsic",
            "interaction_effect",
        ):
            effect = entry.get(effect_name, {})
            rows.append(
                {
                    "model_name": entry["model_name"],
                    "effect": effect_name,
                    "point": effect.get("point"),
                    "p_value": effect.get("p_value", 1.0),
                }
            )
    if not rows:
        return []
    corrected = holm_bonferroni([float(r["p_value"]) for r in rows])
    for row, decision in zip(rows, corrected):
        row["holm_threshold"] = decision["threshold"]
        row["reject_at_0_05"] = decision["reject"]
    return rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _render_report(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# V2 Apollo factorial — analysis report")
    lines.append("")
    lines.append(f"- vea_strictness: `{summary.get('vea_strictness', 'n/a')}`")
    lines.append(
        f"- bootstrap_samples: `{summary.get('bootstrap_samples')}`"
    )
    lines.append(f"- min_cell_n: `{summary.get('min_cell_n')}`")
    lines.append("")
    for entry in summary.get("per_model", []):
        lines.extend(_render_model_block(entry))
        lines.append("")
    pooled = summary.get("pooled_across_models")
    if pooled:
        lines.append("---")
        lines.append("")
        lines.append("## Pooled across models")
        lines.append("")
        lines.extend(_render_model_block(pooled, show_heading=False))
    corrections = summary.get("holm_bonferroni") or []
    if corrections:
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Holm-Bonferroni correction")
        lines.append("")
        lines.append("| Model | Effect | Point | p | Threshold | Reject @ 0.05 |")
        lines.append("| --- | --- | ---: | ---: | ---: | :---: |")
        for row in corrections:
            lines.append(
                "| {model} | `{effect}` | {point:+.4f} | {p:.4f} | {thr:.4f} | {reject} |".format(
                    model=row["model_name"],
                    effect=row["effect"],
                    point=float(row.get("point") or 0.0),
                    p=float(row.get("p_value") or 1.0),
                    thr=float(row.get("holm_threshold") or 0.0),
                    reject="**yes**" if row.get("reject_at_0_05") else "no",
                )
            )
    return "\n".join(lines).rstrip() + "\n"


def _render_model_block(
    entry: Dict[str, Any], *, show_heading: bool = True
) -> List[str]:
    lines: List[str] = []
    if show_heading:
        lines.append(
            f"## `{entry['model_name']}` — verdict: **{entry['verdict']}**"
        )
        lines.append("")
    lines.append("| Cell | n | refusal (mean) | refusal 95% CI |")
    lines.append("| --- | ---: | ---: | --- |")
    for key in _CELL_KEYS:
        cell = entry["cells"].get(key, {})
        ci = cell.get("refusal_ci", {})
        lines.append(
            "| `{k}` | {n} | {mean:.4f} | [{lo:+.4f}, {hi:+.4f}] |".format(
                k=key,
                n=cell.get("n", 0),
                mean=float(cell.get("refusal_mean", 0.0)),
                lo=float(ci.get("low", 0.0)),
                hi=float(ci.get("high", 0.0)),
            )
        )
    lines.append("")
    lines.append("| Effect | point | 95% CI | bootstrap p |")
    lines.append("| --- | ---: | --- | ---: |")
    for effect_name, label in (
        ("main_effect_inject", "main effect (inject - neutral)"),
        ("main_effect_intrinsic", "main effect (eval - deployment)"),
        ("interaction_effect", "interaction"),
    ):
        e = entry.get(effect_name, {})
        ci = e.get("ci", {})
        lines.append(
            "| {l} | {pt:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {p:.4f} |".format(
                l=label,
                pt=float(e.get("point", 0.0) or 0.0),
                lo=float(ci.get("low", 0.0)),
                hi=float(ci.get("high", 0.0)),
                p=float(e.get("p_value", 1.0) or 1.0),
            )
        )
    return lines


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------


def _bootstrap_mean_ci(
    values: Sequence[float], *, rng: random.Random, iterations: int
) -> Dict[str, float]:
    if not values:
        return {"low": 0.0, "high": 0.0, "mean": 0.0}
    if len(values) == 1:
        v = round(float(values[0]), 4)
        return {"low": v, "high": v, "mean": v}
    means = []
    for _ in range(iterations):
        sample = [rng.choice(values) for _ in values]
        means.append(_mean(sample))
    means.sort()
    lo_idx = int(round(iterations * 0.025))
    hi_idx = min(iterations - 1, int(round(iterations * 0.975)))
    return {
        "low": _round(means[lo_idx]),
        "high": _round(means[hi_idx]),
        "mean": _round(_mean(means)),
    }


def _bootstrap_delta_result(
    deltas: Sequence[float], *, rng: random.Random, iterations: int
) -> Dict[str, Any]:
    if not deltas:
        return _empty_effect()
    point = _mean(deltas)
    boot_means = []
    for _ in range(iterations):
        sample = [rng.choice(deltas) for _ in deltas]
        boot_means.append(_mean(sample))
    return _effect_from_bootstrap(point, boot_means)


def _effect_from_bootstrap(
    point: float, boot_values: Sequence[float]
) -> Dict[str, Any]:
    if not boot_values:
        return _empty_effect()
    sorted_vals = sorted(boot_values)
    n = len(sorted_vals)
    lo_idx = int(round(n * 0.025))
    hi_idx = min(n - 1, int(round(n * 0.975)))
    return {
        "point": _round(point),
        "ci": {
            "low": _round(sorted_vals[lo_idx]),
            "high": _round(sorted_vals[hi_idx]),
            "mean": _round(_mean(sorted_vals)),
        },
        "p_value": _bootstrap_two_sided_p(boot_values),
    }


def _bootstrap_two_sided_p(boot_values: Sequence[float]) -> float:
    """Two-sided bootstrap p: 2 * min(P(b<=0), P(b>=0)), clipped to [0, 1]."""
    if not boot_values:
        return 1.0
    n = len(boot_values)
    leq = sum(1 for v in boot_values if v <= 0)
    geq = sum(1 for v in boot_values if v >= 0)
    p = 2.0 * min(leq, geq) / n
    return _round(min(1.0, max(0.0, p)))


def _empty_effect() -> Dict[str, Any]:
    return {
        "point": None,
        "ci": {"low": 0.0, "high": 0.0, "mean": 0.0},
        "p_value": 1.0,
    }


def _ci_excludes_zero(ci: Dict[str, Any]) -> bool:
    low = float(ci.get("low", 0.0) or 0.0)
    high = float(ci.get("high", 0.0) or 0.0)
    return low > 0 or high < 0


def _clone_rng(rng: random.Random, salt: str) -> random.Random:
    """Derive a child RNG so multiple bootstraps stay reproducible but independent."""
    return random.Random(f"{rng.random():.16f}:{salt}")


def _derive_seed(seed: int, salt: str) -> int:
    return abs(hash((seed, salt))) % (2**31)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _round(value: float) -> float:
    return round(float(value), 4)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def analyze(
    rollouts: Sequence[Dict[str, Any]],
    classifications: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    strict_vea: bool = False,
    bootstrap_samples: int = _DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 0,
    min_cell_n: int = _DEFAULT_MIN_CELL_N,
    interaction_threshold: float = _DEFAULT_INTERACTION_THRESHOLD,
) -> Dict[str, Any]:
    has_vea = bool(classifications)
    records = (
        _join_vea(rollouts, classifications, strict_vea=strict_vea)
        if has_vea
        else list(rollouts)
    )
    per_model_records = _index_by_model(records)
    per_model: List[Dict[str, Any]] = []
    for model_name, model_records in sorted(per_model_records.items()):
        per_model.append(
            _analyze_model(
                model_name,
                model_records,
                seed=seed,
                bootstrap_samples=bootstrap_samples,
                min_cell_n=min_cell_n,
                interaction_threshold=interaction_threshold,
                has_vea=has_vea,
            )
        )
    pooled = _analyze_pooled(
        per_model,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
        min_cell_n=min_cell_n,
        interaction_threshold=interaction_threshold,
        has_vea=has_vea,
        all_records=records,
    )
    return {
        "vea_strictness": (
            "strict" if has_vea and strict_vea else ("broad" if has_vea else "none")
        ),
        "bootstrap_samples": bootstrap_samples,
        "min_cell_n": min_cell_n,
        "interaction_threshold": interaction_threshold,
        "control_condition": "neutral",
        "treatment_condition": "inject",
        "per_model": per_model,
        "pooled_across_models": pooled,
        "holm_bonferroni": _apply_holm_bonferroni(per_model),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing rollouts.jsonl (and optionally vea_classifications.jsonl).",
    )
    parser.add_argument(
        "--rollouts-path",
        type=Path,
        default=None,
        help="Override the rollouts file path (defaults to run-dir/rollouts.jsonl).",
    )
    parser.add_argument(
        "--classifications-path",
        type=Path,
        default=None,
        help="Override the VEA classifications path (default: run-dir/vea_classifications.jsonl, optional).",
    )
    parser.add_argument(
        "--strict-vea",
        action="store_true",
        help="Apply the validated strict VEA subtype filter when joining classifications.",
    )
    parser.add_argument(
        "--bootstrap-samples", type=int, default=_DEFAULT_BOOTSTRAP_SAMPLES
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-cell-n",
        type=int,
        default=_DEFAULT_MIN_CELL_N,
        help="Minimum rollouts per cell before producing a verdict.",
    )
    parser.add_argument(
        "--interaction-threshold",
        type=float,
        default=_DEFAULT_INTERACTION_THRESHOLD,
    )
    parser.add_argument(
        "--out-suffix",
        default="",
        help="Suffix added to output filenames, e.g. '.strict'.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_dir: Path = args.run_dir
    rollouts_path = args.rollouts_path or (run_dir / "rollouts.jsonl")
    classifications_path = args.classifications_path or (
        run_dir / "vea_classifications.jsonl"
    )
    if not rollouts_path.exists():
        LOGGER.error("rollouts file not found: %s", rollouts_path)
        return 1
    rollouts = _load_jsonl(rollouts_path)
    classifications: Optional[List[Dict[str, Any]]] = None
    if classifications_path.exists():
        classifications = _load_jsonl(classifications_path)
        LOGGER.info(
            "loaded %d rollouts + %d classifications",
            len(rollouts),
            len(classifications),
        )
    else:
        LOGGER.info(
            "loaded %d rollouts (no classifications at %s)",
            len(rollouts),
            classifications_path,
        )

    summary = analyze(
        rollouts,
        classifications,
        strict_vea=args.strict_vea,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        min_cell_n=args.min_cell_n,
        interaction_threshold=args.interaction_threshold,
    )

    suffix = args.out_suffix
    summary_path = run_dir / f"v2_apollo_factorial_summary{suffix}.json"
    report_path = run_dir / f"v2_apollo_factorial_report{suffix}.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    report_path.write_text(_render_report(summary), encoding="utf-8")
    LOGGER.info("wrote %s", summary_path)
    LOGGER.info("wrote %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
