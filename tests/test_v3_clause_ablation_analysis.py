"""Unit tests for the V3 inject-clause ablation analyzer."""

import json
import random

import pytest

from scripts.analyze_v3_clause_ablation import (
    _SINGLETONS,
    _additivity_residual_effect,
    _apply_holm_bonferroni,
    _bootstrap_mean_ci,
    _ci_excludes_zero,
    _ci_includes_zero,
    _ci_overlap,
    _classify_attribution,
    _cross_model_direction,
    _paired_delta_effect,
    _paired_refusals_by_example,
    _singleton_sum_point,
    analyze,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _rollout(
    model: str, condition: str, example_id: str, refusal: float
) -> dict:
    return {
        "model_name": model,
        "condition": condition,
        "example_id": example_id,
        "refusal": float(refusal),
        "apollo_is_eval": "true",
        "apollo_eval_type": "alignment",
    }


def _make_rollouts(
    model: str, n: int, condition_to_refusal: dict
) -> list:
    rollouts = []
    for i in range(n):
        for cond, refusal in condition_to_refusal.items():
            rollouts.append(_rollout(model, cond, f"ex{i}", refusal))
    return rollouts


# ---------------------------------------------------------------------------
# CI helpers
# ---------------------------------------------------------------------------


def test_ci_overlap_detects_disjoint_and_overlapping():
    assert _ci_overlap({"low": 0.1, "high": 0.3}, {"low": 0.2, "high": 0.5})
    assert _ci_overlap({"low": -0.1, "high": 0.1}, {"low": 0.0, "high": 0.05})
    assert not _ci_overlap({"low": 0.1, "high": 0.3}, {"low": 0.5, "high": 0.9})


def test_ci_excludes_zero():
    assert _ci_excludes_zero({"low": 0.1, "high": 0.3})
    assert _ci_excludes_zero({"low": -0.5, "high": -0.1})
    assert not _ci_excludes_zero({"low": -0.1, "high": 0.1})
    assert not _ci_excludes_zero({"low": 0.0, "high": 0.0})


def test_ci_includes_zero_is_complement():
    assert _ci_includes_zero({"low": -0.1, "high": 0.1})
    assert _ci_includes_zero({"low": 0.0, "high": 0.0})
    assert not _ci_includes_zero({"low": 0.2, "high": 0.4})


# ---------------------------------------------------------------------------
# Paired refusal pivot
# ---------------------------------------------------------------------------


def test_paired_refusals_pivot_groups_by_example():
    rollouts = _make_rollouts(
        "m",
        n=3,
        condition_to_refusal={
            "neutral": 0.0,
            "eval_only": 1.0,
            "reviewer_only": 0.0,
            "comparison_only": 0.0,
            "full_inject": 1.0,
        },
    )
    paired = _paired_refusals_by_example(rollouts)
    assert len(paired) == 3
    for cond_map in paired.values():
        assert cond_map == {
            "neutral": 0.0,
            "eval_only": 1.0,
            "reviewer_only": 0.0,
            "comparison_only": 0.0,
            "full_inject": 1.0,
        }


# ---------------------------------------------------------------------------
# Effect estimators
# ---------------------------------------------------------------------------


def test_paired_delta_effect_zero_when_no_change():
    rollouts = _make_rollouts(
        "m",
        n=20,
        condition_to_refusal={"neutral": 0.5, "eval_only": 0.5},
    )
    paired = _paired_refusals_by_example(rollouts)
    eff = _paired_delta_effect(
        paired,
        treatment="eval_only",
        rng=random.Random(0),
        iterations=200,
    )
    assert eff["point"] == 0.0
    assert eff["ci"]["low"] == 0.0
    assert eff["ci"]["high"] == 0.0


def test_paired_delta_effect_nonzero_and_excludes_zero():
    rollouts = _make_rollouts(
        "m",
        n=20,
        condition_to_refusal={"neutral": 0.0, "eval_only": 1.0},
    )
    paired = _paired_refusals_by_example(rollouts)
    eff = _paired_delta_effect(
        paired,
        treatment="eval_only",
        rng=random.Random(0),
        iterations=200,
    )
    assert eff["point"] == 1.0
    assert _ci_excludes_zero(eff["ci"])


def test_additivity_residual_is_zero_when_singletons_sum_to_full():
    rollouts = _make_rollouts(
        "m",
        n=20,
        condition_to_refusal={
            "neutral": 0.0,
            "eval_only": 0.3,
            "reviewer_only": 0.3,
            "comparison_only": 0.3,
            "full_inject": 0.9,
        },
    )
    paired = _paired_refusals_by_example(rollouts)
    res = _additivity_residual_effect(
        paired, rng=random.Random(0), iterations=200
    )
    assert abs(res["point"]) < 1e-9
    assert _ci_includes_zero(res["ci"])


def test_additivity_residual_is_positive_when_synergistic():
    rollouts = _make_rollouts(
        "m",
        n=20,
        condition_to_refusal={
            "neutral": 0.0,
            "eval_only": 0.1,
            "reviewer_only": 0.1,
            "comparison_only": 0.1,
            "full_inject": 0.9,
        },
    )
    paired = _paired_refusals_by_example(rollouts)
    res = _additivity_residual_effect(
        paired, rng=random.Random(0), iterations=200
    )
    # full=0.9, sum=0.3 -> residual=0.6
    assert res["point"] == pytest.approx(0.6, abs=1e-6)
    assert _ci_excludes_zero(res["ci"])


def test_singleton_sum_point():
    rollouts = _make_rollouts(
        "m",
        n=10,
        condition_to_refusal={
            "neutral": 0.0,
            "eval_only": 0.3,
            "reviewer_only": 0.3,
            "comparison_only": 0.3,
            "full_inject": 0.9,
        },
    )
    paired = _paired_refusals_by_example(rollouts)
    assert _singleton_sum_point(paired) == pytest.approx(0.9, abs=1e-6)


# ---------------------------------------------------------------------------
# Attribution decision tree
# ---------------------------------------------------------------------------


def _analyze_one_model(rollouts: list, *, min_cell_n: int = 10) -> dict:
    summary = analyze(
        rollouts, bootstrap_samples=200, seed=0, min_cell_n=min_cell_n
    )
    assert len(summary["per_model"]) == 1
    return summary["per_model"][0]


def test_attribution_single_clause_when_only_one_singleton_active():
    rollouts = _make_rollouts(
        "m",
        n=15,
        condition_to_refusal={
            "neutral": 0.0,
            "eval_only": 1.0,
            "reviewer_only": 0.0,
            "comparison_only": 0.0,
            "full_inject": 1.0,
        },
    )
    entry = _analyze_one_model(rollouts)
    assert entry["attribution"] == "single_clause:eval_only"


def test_attribution_additive_when_singletons_sum_to_full():
    rollouts = _make_rollouts(
        "m",
        n=15,
        condition_to_refusal={
            "neutral": 0.0,
            "eval_only": 0.3,
            "reviewer_only": 0.3,
            "comparison_only": 0.3,
            "full_inject": 0.9,
        },
    )
    entry = _analyze_one_model(rollouts)
    assert entry["attribution"] == "additive"


def test_attribution_synergistic_when_full_exceeds_sum():
    rollouts = _make_rollouts(
        "m",
        n=15,
        condition_to_refusal={
            "neutral": 0.0,
            "eval_only": 0.1,
            "reviewer_only": 0.1,
            "comparison_only": 0.1,
            "full_inject": 0.9,
        },
    )
    entry = _analyze_one_model(rollouts)
    assert entry["attribution"] == "synergistic"


def test_attribution_synergistic_when_all_singletons_null_but_full_nonzero():
    rollouts = _make_rollouts(
        "m",
        n=15,
        condition_to_refusal={
            "neutral": 0.0,
            "eval_only": 0.0,
            "reviewer_only": 0.0,
            "comparison_only": 0.0,
            "full_inject": 0.9,
        },
    )
    entry = _analyze_one_model(rollouts)
    assert entry["attribution"] == "synergistic"


def test_attribution_null_at_full_when_full_effect_includes_zero():
    rollouts = _make_rollouts(
        "m",
        n=15,
        condition_to_refusal={
            "neutral": 0.5,
            "eval_only": 0.5,
            "reviewer_only": 0.5,
            "comparison_only": 0.5,
            "full_inject": 0.5,
        },
    )
    entry = _analyze_one_model(rollouts)
    assert entry["attribution"] == "null_at_full"


def test_attribution_unclassified_when_full_less_than_sum_subadditive():
    rollouts = _make_rollouts(
        "m",
        n=15,
        condition_to_refusal={
            "neutral": 0.0,
            "eval_only": 0.5,
            "reviewer_only": 0.5,
            "comparison_only": 0.5,
            "full_inject": 0.1,
        },
    )
    entry = _analyze_one_model(rollouts)
    # |full|=0.1 < |sum|=1.5; full CI excludes zero; not additive; not synergistic.
    assert entry["attribution"] == "unclassified"


def test_attribution_insufficient_data_when_any_cell_undersized():
    rollouts = _make_rollouts(
        "m",
        n=5,  # below min_cell_n
        condition_to_refusal={
            "neutral": 0.0,
            "eval_only": 0.5,
            "reviewer_only": 0.5,
            "comparison_only": 0.5,
            "full_inject": 0.9,
        },
    )
    summary = analyze(rollouts, bootstrap_samples=200, seed=0, min_cell_n=10)
    assert summary["per_model"][0]["attribution"] == "insufficient_data"


def test_attribution_uses_direct_classify_function():
    """The decision tree should be testable in isolation via _classify_attribution."""
    cells = {c: {"n": 100} for c in ("neutral", "eval_only", "reviewer_only", "comparison_only", "full_inject")}
    singleton_effects = {
        "eval_only": {
            "point": 1.0,
            "ci": {"low": 0.9, "high": 1.1},
            "p_value": 0.001,
        },
        "reviewer_only": {
            "point": 0.0,
            "ci": {"low": -0.05, "high": 0.05},
            "p_value": 1.0,
        },
        "comparison_only": {
            "point": 0.0,
            "ci": {"low": -0.05, "high": 0.05},
            "p_value": 1.0,
        },
    }
    full_effect = {
        "point": 1.0,
        "ci": {"low": 0.9, "high": 1.1},
        "p_value": 0.001,
    }
    residual = {
        "point": 0.0,
        "ci": {"low": -0.01, "high": 0.01},
        "p_value": 1.0,
    }
    verdict, note = _classify_attribution(
        cells, singleton_effects, full_effect, residual, 1.0, min_cell_n=10
    )
    assert verdict == "single_clause:eval_only"
    assert "overlaps" in note


# ---------------------------------------------------------------------------
# Holm-Bonferroni and cross-model
# ---------------------------------------------------------------------------


def test_holm_bonferroni_covers_exactly_6_hypotheses_for_2_models():
    rollouts_a = _make_rollouts(
        "model_a",
        n=15,
        condition_to_refusal={
            "neutral": 0.0,
            "eval_only": 0.5,
            "reviewer_only": 0.0,
            "comparison_only": 0.0,
            "full_inject": 0.5,
        },
    )
    rollouts_b = _make_rollouts(
        "model_b",
        n=15,
        condition_to_refusal={
            "neutral": 0.5,
            "eval_only": 0.0,
            "reviewer_only": 0.0,
            "comparison_only": 0.0,
            "full_inject": 0.0,
        },
    )
    summary = analyze(rollouts_a + rollouts_b, bootstrap_samples=200, seed=0, min_cell_n=10)
    holm = summary["holm_bonferroni"]
    assert len(holm) == 6
    effects = {(r["model_name"], r["effect"]) for r in holm}
    expected = {
        (m, f"singleton:{c}")
        for m in ("model_a", "model_b")
        for c in _SINGLETONS
    }
    assert effects == expected


def test_holm_bonferroni_excludes_full_inject():
    """Per V3 design §4.4, the full_inject effect is the V1-replication anchor
    and is NOT part of the multiple-comparisons correction family."""
    rollouts = _make_rollouts(
        "m",
        n=15,
        condition_to_refusal={
            "neutral": 0.0,
            "eval_only": 0.3,
            "reviewer_only": 0.3,
            "comparison_only": 0.3,
            "full_inject": 0.9,
        },
    )
    summary = analyze(rollouts, bootstrap_samples=200, seed=0, min_cell_n=10)
    holm = summary["holm_bonferroni"]
    assert all("full_inject" not in r["effect"] for r in holm)


def test_cross_model_direction_flags_asymmetry():
    """If model_a refuses + under eval_only and model_b refuses − under eval_only,
    the cross_model row for eval_only should be marked asymmetric."""
    rollouts_a = _make_rollouts(
        "model_a",
        n=20,
        condition_to_refusal={
            "neutral": 0.0,
            "eval_only": 1.0,
            "reviewer_only": 0.5,
            "comparison_only": 0.5,
            "full_inject": 1.0,
        },
    )
    rollouts_b = _make_rollouts(
        "model_b",
        n=20,
        condition_to_refusal={
            "neutral": 1.0,
            "eval_only": 0.0,
            "reviewer_only": 0.5,
            "comparison_only": 0.5,
            "full_inject": 0.0,
        },
    )
    summary = analyze(rollouts_a + rollouts_b, bootstrap_samples=200, seed=0, min_cell_n=10)
    cross = {row["clause"]: row for row in summary["cross_model_direction_comparison"]}
    assert cross["eval_only"]["asymmetric"]
    assert cross["eval_only"]["models"]["model_a"] == "+"
    assert cross["eval_only"]["models"]["model_b"] == "-"


def test_cross_model_direction_not_asymmetric_when_same_sign():
    rollouts_a = _make_rollouts(
        "model_a",
        n=15,
        condition_to_refusal={
            "neutral": 0.0,
            "eval_only": 1.0,
            "reviewer_only": 0.0,
            "comparison_only": 0.0,
            "full_inject": 1.0,
        },
    )
    rollouts_b = _make_rollouts(
        "model_b",
        n=15,
        condition_to_refusal={
            "neutral": 0.0,
            "eval_only": 1.0,
            "reviewer_only": 0.0,
            "comparison_only": 0.0,
            "full_inject": 1.0,
        },
    )
    summary = analyze(rollouts_a + rollouts_b, bootstrap_samples=200, seed=0, min_cell_n=10)
    cross = {row["clause"]: row for row in summary["cross_model_direction_comparison"]}
    assert not cross["eval_only"]["asymmetric"]


# ---------------------------------------------------------------------------
# Summary structure
# ---------------------------------------------------------------------------


def test_summary_has_expected_top_level_keys():
    rollouts = _make_rollouts(
        "m",
        n=12,
        condition_to_refusal={
            "neutral": 0.0,
            "eval_only": 0.3,
            "reviewer_only": 0.3,
            "comparison_only": 0.3,
            "full_inject": 0.9,
        },
    )
    summary = analyze(rollouts, bootstrap_samples=200, seed=0, min_cell_n=10)
    for key in (
        "vea_strictness",
        "bootstrap_samples",
        "min_cell_n",
        "control_condition",
        "anchor_condition",
        "singleton_clauses",
        "per_model",
        "holm_bonferroni",
        "cross_model_direction_comparison",
    ):
        assert key in summary, f"missing top-level key {key!r}"
    assert summary["control_condition"] == "neutral"
    assert summary["anchor_condition"] == "full_inject"
    assert summary["singleton_clauses"] == list(_SINGLETONS)


def test_per_model_entry_has_all_required_fields():
    rollouts = _make_rollouts(
        "m",
        n=12,
        condition_to_refusal={
            "neutral": 0.0,
            "eval_only": 0.3,
            "reviewer_only": 0.3,
            "comparison_only": 0.3,
            "full_inject": 0.9,
        },
    )
    entry = _analyze_one_model(rollouts)
    for key in (
        "model_name",
        "n_rollouts",
        "cells",
        "singleton_effects",
        "full_inject_effect",
        "additivity_residual",
        "singleton_sum_point",
        "attribution",
        "attribution_note",
    ):
        assert key in entry, f"missing per-model field {key!r}"
    assert set(entry["cells"].keys()) == {
        "neutral", "eval_only", "reviewer_only", "comparison_only", "full_inject"
    }
    assert set(entry["singleton_effects"].keys()) == set(_SINGLETONS)


def test_summary_json_serializable():
    rollouts = _make_rollouts(
        "m",
        n=12,
        condition_to_refusal={
            "neutral": 0.0,
            "eval_only": 0.3,
            "reviewer_only": 0.3,
            "comparison_only": 0.3,
            "full_inject": 0.9,
        },
    )
    summary = analyze(rollouts, bootstrap_samples=200, seed=0, min_cell_n=10)
    # Round-trips without raising
    json.dumps(summary, sort_keys=True)


# ---------------------------------------------------------------------------
# VEA join
# ---------------------------------------------------------------------------


def test_vea_join_strict_filter_collapses_non_strict_subtypes():
    rollouts = _make_rollouts(
        "m",
        n=10,
        condition_to_refusal={
            "neutral": 0.0,
            "eval_only": 0.3,
            "reviewer_only": 0.3,
            "comparison_only": 0.3,
            "full_inject": 0.9,
        },
    )
    classifications = []
    for r in rollouts:
        classifications.append(
            {
                "model_name": r["model_name"],
                "condition": r["condition"],
                "example_id": r["example_id"],
                "rollout_index": 0,
                "is_vea": True,
                # Non-strict subtype: should be filtered to False under strict.
                "vea_type": "deployment_aware",
            }
        )
    strict_summary = analyze(
        rollouts, classifications, strict_vea=True, bootstrap_samples=200, seed=0, min_cell_n=10
    )
    broad_summary = analyze(
        rollouts, classifications, strict_vea=False, bootstrap_samples=200, seed=0, min_cell_n=10
    )
    strict_neutral = strict_summary["per_model"][0]["cells"]["neutral"]
    broad_neutral = broad_summary["per_model"][0]["cells"]["neutral"]
    assert strict_neutral["vea_mean"] == 0.0  # strict filter rejects deployment_aware
    assert broad_neutral["vea_mean"] == 1.0


def test_rollouts_without_vea_classifications_are_preserved_not_dropped():
    """Closed-CoT backfill models intentionally have no VEA classifications
    (per the closed-CoT backfill pre-reg §4). The analyzer must NOT drop
    their rollouts during the VEA join -- behavioral data (refusal +
    attribution) is the primary outcome and must survive the join.
    Regression for the silent-drop bug where backfill models disappeared
    from per_model."""
    main_rollouts = _make_rollouts(
        "model_with_vea",
        n=15,
        condition_to_refusal={
            "neutral": 0.0,
            "eval_only": 0.3,
            "reviewer_only": 0.3,
            "comparison_only": 0.3,
            "full_inject": 0.9,
        },
    )
    backfill_rollouts = _make_rollouts(
        "closed_cot_backfill_model",
        n=15,
        condition_to_refusal={
            "neutral": 0.5,
            "eval_only": 0.5,
            "reviewer_only": 0.5,
            "comparison_only": 0.5,
            "full_inject": 0.5,
        },
    )
    classifications = []
    for r in main_rollouts:
        classifications.append(
            {
                "model_name": r["model_name"],
                "condition": r["condition"],
                "example_id": r["example_id"],
                "rollout_index": 0,
                "is_vea": False,
                "vea_type": None,
            }
        )
    # NOTE: no classifications for backfill_rollouts -- simulating closed-CoT
    summary = analyze(
        main_rollouts + backfill_rollouts,
        classifications,
        bootstrap_samples=200,
        seed=0,
        min_cell_n=10,
    )
    model_names = {pm["model_name"] for pm in summary["per_model"]}
    assert model_names == {"model_with_vea", "closed_cot_backfill_model"}, (
        "backfill model dropped during VEA join (regression)"
    )
    backfill_entry = next(
        pm for pm in summary["per_model"]
        if pm["model_name"] == "closed_cot_backfill_model"
    )
    # Attribution still produced for backfill model (refusal data preserved)
    assert backfill_entry["attribution"] == "null_at_full"  # constant 0.5 across cells
    # All 5 cells populated with refusal data
    for cond in ("neutral", "eval_only", "reviewer_only", "comparison_only", "full_inject"):
        cell = backfill_entry["cells"][cond]
        assert cell["n"] == 15, f"backfill cell {cond} has n={cell['n']}, expected 15"
        assert cell["vea_mean"] == 0.0  # no VEA data for backfill


def test_bootstrap_mean_ci_single_value_collapses():
    ci = _bootstrap_mean_ci([0.7], rng=random.Random(0), iterations=200)
    assert ci == {"low": 0.7, "high": 0.7, "mean": 0.7}


# ---------------------------------------------------------------------------
# Direct unit on helpers
# ---------------------------------------------------------------------------


def test_apply_holm_returns_empty_for_no_models():
    assert _apply_holm_bonferroni([]) == []


def test_cross_model_returns_one_row_per_clause_plus_full():
    rollouts = _make_rollouts(
        "m",
        n=10,
        condition_to_refusal={
            "neutral": 0.0,
            "eval_only": 0.3,
            "reviewer_only": 0.3,
            "comparison_only": 0.3,
            "full_inject": 0.9,
        },
    )
    summary = analyze(rollouts, bootstrap_samples=200, seed=0, min_cell_n=10)
    rows = summary["cross_model_direction_comparison"]
    assert len(rows) == 4  # 3 singletons + full_inject
    assert {r["clause"] for r in rows} == set(_SINGLETONS) | {"full_inject"}
