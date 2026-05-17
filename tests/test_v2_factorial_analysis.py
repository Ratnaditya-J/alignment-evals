"""Tests for the V2 Apollo factorial analysis script.

Uses fully synthetic rollouts so the bootstrap CIs are reproducible and the
designed effect signs are predictable.
"""

import json
from pathlib import Path

from scripts.analyze_v2_factorial import (
    _DEFAULT_BOOTSTRAP_SAMPLES,
    analyze,
    main as analyze_main,
)


# ---------------------------------------------------------------------------
# Synthetic rollouts helpers
# ---------------------------------------------------------------------------


def _rollout(
    model_name: str,
    condition: str,
    is_eval: bool,
    example_id: str,
    refusal: float,
    *,
    rollout_index: int = 0,
    apollo_id: str = "",
) -> dict:
    return {
        "model_name": model_name,
        "condition": condition,
        "example_id": example_id,
        "task_family": "apollo_eval_aware",
        "apollo_id": apollo_id or example_id.removeprefix("apollo:"),
        "apollo_is_eval": "true" if is_eval else "false",
        "apollo_eval_category": "Capability" if is_eval else "",
        "apollo_category": "mmlu" if is_eval else "support_ticket",
        "apollo_prompt_mode": "full_transcript",
        "rollout_index": rollout_index,
        "prompt": f"prompt for {example_id}",
        "system_prompt": "You are a helpful assistant.",
        "response": "ok",
        "reasoning_trace": "",
        "refusal": float(refusal),
    }


def _make_paired_set(
    model: str,
    *,
    n_per_cell: int,
    eval_neutral: float,
    eval_inject: float,
    deploy_neutral: float,
    deploy_inject: float,
) -> list:
    """Return rollouts with deterministic per-cell refusal rates.

    Each example_id appears in both conditions (paired by id within a label
    stratum). Refusal is set so that the per-cell mean exactly matches the
    requested rate (we vary the per-example value 0/1 in a tile pattern).
    """

    def _tile(rate: float, n: int) -> list:
        positives = int(round(rate * n))
        return [1.0] * positives + [0.0] * (n - positives)

    rollouts = []
    eval_values_neutral = _tile(eval_neutral, n_per_cell)
    eval_values_inject = _tile(eval_inject, n_per_cell)
    deploy_values_neutral = _tile(deploy_neutral, n_per_cell)
    deploy_values_inject = _tile(deploy_inject, n_per_cell)
    for i in range(n_per_cell):
        rollouts.append(_rollout(model, "neutral", True, f"apollo:e{i}", eval_values_neutral[i]))
        rollouts.append(_rollout(model, "inject", True, f"apollo:e{i}", eval_values_inject[i]))
        rollouts.append(_rollout(model, "neutral", False, f"apollo:d{i}", deploy_values_neutral[i]))
        rollouts.append(_rollout(model, "inject", False, f"apollo:d{i}", deploy_values_inject[i]))
    return rollouts


# ---------------------------------------------------------------------------
# Core analyses
# ---------------------------------------------------------------------------


def test_interaction_effect_zero_when_inject_has_no_effect():
    rollouts = _make_paired_set(
        "m",
        n_per_cell=50,
        eval_neutral=0.4, eval_inject=0.4,
        deploy_neutral=0.2, deploy_inject=0.2,
    )
    summary = analyze(rollouts, bootstrap_samples=400, seed=0)
    [model] = summary["per_model"]
    interaction = model["interaction_effect"]
    # Point estimate should be near zero; CI should contain zero.
    assert abs(float(interaction["point"])) < 0.05
    assert interaction["ci"]["low"] <= 0 <= interaction["ci"]["high"]


def test_interaction_effect_excludes_zero_when_inject_moves_only_eval():
    rollouts = _make_paired_set(
        "m",
        n_per_cell=80,
        eval_neutral=0.2, eval_inject=0.8,    # +0.6 under inject on eval cells
        deploy_neutral=0.2, deploy_inject=0.2, # no change on deployment cells
    )
    summary = analyze(rollouts, bootstrap_samples=400, seed=0)
    [model] = summary["per_model"]
    interaction = model["interaction_effect"]
    assert float(interaction["point"]) > 0.4
    # Strong designed effect — CI should clearly exclude zero
    assert interaction["ci"]["low"] > 0


def test_cell_means_match_simple_average():
    rollouts = _make_paired_set(
        "m",
        n_per_cell=40,
        eval_neutral=0.3, eval_inject=0.7,
        deploy_neutral=0.1, deploy_inject=0.5,
    )
    summary = analyze(rollouts, bootstrap_samples=200, seed=0)
    cells = summary["per_model"][0]["cells"]
    assert cells["neutral_eval"]["refusal_mean"] == 0.3
    assert cells["inject_eval"]["refusal_mean"] == 0.7
    assert cells["neutral_deployment"]["refusal_mean"] == 0.1
    assert cells["inject_deployment"]["refusal_mean"] == 0.5
    assert cells["neutral_eval"]["n"] == 40


def test_main_effect_inject_is_paired_delta():
    rollouts = _make_paired_set(
        "m",
        n_per_cell=50,
        eval_neutral=0.2, eval_inject=0.6,    # +0.4
        deploy_neutral=0.4, deploy_inject=0.6, # +0.2
    )
    summary = analyze(rollouts, bootstrap_samples=300, seed=0)
    main_inject = summary["per_model"][0]["main_effect_inject"]
    # Pooled paired delta = mean of per-example (inject - neutral). With our
    # tile pattern this equals the average of the two cell deltas (+0.3).
    assert abs(float(main_inject["point"]) - 0.3) < 0.05


def test_verdict_insufficient_data_when_cell_below_min_n():
    rollouts = _make_paired_set(
        "m",
        n_per_cell=10,
        eval_neutral=0.2, eval_inject=0.8,
        deploy_neutral=0.2, deploy_inject=0.2,
    )
    summary = analyze(rollouts, bootstrap_samples=200, seed=0, min_cell_n=30)
    assert summary["per_model"][0]["verdict"] == "insufficient_data"


def test_verdict_null_both_when_no_effect_anywhere():
    rollouts = _make_paired_set(
        "m",
        n_per_cell=60,
        eval_neutral=0.4, eval_inject=0.4,
        deploy_neutral=0.4, deploy_inject=0.4,
    )
    summary = analyze(rollouts, bootstrap_samples=300, seed=0)
    assert summary["per_model"][0]["verdict"] == "null_both"


def test_verdict_interaction_supported_when_designed():
    rollouts = _make_paired_set(
        "m",
        n_per_cell=80,
        eval_neutral=0.2, eval_inject=0.8,
        deploy_neutral=0.2, deploy_inject=0.2,
    )
    summary = analyze(rollouts, bootstrap_samples=400, seed=0)
    assert summary["per_model"][0]["verdict"] == "interaction_supported"


# ---------------------------------------------------------------------------
# Multi-model + correction
# ---------------------------------------------------------------------------


def test_per_model_breakdown_separates_models():
    rollouts_a = _make_paired_set(
        "model-a", n_per_cell=40,
        eval_neutral=0.2, eval_inject=0.8,
        deploy_neutral=0.2, deploy_inject=0.2,
    )
    rollouts_b = _make_paired_set(
        "model-b", n_per_cell=40,
        eval_neutral=0.5, eval_inject=0.5,
        deploy_neutral=0.5, deploy_inject=0.5,
    )
    summary = analyze(rollouts_a + rollouts_b, bootstrap_samples=200, seed=0)
    names = sorted(entry["model_name"] for entry in summary["per_model"])
    assert names == ["model-a", "model-b"]


def test_holm_bonferroni_correction_applied():
    rollouts = _make_paired_set(
        "model-a", n_per_cell=60,
        eval_neutral=0.2, eval_inject=0.8,
        deploy_neutral=0.2, deploy_inject=0.2,
    ) + _make_paired_set(
        "model-b", n_per_cell=60,
        eval_neutral=0.5, eval_inject=0.5,
        deploy_neutral=0.5, deploy_inject=0.5,
    )
    summary = analyze(rollouts, bootstrap_samples=400, seed=0)
    rows = summary["holm_bonferroni"]
    # 2 models x 3 effects = 6 rows
    assert len(rows) == 6
    # Each row carries the corrected decision
    for row in rows:
        assert "holm_threshold" in row
        assert "reject_at_0_05" in row
        assert row["effect"] in {
            "main_effect_inject",
            "main_effect_intrinsic",
            "interaction_effect",
        }


def test_pooled_aggregate_present_and_uses_pooled_label():
    rollouts = _make_paired_set(
        "m", n_per_cell=40,
        eval_neutral=0.2, eval_inject=0.8,
        deploy_neutral=0.2, deploy_inject=0.2,
    )
    summary = analyze(rollouts, bootstrap_samples=200, seed=0)
    pooled = summary["pooled_across_models"]
    assert pooled["model_name"] == "pooled_across_models"
    assert set(pooled["cells"].keys()) == {
        "neutral_eval",
        "inject_eval",
        "neutral_deployment",
        "inject_deployment",
    }


# ---------------------------------------------------------------------------
# VEA join + strictness
# ---------------------------------------------------------------------------


def test_vea_strictness_label_reflects_inputs():
    rollouts = _make_paired_set(
        "m", n_per_cell=40,
        eval_neutral=0.2, eval_inject=0.8,
        deploy_neutral=0.2, deploy_inject=0.2,
    )
    s_none = analyze(rollouts, bootstrap_samples=200, seed=0)
    assert s_none["vea_strictness"] == "none"
    classifications = [
        {
            "model_name": r["model_name"],
            "condition": r["condition"],
            "example_id": r["example_id"],
            "rollout_index": r["rollout_index"],
            "is_vea": True,
            "vea_type": "explicit_eval",
            "confidence": 1.0,
        }
        for r in rollouts
    ]
    s_broad = analyze(rollouts, classifications, bootstrap_samples=200, seed=0)
    assert s_broad["vea_strictness"] == "broad"
    s_strict = analyze(
        rollouts, classifications, strict_vea=True, bootstrap_samples=200, seed=0
    )
    assert s_strict["vea_strictness"] == "strict"


def test_vea_cells_added_when_classifications_present():
    rollouts = _make_paired_set(
        "m", n_per_cell=40,
        eval_neutral=0.2, eval_inject=0.8,
        deploy_neutral=0.2, deploy_inject=0.2,
    )
    classifications = []
    for r in rollouts:
        # Alternate is_vea so cells have meaningful means.
        classifications.append(
            {
                "model_name": r["model_name"],
                "condition": r["condition"],
                "example_id": r["example_id"],
                "rollout_index": r["rollout_index"],
                "is_vea": r["condition"] == "inject",
                "vea_type": "explicit_eval" if r["condition"] == "inject" else "none",
                "confidence": 0.9,
            }
        )
    summary = analyze(rollouts, classifications, bootstrap_samples=200, seed=0)
    cell = summary["per_model"][0]["cells"]["inject_eval"]
    assert cell["vea_mean"] == 1.0
    neutral_cell = summary["per_model"][0]["cells"]["neutral_eval"]
    assert neutral_cell["vea_mean"] == 0.0


def test_rollouts_without_vea_classifications_are_preserved_not_dropped():
    """Closed-CoT backfill models intentionally have no VEA classifications
    (per the closed-CoT backfill pre-reg §4). The analyzer must NOT drop
    their rollouts during the VEA join -- behavioral data (refusal) is
    the primary outcome and must survive the join. Regression for the
    silent-drop bug where backfill models disappeared from per_model."""
    main_rollouts = _make_paired_set(
        "model_with_vea", n_per_cell=40,
        eval_neutral=0.2, eval_inject=0.4,
        deploy_neutral=0.1, deploy_inject=0.1,
    )
    backfill_rollouts = _make_paired_set(
        "closed_cot_backfill_model", n_per_cell=40,
        eval_neutral=0.5, eval_inject=0.5,
        deploy_neutral=0.1, deploy_inject=0.1,
    )
    classifications = []
    for r in main_rollouts:
        classifications.append(
            {
                "model_name": r["model_name"],
                "condition": r["condition"],
                "example_id": r["example_id"],
                "rollout_index": r["rollout_index"],
                "is_vea": r["condition"] == "inject",
                "vea_type": "explicit_eval" if r["condition"] == "inject" else "none",
            }
        )
    # NOTE: no classifications for backfill_rollouts -- simulating closed-CoT
    summary = analyze(
        main_rollouts + backfill_rollouts,
        classifications,
        bootstrap_samples=200,
        seed=0,
    )
    model_names = {pm["model_name"] for pm in summary["per_model"]}
    assert model_names == {"model_with_vea", "closed_cot_backfill_model"}, (
        "backfill model dropped during VEA join (regression)"
    )
    backfill_entry = next(
        pm for pm in summary["per_model"] if pm["model_name"] == "closed_cot_backfill_model"
    )
    assert backfill_entry["n_rollouts"] == 160, "backfill rollouts incomplete"
    # All 4 cells must be populated with refusal data for the backfill model
    for cell_key in ("neutral_eval", "inject_eval", "neutral_deployment", "inject_deployment"):
        cell = backfill_entry["cells"][cell_key]
        assert cell["n"] == 40, f"backfill cell {cell_key} has n={cell['n']}, expected 40"
        assert "refusal_mean" in cell
        # VEA cell mean should be 0 for the backfill model (no classifications)
        assert cell["vea_mean"] == 0.0


def test_strict_vea_drops_low_precision_subtypes():
    rollouts = _make_paired_set(
        "m", n_per_cell=40,
        eval_neutral=0.2, eval_inject=0.8,
        deploy_neutral=0.2, deploy_inject=0.2,
    )
    classifications = []
    for r in rollouts:
        subtype = "explicit_eval" if r["condition"] == "inject" else "artificial_scenario"
        classifications.append(
            {
                "model_name": r["model_name"],
                "condition": r["condition"],
                "example_id": r["example_id"],
                "rollout_index": r["rollout_index"],
                "is_vea": True,
                "vea_type": subtype,
                "confidence": 0.9,
            }
        )
    broad = analyze(rollouts, classifications, bootstrap_samples=200, seed=0)
    strict = analyze(
        rollouts, classifications, strict_vea=True, bootstrap_samples=200, seed=0
    )
    # Broad: every record is_vea=True -> 1.0 everywhere.
    assert broad["per_model"][0]["cells"]["neutral_eval"]["vea_mean"] == 1.0
    # Strict: artificial_scenario is dropped, so neutral_eval drops to 0.
    assert strict["per_model"][0]["cells"]["neutral_eval"]["vea_mean"] == 0.0
    # And inject_eval (explicit_eval subtype) stays at 1.0.
    assert strict["per_model"][0]["cells"]["inject_eval"]["vea_mean"] == 1.0


# ---------------------------------------------------------------------------
# CLI end-to-end
# ---------------------------------------------------------------------------


def test_cli_writes_summary_and_report_files(tmp_path: Path):
    rollouts = _make_paired_set(
        "m", n_per_cell=40,
        eval_neutral=0.2, eval_inject=0.8,
        deploy_neutral=0.2, deploy_inject=0.2,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rollouts_path = run_dir / "rollouts.jsonl"
    rollouts_path.write_text(
        "\n".join(json.dumps(r) for r in rollouts) + "\n", encoding="utf-8"
    )
    exit_code = analyze_main(
        [
            "--run-dir", str(run_dir),
            "--bootstrap-samples", "200",
            "--out-suffix", ".test",
        ]
    )
    assert exit_code == 0
    summary_path = run_dir / "v2_apollo_factorial_summary.test.json"
    report_path = run_dir / "v2_apollo_factorial_report.test.md"
    assert summary_path.exists()
    assert report_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["per_model"][0]["model_name"] == "m"
    report = report_path.read_text(encoding="utf-8")
    assert "V2 Apollo factorial" in report
    assert "interaction" in report.lower()


def test_analyze_handles_empty_rollouts_gracefully():
    summary = analyze([], bootstrap_samples=200, seed=0)
    assert summary["per_model"] == []
    pooled = summary["pooled_across_models"]
    assert pooled["verdict"] == "insufficient_data"


def test_default_bootstrap_samples_is_two_thousand():
    assert _DEFAULT_BOOTSTRAP_SAMPLES == 2000
