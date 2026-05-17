"""Tests for the V2 opus deep-dive analysis."""

import json
from pathlib import Path

from scripts.analyze_v2_opus_deepdive import (
    _by_field,
    _filter_to_model,
    _per_category_ranking,
    _split_eval_deployment,
    _strata_paired_with_deployment,
    deepdive,
    main as deepdive_main,
)


def _rollout(
    *,
    model: str,
    condition: str,
    example_id: str,
    is_eval: bool,
    refusal: float,
    eval_type: str = "",
    category: str = "support_ticket",
    scaffold_type: str = "chat",
) -> dict:
    return {
        "model_name": model,
        "condition": condition,
        "example_id": example_id,
        "task_family": "apollo_eval_aware",
        "apollo_id": example_id.removeprefix("apollo:"),
        "apollo_is_eval": "true" if is_eval else "false",
        "apollo_eval_type": eval_type,
        "apollo_category": category,
        "apollo_subcategory": "",
        "apollo_safety_category": "",
        "apollo_scaffold_type": scaffold_type,
        "apollo_prompt_mode": "full_transcript",
        "rollout_index": 0,
        "prompt": "x",
        "system_prompt": "x",
        "response": "x",
        "reasoning_trace": "trace",
        "refusal": float(refusal),
    }


def _two_model_dataset(n_per_cell: int = 30) -> tuple:
    rollouts = []
    for model in ("anthropic:claude-opus-4-7", "openrouter:qwen/qwen3-235b"):
        for i in range(n_per_cell):
            eval_type = "capability" if i % 2 == 0 else "alignment"
            scaffold = "agentic" if i < n_per_cell // 2 else "chat"
            rollouts.append(_rollout(model=model, condition="neutral", example_id=f"apollo:e{i}",
                                      is_eval=True, refusal=0.2, eval_type=eval_type,
                                      category="mmlu", scaffold_type=scaffold))
            rollouts.append(_rollout(model=model, condition="inject", example_id=f"apollo:e{i}",
                                      is_eval=True, refusal=0.2, eval_type=eval_type,
                                      category="mmlu", scaffold_type=scaffold))
            rollouts.append(_rollout(model=model, condition="neutral", example_id=f"apollo:d{i}",
                                      is_eval=False, refusal=0.1, eval_type="",
                                      category="support_ticket", scaffold_type=scaffold))
            rollouts.append(_rollout(model=model, condition="inject", example_id=f"apollo:d{i}",
                                      is_eval=False, refusal=0.1, eval_type="",
                                      category="support_ticket", scaffold_type=scaffold))
    return rollouts, []  # no classifications for these tests


def test_filter_keeps_only_named_model():
    rollouts, _ = _two_model_dataset(n_per_cell=5)
    only_opus = _filter_to_model(rollouts, "anthropic:claude-opus-4-7")
    assert all(r["model_name"] == "anthropic:claude-opus-4-7" for r in only_opus)
    assert len(only_opus) == 5 * 4  # n_per_cell × 4 cells per model


def test_split_eval_deployment_partitions_by_label():
    rollouts, _ = _two_model_dataset(n_per_cell=5)
    eval_rollouts, deploy_rollouts = _split_eval_deployment(rollouts)
    assert {r["apollo_is_eval"] for r in eval_rollouts} == {"true"}
    assert {r["apollo_is_eval"] for r in deploy_rollouts} == {"false"}
    assert len(eval_rollouts) + len(deploy_rollouts) == len(rollouts)


def test_by_field_groups_by_value():
    rollouts, _ = _two_model_dataset(n_per_cell=5)
    by_type = _by_field(rollouts, "apollo_eval_type")
    assert set(by_type.keys()) == {"capability", "alignment", "(missing)"}


def test_eval_type_strata_pair_with_full_deployment():
    rollouts, _ = _two_model_dataset(n_per_cell=10)
    opus = _filter_to_model(rollouts, "anthropic:claude-opus-4-7")
    strata = _strata_paired_with_deployment(opus, "apollo_eval_type")
    assert set(strata.keys()) == {"capability", "alignment"}
    # Each stratum should include the model's full deployment set
    for stratum_rollouts in strata.values():
        deploy_count = sum(1 for r in stratum_rollouts if r["apollo_is_eval"] == "false")
        assert deploy_count == 20  # n_per_cell × 2 conditions per deployment


def test_deepdive_produces_expected_top_level_structure():
    rollouts, classifications = _two_model_dataset(n_per_cell=30)
    summary = deepdive(
        rollouts, classifications,
        model_filter="anthropic:claude-opus-4-7",
        strict_vea=False,
        bootstrap_samples=200,
        seed=0,
        min_cell_n=10,
        min_per_category=8,
    )
    assert summary["model_name"] == "anthropic:claude-opus-4-7"
    assert "by_eval_type" in summary
    assert "by_scaffold_type" in summary
    assert "per_category_ranking" in summary
    # Eval-type strata exist for both capability and alignment
    assert set(summary["by_eval_type"].keys()) == {"capability", "alignment"}


def test_deepdive_baseline_matches_unfiltered_analysis():
    """Baseline (no stratification) should match what analyze_v2_factorial would
    return for the same model on its own."""
    from scripts.analyze_v2_factorial import analyze
    rollouts, classifications = _two_model_dataset(n_per_cell=30)
    summary = deepdive(
        rollouts, classifications,
        model_filter="anthropic:claude-opus-4-7",
        strict_vea=False, bootstrap_samples=200, seed=0,
        min_cell_n=10, min_per_category=8,
    )
    opus_only = [r for r in rollouts if r["model_name"] == "anthropic:claude-opus-4-7"]
    reference = analyze(
        opus_only, None,
        strict_vea=False, bootstrap_samples=200, seed=0, min_cell_n=10,
    )
    ref_point = reference["per_model"][0]["interaction_effect"]["point"]
    assert summary["baseline"]["point"] == ref_point


def test_main_writes_outputs(tmp_path: Path):
    rollouts, _ = _two_model_dataset(n_per_cell=30)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "rollouts.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rollouts) + "\n",
        encoding="utf-8",
    )
    exit_code = deepdive_main(
        [
            "--run-dir", str(run_dir),
            "--model-filter", "anthropic:claude-opus-4-7",
            "--bootstrap-samples", "100",
            "--out-suffix", ".test",
        ]
    )
    assert exit_code == 0
    json_path = run_dir / "deepdive_anthropic_claude-opus-4-7.test.json"
    md_path = run_dir / "deepdive_anthropic_claude-opus-4-7.test.md"
    assert json_path.exists()
    assert md_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["model_name"] == "anthropic:claude-opus-4-7"
    assert "By `apollo_eval_type`" in md_path.read_text(encoding="utf-8")
