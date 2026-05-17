"""Integration tests for the V2 Apollo 2x2 factorial runner.

Uses ``ScriptedModelClient`` so the test runs without network access or API
keys. The point is to verify the runner's task plumbing and the rollouts.jsonl
output shape, not to measure model behavior.
"""

import json
from pathlib import Path

import pytest

from scripts.run_v2_apollo_factorial import (
    CONDITIONS,
    _build_condition_transcript,
    _build_model_lineup,
    load_apollo_examples,
    run_factorial,
    write_rollouts_jsonl,
)
from scripts.run_goodfire_vea import _AnthropicWithThinking
from src.eval_awareness import TranscriptInput
from src.eval_awareness.dataset import EvalAwarenessExample
from src.eval_awareness.runner import ScriptedModelClient


def _apollo_dataset(tmp_path: Path, *, n_eval: int, n_deployment: int) -> Path:
    entries = []
    for i in range(n_eval):
        entries.append(
            {
                "id": f"e{i}",
                "input": [
                    {"role": "user", "content": f"eval prompt {i}: 17 * 23"}
                ],
                "metadata": {"category": "mmlu", "eval_category": "Capability"},
            }
        )
    for i in range(n_deployment):
        entries.append(
            {
                "id": f"d{i}",
                "input": [
                    {"role": "user", "content": f"deployment prompt {i}: production user"}
                ],
                "metadata": {"category": "support_ticket", "eval_category": None},
            }
        )
    path = tmp_path / "apollo.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def _model(name: str) -> ScriptedModelClient:
    return ScriptedModelClient(name=name)


def test_runner_produces_full_cross_product_of_rollouts(tmp_path):
    dataset = _apollo_dataset(tmp_path, n_eval=5, n_deployment=5)
    examples = load_apollo_examples(
        dataset,
        n_eval=5,
        n_deployment=5,
        prompt_mode="full_transcript",
        seed=0,
    )
    models = [_model("scripted-a"), _model("scripted-b")]
    rollouts = run_factorial(models, examples, max_concurrent=1, seed=0)
    # 2 models x 10 examples x 2 conditions = 40 rollouts
    assert len(rollouts) == 2 * 10 * 2
    assert {r["condition"] for r in rollouts} == set(CONDITIONS)
    assert {r["model_name"] for r in rollouts} == {"scripted-a", "scripted-b"}


def test_each_example_appears_in_both_conditions_for_each_model(tmp_path):
    dataset = _apollo_dataset(tmp_path, n_eval=3, n_deployment=3)
    examples = load_apollo_examples(
        dataset, n_eval=3, n_deployment=3, prompt_mode="full_transcript", seed=0
    )
    models = [_model("scripted-a")]
    rollouts = run_factorial(models, examples, max_concurrent=1, seed=0)
    by_example = {}
    for r in rollouts:
        by_example.setdefault(r["example_id"], set()).add(r["condition"])
    assert len(by_example) == 6
    for conditions in by_example.values():
        assert conditions == set(CONDITIONS)


def test_apollo_metadata_is_preserved_on_every_rollout(tmp_path):
    dataset = _apollo_dataset(tmp_path, n_eval=2, n_deployment=2)
    examples = load_apollo_examples(
        dataset, n_eval=2, n_deployment=2, prompt_mode="full_transcript", seed=0
    )
    rollouts = run_factorial([_model("m")], examples, max_concurrent=1, seed=0)
    required_keys = {
        "apollo_id",
        "apollo_is_eval",
        "apollo_eval_type",
        "apollo_category",
        "apollo_subcategory",
        "apollo_safety_category",
        "apollo_scaffold_type",
        "apollo_prompt_mode",
    }
    for r in rollouts:
        assert required_keys <= set(r.keys())
        assert r["apollo_prompt_mode"] == "full_transcript"
        assert r["apollo_is_eval"] in {"true", "false"}
    # Each label should appear for at least one rollout
    labels = {r["apollo_is_eval"] for r in rollouts}
    assert labels == {"true", "false"}


def test_inject_condition_uses_inject_framing_neutral_uses_neutral(tmp_path):
    dataset = _apollo_dataset(tmp_path, n_eval=1, n_deployment=1)
    examples = load_apollo_examples(
        dataset, n_eval=1, n_deployment=1, prompt_mode="full_transcript", seed=0
    )
    rollouts = run_factorial([_model("m")], examples, max_concurrent=1, seed=0)
    for r in rollouts:
        if r["condition"] == "inject":
            assert "evaluation" in r["system_prompt"].lower()
            assert "safety researchers" in r["system_prompt"].lower()
        else:
            assert r["system_prompt"] == "You are a helpful assistant."


def test_condition_framing_is_prepended_when_apollo_has_system_message():
    example = EvalAwarenessExample(
        example_id="apollo:x",
        label="hidden_eval",
        task_family="apollo_eval_aware",
        source="mmlu",
        transcript=TranscriptInput(
            user_prompt="What is 17*23?",
            system_prompt="Original Apollo system prompt.",
        ),
        metadata={"apollo_is_eval": "true"},
    )
    inject_transcript = _build_condition_transcript(example, "inject")
    # framing comes first, original Apollo system second
    lines = inject_transcript.system_prompt.split("\n\n")
    assert "evaluation" in lines[0].lower()
    assert lines[-1] == "Original Apollo system prompt."


def test_condition_framing_alone_when_apollo_has_no_system_message():
    example = EvalAwarenessExample(
        example_id="apollo:y",
        label="deployment_like",
        task_family="apollo_eval_aware",
        source="support_ticket",
        transcript=TranscriptInput(user_prompt="Production user prompt."),
        metadata={"apollo_is_eval": "false"},
    )
    inject_transcript = _build_condition_transcript(example, "inject")
    neutral_transcript = _build_condition_transcript(example, "neutral")
    assert "\n\n" not in inject_transcript.system_prompt
    assert neutral_transcript.system_prompt == "You are a helpful assistant."


def test_full_transcript_mode_preserves_conversation_history():
    example = EvalAwarenessExample(
        example_id="apollo:multi",
        label="hidden_eval",
        task_family="apollo_eval_aware",
        source="mmlu",
        transcript=TranscriptInput(
            user_prompt="Follow-up question?",
            conversation_history=[
                {"role": "user", "content": "First question?"},
                {"role": "assistant", "content": "First answer."},
            ],
        ),
        metadata={"apollo_is_eval": "true"},
    )
    transcript = _build_condition_transcript(example, "neutral")
    assert transcript.user_prompt == "Follow-up question?"
    assert transcript.conversation_history == [
        {"role": "user", "content": "First question?"},
        {"role": "assistant", "content": "First answer."},
    ]


def test_rollouts_jsonl_written_and_round_trips(tmp_path):
    dataset = _apollo_dataset(tmp_path, n_eval=2, n_deployment=2)
    examples = load_apollo_examples(
        dataset, n_eval=2, n_deployment=2, prompt_mode="full_transcript", seed=0
    )
    rollouts = run_factorial([_model("m")], examples, max_concurrent=1, seed=0)
    out = tmp_path / "out" / "rollouts.jsonl"
    write_rollouts_jsonl(rollouts, out)
    assert out.exists()
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == len(rollouts)
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["example_id"].startswith("apollo:")
    assert "refusal" in parsed[0]


def test_concurrent_runner_produces_same_rollouts_as_serial(tmp_path):
    dataset = _apollo_dataset(tmp_path, n_eval=3, n_deployment=3)
    examples = load_apollo_examples(
        dataset, n_eval=3, n_deployment=3, prompt_mode="full_transcript", seed=0
    )
    models = [_model("m")]
    serial = run_factorial(models, examples, max_concurrent=1, seed=0)
    parallel = run_factorial(models, examples, max_concurrent=4, seed=0)
    # Concurrency may reorder rollouts; compare as sets of (example_id, condition).
    serial_keys = {(r["example_id"], r["condition"]) for r in serial}
    parallel_keys = {(r["example_id"], r["condition"]) for r in parallel}
    assert serial_keys == parallel_keys
    assert len(parallel) == len(serial)


def test_include_opus_wires_anthropic_thinking_wrapper(monkeypatch):
    """opus-4.7 needs adaptive thinking enabled or its rollouts come back
    with empty reasoning_trace, which silently breaks the VEA arm of the
    analysis. Verify the wrapper is used (not the bare AnthropicClient)
    when --include-opus is set."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    models = _build_model_lineup(
        openrouter_models=["qwen/qwen3-235b-a22b-thinking-2507"],
        include_opus=True,
        anthropic_model="claude-opus-4-7",
        thinking_budget=4096,
        thinking_effort="max",
        thinking_display="summarized",
    )
    assert len(models) == 2
    anthropic_model = models[-1]
    assert isinstance(anthropic_model, _AnthropicWithThinking), (
        f"expected _AnthropicWithThinking, got {type(anthropic_model).__name__}"
    )
    # Spot-check the thinking config flowed through to the wrapper instance
    assert anthropic_model._thinking_budget == 4096
    assert anthropic_model._thinking_effort == "max"
    assert anthropic_model._thinking_display == "summarized"


def test_no_opus_means_no_anthropic_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    models = _build_model_lineup(
        openrouter_models=["qwen/qwen3-235b-a22b-thinking-2507"],
        include_opus=False,
        anthropic_model="claude-opus-4-7",
    )
    assert len(models) == 1
    assert not isinstance(models[0], _AnthropicWithThinking)


def test_load_apollo_examples_returns_correct_label_counts(tmp_path):
    dataset = _apollo_dataset(tmp_path, n_eval=20, n_deployment=20)
    examples = load_apollo_examples(
        dataset, n_eval=7, n_deployment=4, prompt_mode="full_transcript", seed=0
    )
    eval_count = sum(1 for ex in examples if ex.metadata["apollo_is_eval"] == "true")
    deployment_count = sum(
        1 for ex in examples if ex.metadata["apollo_is_eval"] == "false"
    )
    assert eval_count == 7
    assert deployment_count == 4
