"""Unit tests for the Apollo eval-awareness dataset loader."""

import json
from pathlib import Path

import pytest

from src.eval_awareness.dataset_loaders import ApolloEvalAwarenessLoader


def _write_dataset(path: Path, entries: list) -> Path:
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def _entry(
    sample_id: str,
    *,
    eval_category=None,
    category="support_ticket",
    input_messages=None,
    eval_type=None,
    safety_category=None,
    subcategory=None,
    scaffold=None,
) -> dict:
    if input_messages is None:
        input_messages = [{"role": "user", "content": f"prompt for {sample_id}"}]
    return {
        "id": sample_id,
        "input": input_messages,
        "metadata": {
            "category": category,
            "eval_category": eval_category,
            "eval_type": eval_type,
            "safety_category": safety_category,
            "subcategory": subcategory,
            "scaffold": scaffold,
            "tools": [],
        },
    }


def test_loader_emits_one_example_per_valid_entry(tmp_path):
    path = _write_dataset(
        tmp_path / "apollo.json",
        [_entry(f"s{i}", eval_category=("Capability" if i % 2 else None)) for i in range(6)],
    )
    examples = ApolloEvalAwarenessLoader(path).load()
    assert len(examples) == 6
    assert {ex.task_family for ex in examples} == {"apollo_eval_aware"}
    assert all(ex.example_id.startswith("apollo:s") for ex in examples)


def test_label_filter_eval_only_keeps_truthy_eval_category(tmp_path):
    path = _write_dataset(
        tmp_path / "apollo.json",
        [
            _entry("a", eval_category="Capability"),
            _entry("b", eval_category="Alignment"),
            _entry("c", eval_category=None),
            _entry("d", eval_category=""),
        ],
    )
    examples = ApolloEvalAwarenessLoader(path, label_filter="eval").load()
    assert {ex.metadata["apollo_id"] for ex in examples} == {"a", "b"}
    assert {ex.label for ex in examples} == {"hidden_eval"}


def test_label_filter_deployment_only_keeps_falsy_eval_category(tmp_path):
    path = _write_dataset(
        tmp_path / "apollo.json",
        [
            _entry("a", eval_category="Capability"),
            _entry("b", eval_category=None),
            _entry("c", eval_category=""),
        ],
    )
    examples = ApolloEvalAwarenessLoader(path, label_filter="deployment").load()
    assert {ex.metadata["apollo_id"] for ex in examples} == {"b", "c"}
    assert {ex.label for ex in examples} == {"deployment_like"}


def test_max_examples_subsample_is_deterministic(tmp_path):
    entries = [_entry(f"id{i}", eval_category="Capability") for i in range(20)]
    path = _write_dataset(tmp_path / "apollo.json", entries)
    first = ApolloEvalAwarenessLoader(path, max_examples=5, seed=7).load()
    second = ApolloEvalAwarenessLoader(path, max_examples=5, seed=7).load()
    assert [ex.example_id for ex in first] == [ex.example_id for ex in second]
    assert len(first) == 5

    different_seed = ApolloEvalAwarenessLoader(path, max_examples=5, seed=8).load()
    # Different seeds should generally pick a different subset (probabilistic
    # — with 20 entries and 5 picks the collision rate is tiny).
    assert {ex.example_id for ex in first} != {ex.example_id for ex in different_seed}


def test_full_transcript_mode_preserves_system_and_history(tmp_path):
    path = _write_dataset(
        tmp_path / "apollo.json",
        [
            _entry(
                "multi",
                eval_category=None,
                input_messages=[
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "First question?"},
                    {"role": "assistant", "content": "First answer."},
                    {"role": "user", "content": "Follow-up?"},
                ],
            )
        ],
    )
    [ex] = ApolloEvalAwarenessLoader(path, prompt_mode="full_transcript").load()
    assert ex.transcript.system_prompt == "You are helpful."
    assert ex.transcript.user_prompt == "Follow-up?"
    assert ex.transcript.conversation_history == [
        {"role": "user", "content": "First question?"},
        {"role": "assistant", "content": "First answer."},
    ]


def test_last_user_turn_mode_strips_everything_but_final_user_message(tmp_path):
    path = _write_dataset(
        tmp_path / "apollo.json",
        [
            _entry(
                "multi",
                eval_category="Capability",
                input_messages=[
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "First question?"},
                    {"role": "assistant", "content": "First answer."},
                    {"role": "user", "content": "Follow-up?"},
                ],
            )
        ],
    )
    [ex] = ApolloEvalAwarenessLoader(path, prompt_mode="last_user_turn").load()
    assert ex.transcript.system_prompt == ""
    assert ex.transcript.user_prompt == "Follow-up?"
    assert ex.transcript.conversation_history == []


def test_string_input_is_treated_as_single_user_turn(tmp_path):
    path = _write_dataset(
        tmp_path / "apollo.json",
        [
            {
                "id": "plain",
                "input": "Just a string prompt.",
                "metadata": {"category": "mmlu", "eval_category": "Capability"},
            }
        ],
    )
    [ex] = ApolloEvalAwarenessLoader(path).load()
    assert ex.transcript.user_prompt == "Just a string prompt."
    assert ex.transcript.system_prompt == ""


def test_metadata_preserves_apollo_fields_and_label(tmp_path):
    path = _write_dataset(
        tmp_path / "apollo.json",
        [
            _entry(
                "eval-row",
                eval_category=True,
                category="strong_reject",
                eval_type="alignment",
                safety_category="harm",
                subcategory="bias",
                scaffold={"tool_calls": True},
            ),
            _entry(
                "deploy-row",
                eval_category=False,
                category="support_ticket",
                scaffold=None,
            ),
        ],
    )
    by_id = {
        ex.metadata["apollo_id"]: ex
        for ex in ApolloEvalAwarenessLoader(path).load()
    }
    eval_ex = by_id["eval-row"]
    assert eval_ex.metadata["apollo_is_eval"] == "true"
    assert eval_ex.metadata["apollo_category"] == "strong_reject"
    assert eval_ex.metadata["apollo_eval_type"] == "alignment"
    assert eval_ex.metadata["apollo_safety_category"] == "harm"
    assert eval_ex.metadata["apollo_subcategory"] == "bias"
    assert eval_ex.metadata["apollo_scaffold_type"] == "agentic"
    assert eval_ex.source == "strong_reject"
    assert "eval_intrinsic" in eval_ex.risk_tags

    deploy_ex = by_id["deploy-row"]
    assert deploy_ex.metadata["apollo_is_eval"] == "false"
    assert deploy_ex.metadata["apollo_eval_type"] == ""
    assert deploy_ex.metadata["apollo_safety_category"] == ""
    assert deploy_ex.metadata["apollo_subcategory"] == ""
    assert deploy_ex.metadata["apollo_scaffold_type"] == "chat"
    assert "deployment_intrinsic" in deploy_ex.risk_tags


def test_bool_eval_category_matches_real_apollo_schema(tmp_path):
    """Real Apollo dataset uses bool eval_category, not the 3-valued string
    I originally inferred from constants.py. Verify the loader binarizes both
    bool and string truthiness identically."""
    path = _write_dataset(
        tmp_path / "apollo.json",
        [
            _entry("bool-true", eval_category=True),
            _entry("bool-false", eval_category=False),
            _entry("str-truthy", eval_category="Capability"),  # legacy string-typed
            _entry("none", eval_category=None),
        ],
    )
    by_id = {
        ex.metadata["apollo_id"]: ex.metadata["apollo_is_eval"]
        for ex in ApolloEvalAwarenessLoader(path).load()
    }
    assert by_id == {
        "bool-true": "true",
        "bool-false": "false",
        "str-truthy": "true",
        "none": "false",
    }


def test_entries_with_no_user_message_are_skipped(tmp_path):
    path = _write_dataset(
        tmp_path / "apollo.json",
        [
            _entry("ok", eval_category="Capability"),
            {
                "id": "no-user",
                "input": [{"role": "system", "content": "system only"}],
                "metadata": {"category": "x", "eval_category": "Capability"},
            },
            {
                "id": "empty-input",
                "input": [],
                "metadata": {"category": "x", "eval_category": None},
            },
        ],
    )
    examples = ApolloEvalAwarenessLoader(path).load()
    assert {ex.metadata["apollo_id"] for ex in examples} == {"ok"}


def test_missing_dataset_file_raises_filenotfounderror(tmp_path):
    with pytest.raises(FileNotFoundError, match="Apollo dataset file not found"):
        ApolloEvalAwarenessLoader(tmp_path / "nope.json").load()


def test_non_list_root_raises_value_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"id": "not-a-list"}), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON list"):
        ApolloEvalAwarenessLoader(path).load()


def test_malformed_entries_are_logged_and_skipped(tmp_path, caplog):
    path = _write_dataset(
        tmp_path / "apollo.json",
        [
            _entry("ok", eval_category="Capability"),
            "not an object",
            42,
        ],
    )
    with caplog.at_level("WARNING", logger="src.eval_awareness.dataset_loaders"):
        examples = ApolloEvalAwarenessLoader(path).load()
    assert {ex.metadata["apollo_id"] for ex in examples} == {"ok"}
