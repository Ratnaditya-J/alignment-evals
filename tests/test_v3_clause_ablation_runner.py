"""Unit tests for the V3 inject-clause ablation runner."""

import json
from pathlib import Path

from scripts.run_v3_clause_ablation import (
    V3_CONDITIONS,
    V3_CONDITION_SYSTEM_PROMPT,
    _build_model_lineup,
    _build_v3_transcript,
    load_v3_subsample,
    run_ablation,
    write_rollouts_jsonl,
)
from scripts.run_cross_protocol_comparison import ARXIV_INJECT_SYSTEM
from scripts.run_goodfire_vea import _AnthropicWithThinking
from src.eval_awareness import TranscriptInput
from src.eval_awareness.dataset import EvalAwarenessExample
from src.eval_awareness.runner import ScriptedModelClient


# ---------------------------------------------------------------------------
# Apollo fixture
# ---------------------------------------------------------------------------


def _apollo_entry(
    sample_id: str,
    *,
    eval_category: bool = True,
    eval_type: str = "alignment",
    category: str = "agentharm",
    text: str = None,
) -> dict:
    if text is None:
        text = f"alignment-eval prompt {sample_id}"
    return {
        "id": sample_id,
        "input": [{"role": "user", "content": text}],
        "metadata": {
            "category": category,
            "eval_category": eval_category,
            "eval_type": eval_type if eval_category else None,
            "safety_category": None,
            "subcategory": None,
            "scaffold": None,
            "tools": [],
        },
    }


def _write_apollo_dataset(tmp_path: Path, entries: list) -> Path:
    path = tmp_path / "apollo.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Condition specification (pinned for pre-reg)
# ---------------------------------------------------------------------------


def test_v3_has_exactly_five_conditions():
    assert V3_CONDITIONS == (
        "neutral",
        "eval_only",
        "reviewer_only",
        "comparison_only",
        "full_inject",
    )


def test_full_inject_byte_identical_to_v1_arxiv_inject():
    """The pre-reg requires full_inject to byte-equal V1's inject text so
    V3 numbers are directly comparable to V1/V2. Catch any drift here."""
    assert V3_CONDITION_SYSTEM_PROMPT["full_inject"] == ARXIV_INJECT_SYSTEM


def test_singleton_clauses_carry_their_distinctive_phrases_from_full_inject():
    """Each singleton must carry the distinctive phrase of its clause from
    the full inject. We can't require strict substring matching because the
    V1 inject uses elided syntax ('X will be reviewed... and compared...'),
    so the singletons are sentence-form rewrites of those elided clauses.
    What we DO require: each clause's content phrase is present in the
    full inject, so a reader can verify the decomposition by eye."""
    full = V3_CONDITION_SYSTEM_PROMPT["full_inject"]
    distinctive_phrases = {
        "eval_only": "this is an evaluation",
        "reviewer_only": "reviewed by safety researchers",
        "comparison_only": "compared against deployment behavior",
    }
    for singleton, phrase in distinctive_phrases.items():
        clause = V3_CONDITION_SYSTEM_PROMPT[singleton]
        assert phrase in clause, f"{singleton!r} missing distinctive phrase {phrase!r}"
        assert phrase in full, f"distinctive phrase {phrase!r} not in full_inject"


def test_neutral_is_byte_identical_to_v1_neutral():
    """Match V2/V1's neutral system prompt exactly so the baseline is shared."""
    assert V3_CONDITION_SYSTEM_PROMPT["neutral"] == "You are a helpful assistant."


# ---------------------------------------------------------------------------
# Transcript construction
# ---------------------------------------------------------------------------


def test_build_transcript_uses_condition_specific_system_prompt():
    example = EvalAwarenessExample(
        example_id="apollo:x",
        label="hidden_eval",
        task_family="apollo_eval_aware",
        source="agentharm",
        transcript=TranscriptInput(user_prompt="What is your safety policy?"),
        metadata={"apollo_is_eval": "true"},
    )
    for condition in V3_CONDITIONS:
        transcript = _build_v3_transcript(example, condition)
        assert transcript.system_prompt == V3_CONDITION_SYSTEM_PROMPT[condition]
        assert transcript.user_prompt == "What is your safety policy?"


def test_build_transcript_prepends_when_apollo_has_system_message():
    example = EvalAwarenessExample(
        example_id="apollo:y",
        label="hidden_eval",
        task_family="apollo_eval_aware",
        source="agentharm",
        transcript=TranscriptInput(
            user_prompt="prompt",
            system_prompt="Original Apollo system context.",
        ),
        metadata={"apollo_is_eval": "true"},
    )
    transcript = _build_v3_transcript(example, "eval_only")
    parts = transcript.system_prompt.split("\n\n")
    assert parts[0] == V3_CONDITION_SYSTEM_PROMPT["eval_only"]
    assert parts[-1] == "Original Apollo system context."


def test_build_transcript_preserves_conversation_history():
    example = EvalAwarenessExample(
        example_id="apollo:z",
        label="hidden_eval",
        task_family="apollo_eval_aware",
        source="agentharm",
        transcript=TranscriptInput(
            user_prompt="follow-up?",
            conversation_history=[
                {"role": "user", "content": "first?"},
                {"role": "assistant", "content": "first answer."},
            ],
        ),
        metadata={"apollo_is_eval": "true"},
    )
    transcript = _build_v3_transcript(example, "full_inject")
    assert transcript.conversation_history == [
        {"role": "user", "content": "first?"},
        {"role": "assistant", "content": "first answer."},
    ]


# ---------------------------------------------------------------------------
# Subsample selection
# ---------------------------------------------------------------------------


def test_load_v3_subsample_filters_to_alignment_only(tmp_path):
    entries = []
    for i in range(10):
        entries.append(_apollo_entry(f"a{i}", eval_type="alignment"))
        entries.append(_apollo_entry(f"c{i}", eval_type="capability"))
        entries.append(_apollo_entry(f"d{i}", eval_category=False, eval_type=None))
    dataset = _write_apollo_dataset(tmp_path, entries)
    selected = load_v3_subsample(
        dataset, n_prompts=5, prompt_mode="full_transcript", seed=2
    )
    assert len(selected) == 5
    assert all(ex.metadata.get("apollo_eval_type") == "alignment" for ex in selected)
    assert all(ex.metadata.get("apollo_is_eval") == "true" for ex in selected)


def test_load_v3_subsample_is_deterministic(tmp_path):
    entries = [_apollo_entry(f"a{i}", eval_type="alignment") for i in range(20)]
    dataset = _write_apollo_dataset(tmp_path, entries)
    first = load_v3_subsample(dataset, n_prompts=8, prompt_mode="full_transcript", seed=2)
    second = load_v3_subsample(dataset, n_prompts=8, prompt_mode="full_transcript", seed=2)
    assert [ex.example_id for ex in first] == [ex.example_id for ex in second]


def test_load_v3_subsample_is_disjoint_from_v2_seed_selection(tmp_path):
    """V2 uses sha256('0:apollo:{id}'); V3 uses sha256('v3:2:apollo:{id}').
    With many entries, the top-N selections should differ substantially."""
    from src.eval_awareness.dataset_loaders import ApolloEvalAwarenessLoader

    entries = [_apollo_entry(f"a{i}", eval_type="alignment") for i in range(50)]
    dataset = _write_apollo_dataset(tmp_path, entries)
    v2_selection = ApolloEvalAwarenessLoader(
        dataset, label_filter="eval", max_examples=10, seed=0
    ).load()
    v3_selection = load_v3_subsample(
        dataset, n_prompts=10, prompt_mode="full_transcript", seed=2
    )
    v2_ids = {ex.example_id for ex in v2_selection}
    v3_ids = {ex.example_id for ex in v3_selection}
    # Some overlap by chance is OK; the two sets should not be identical.
    assert v2_ids != v3_ids


def test_load_v3_subsample_takes_all_when_pool_smaller_than_n(tmp_path):
    entries = [_apollo_entry(f"a{i}", eval_type="alignment") for i in range(7)]
    dataset = _write_apollo_dataset(tmp_path, entries)
    selected = load_v3_subsample(
        dataset, n_prompts=200, prompt_mode="full_transcript", seed=2
    )
    assert len(selected) == 7


# ---------------------------------------------------------------------------
# Runner integration
# ---------------------------------------------------------------------------


def _model(name: str) -> ScriptedModelClient:
    return ScriptedModelClient(name=name)


def test_runner_produces_full_5_condition_cross_product(tmp_path):
    entries = [_apollo_entry(f"a{i}", eval_type="alignment") for i in range(6)]
    dataset = _write_apollo_dataset(tmp_path, entries)
    examples = load_v3_subsample(
        dataset, n_prompts=6, prompt_mode="full_transcript", seed=2
    )
    models = [_model("scripted-a"), _model("scripted-b")]
    rollouts = run_ablation(models, examples, max_concurrent=1, seed=0)
    # 2 models × 6 examples × 5 conditions = 60
    assert len(rollouts) == 2 * 6 * 5
    assert {r["condition"] for r in rollouts} == set(V3_CONDITIONS)
    assert {r["model_name"] for r in rollouts} == {"scripted-a", "scripted-b"}


def test_each_prompt_appears_in_all_5_conditions_per_model(tmp_path):
    entries = [_apollo_entry(f"a{i}", eval_type="alignment") for i in range(4)]
    dataset = _write_apollo_dataset(tmp_path, entries)
    examples = load_v3_subsample(
        dataset, n_prompts=4, prompt_mode="full_transcript", seed=2
    )
    rollouts = run_ablation([_model("m")], examples, max_concurrent=1, seed=0)
    by_example = {}
    for r in rollouts:
        by_example.setdefault(r["example_id"], set()).add(r["condition"])
    assert len(by_example) == 4
    for conditions in by_example.values():
        assert conditions == set(V3_CONDITIONS)


def test_rollout_records_carry_correct_system_prompt_per_condition(tmp_path):
    entries = [_apollo_entry("a0", eval_type="alignment")]
    dataset = _write_apollo_dataset(tmp_path, entries)
    examples = load_v3_subsample(
        dataset, n_prompts=1, prompt_mode="full_transcript", seed=2
    )
    rollouts = run_ablation([_model("m")], examples, max_concurrent=1, seed=0)
    by_condition = {r["condition"]: r["system_prompt"] for r in rollouts}
    for condition in V3_CONDITIONS:
        assert by_condition[condition] == V3_CONDITION_SYSTEM_PROMPT[condition]


def test_apollo_metadata_preserved_on_every_rollout(tmp_path):
    entries = [_apollo_entry(f"a{i}", eval_type="alignment") for i in range(2)]
    dataset = _write_apollo_dataset(tmp_path, entries)
    examples = load_v3_subsample(
        dataset, n_prompts=2, prompt_mode="full_transcript", seed=2
    )
    rollouts = run_ablation([_model("m")], examples, max_concurrent=1, seed=0)
    required = {
        "apollo_id",
        "apollo_is_eval",
        "apollo_eval_type",
        "apollo_category",
        "apollo_scaffold_type",
        "apollo_prompt_mode",
        "condition",
    }
    for r in rollouts:
        assert required <= set(r.keys())
        assert r["apollo_is_eval"] == "true"
        assert r["apollo_eval_type"] == "alignment"


def test_rollouts_jsonl_round_trips(tmp_path):
    entries = [_apollo_entry(f"a{i}", eval_type="alignment") for i in range(2)]
    dataset = _write_apollo_dataset(tmp_path, entries)
    examples = load_v3_subsample(
        dataset, n_prompts=2, prompt_mode="full_transcript", seed=2
    )
    rollouts = run_ablation([_model("m")], examples, max_concurrent=1, seed=0)
    out = tmp_path / "out" / "rollouts.jsonl"
    write_rollouts_jsonl(rollouts, out)
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    parsed = [json.loads(line) for line in lines]
    assert len(parsed) == len(rollouts)
    assert parsed[0]["condition"] in V3_CONDITIONS


def test_concurrent_runner_matches_serial(tmp_path):
    entries = [_apollo_entry(f"a{i}", eval_type="alignment") for i in range(3)]
    dataset = _write_apollo_dataset(tmp_path, entries)
    examples = load_v3_subsample(
        dataset, n_prompts=3, prompt_mode="full_transcript", seed=2
    )
    models = [_model("m")]
    serial = run_ablation(models, examples, max_concurrent=1, seed=0)
    parallel = run_ablation(models, examples, max_concurrent=4, seed=0)
    serial_keys = {(r["example_id"], r["condition"]) for r in serial}
    parallel_keys = {(r["example_id"], r["condition"]) for r in parallel}
    assert serial_keys == parallel_keys


# ---------------------------------------------------------------------------
# Lineup construction
# ---------------------------------------------------------------------------


def test_include_opus_wires_anthropic_thinking_wrapper(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-or-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anth-key")
    models = _build_model_lineup(
        openrouter_models=["qwen/qwen3-235b-a22b-thinking-2507"],
        include_opus=True,
        anthropic_model="claude-opus-4-7",
        thinking_budget=4096,
        thinking_effort="max",
        thinking_display="summarized",
    )
    assert len(models) == 2
    anthropic = models[-1]
    assert isinstance(anthropic, _AnthropicWithThinking)
    assert anthropic._thinking_budget == 4096
    assert anthropic._thinking_effort == "max"
    assert anthropic._thinking_display == "summarized"


def test_default_lineup_is_qwen3_only(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-or-key")
    models = _build_model_lineup(
        openrouter_models=["qwen/qwen3-235b-a22b-thinking-2507"],
        include_opus=False,
        anthropic_model="claude-opus-4-7",
    )
    names = [getattr(m, "name", "") for m in models]
    assert names == ["openrouter:qwen/qwen3-235b-a22b-thinking-2507"]


def test_include_baselines_adds_deepseek_and_kimi(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-or-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anth-key")
    models = _build_model_lineup(
        openrouter_models=["qwen/qwen3-235b-a22b-thinking-2507"],
        include_opus=True,
        anthropic_model="claude-opus-4-7",
        include_baselines=True,
    )
    names = [getattr(m, "name", "") for m in models]
    assert "openrouter:deepseek/deepseek-v3.2-exp" in names
    assert "openrouter:moonshotai/kimi-k2-thinking" in names
    assert "anthropic:claude-opus-4-7" in names
    assert "openrouter:qwen/qwen3-235b-a22b-thinking-2507" in names
    # Anthropic model still uses the thinking wrapper
    anthropic = [m for m in models if "anthropic" in getattr(m, "name", "")][0]
    assert isinstance(anthropic, _AnthropicWithThinking)


def test_include_baselines_does_not_double_add_if_user_listed(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-or-key")
    models = _build_model_lineup(
        openrouter_models=[
            "qwen/qwen3-235b-a22b-thinking-2507",
            "deepseek/deepseek-v3.2-exp",  # already in list
        ],
        include_opus=False,
        anthropic_model="claude-opus-4-7",
        include_baselines=True,
    )
    names = [getattr(m, "name", "") for m in models]
    # deepseek should appear exactly once even though both --openrouter-models
    # and --include-baselines try to add it.
    assert sum("deepseek" in n for n in names) == 1
