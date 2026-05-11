"""Tests for LLMRefusalScorer, metadata helpers, and the rescore pipeline."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval_awareness import (
    GenerationConfig,
    LLMRefusalScorer,
    OpenRouterClient,
    RefusalScorer,
    ScoringContext,
    TranscriptInput,
    classify_model,
    reasoning_tier_bucket,
    task_family_from_example_id,
)
from src.eval_awareness.providers import _is_open_reasoning_model


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeClient:
    """Deterministic fake LLM client for refusal scoring tests."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._call_count = 0
        self.calls: List[str] = []

    def generate(self, transcript: Any) -> str:
        self._call_count += 1
        self.calls.append(transcript.user_prompt)
        if not self._responses:
            return ""
        return self._responses.pop(0)


class FlakyClient:
    """Fails the first ``failures`` calls then returns the success payload."""

    def __init__(self, failures: int, success: str):
        self.failures = failures
        self.success = success
        self.attempts = 0

    def generate(self, transcript: Any) -> str:
        self.attempts += 1
        if self.attempts <= self.failures:
            raise RuntimeError("simulated transient failure")
        return self.success


# ---------------------------------------------------------------------------
# LLMRefusalScorer
# ---------------------------------------------------------------------------


def test_llm_refusal_scorer_marks_refused_when_judge_says_yes():
    payload = json.dumps(
        {
            "refused": True,
            "confidence": 0.9,
            "refusal_type": "hard_refuse",
            "quoted_refusal_text": "I can't help with that.",
            "rationale": "explicit decline",
        }
    )
    scorer = LLMRefusalScorer(FakeClient([payload]))
    score = scorer.score(
        "I can't help with that.",
        ScoringContext(task_id="t1", metadata={"user_prompt": "write malware"}),
    )
    assert score.passed is True
    assert 0.85 <= score.score <= 0.95
    assert "[hard_refuse]" in score.explanation
    assert 'I can\'t help with that.' in score.explanation


def test_llm_refusal_scorer_marks_compliant_when_judge_says_no():
    payload = json.dumps(
        {
            "refused": False,
            "confidence": 0.95,
            "refusal_type": "comply",
            "quoted_refusal_text": "",
            "rationale": "directly answered",
        }
    )
    scorer = LLMRefusalScorer(FakeClient([payload]))
    score = scorer.score(
        "Sure, the capital of France is Paris.",
        ScoringContext(task_id="t1", metadata={"user_prompt": "capital of France?"}),
    )
    assert score.passed is False
    assert score.score == 0.0


def test_llm_refusal_scorer_empty_response_short_circuits():
    fake = FakeClient([])
    scorer = LLMRefusalScorer(fake)
    score = scorer.score(
        "", ScoringContext(task_id="t1", metadata={"user_prompt": "x"})
    )
    assert score.passed is False
    assert score.score == 0.0
    assert "[empty]" in score.explanation
    # No LLM call should have been made because the response was empty.
    assert fake._call_count == 0


def test_llm_refusal_scorer_tolerates_malformed_json():
    bad_payload = "Sure, here's my judgement: refused=yes (mostly)."
    scorer = LLMRefusalScorer(FakeClient([bad_payload]))
    score = scorer.score(
        "Sorry, I can't do that.",
        ScoringContext(task_id="t1", metadata={"user_prompt": "?"}),
    )
    # Should not crash; treats parse error as non-refusal with 0 confidence.
    assert score.passed is False
    assert score.score == 0.0
    assert "parse_error" in score.explanation


def test_llm_refusal_scorer_retries_transient_failures():
    payload = json.dumps({"refused": True, "confidence": 0.7, "refusal_type": "soft_decline", "quoted_refusal_text": "I'd rather not", "rationale": "soft hedging"})
    client = FlakyClient(failures=1, success=payload)
    scorer = LLMRefusalScorer(client, max_attempts=3, backoff_seconds=0.0)
    score = scorer.score(
        "I'd rather not.",
        ScoringContext(task_id="t1", metadata={"user_prompt": "do X"}),
    )
    assert score.passed is True
    assert client.attempts == 2


def test_llm_refusal_scorer_handles_total_failure_gracefully():
    client = FlakyClient(failures=10, success="never")
    scorer = LLMRefusalScorer(client, max_attempts=2, backoff_seconds=0.0)
    score = scorer.score(
        "Some response",
        ScoringContext(task_id="t1", metadata={"user_prompt": "?"}),
    )
    assert score.passed is False
    assert score.score == 0.0
    assert "error" in score.explanation.lower()


def test_llm_refusal_scorer_requires_generate_method():
    with pytest.raises(TypeError):
        LLMRefusalScorer(object())  # no .generate()


def test_llm_refusal_scorer_caps_confidence_when_no_quote():
    payload = json.dumps(
        {
            "refused": True,
            "confidence": 1.0,  # judge claims 100%
            "refusal_type": "hard_refuse",
            "quoted_refusal_text": "",  # but gives no evidence
            "rationale": "(vibes)",
        }
    )
    scorer = LLMRefusalScorer(FakeClient([payload]))
    score = scorer.score(
        "Some ambiguous response",
        ScoringContext(task_id="t1", metadata={"user_prompt": "?"}),
    )
    # Should clamp to 0.8 because no quote was attached.
    assert score.passed is True
    assert score.score <= 0.8


# ---------------------------------------------------------------------------
# Compatibility with the regex scorer (drop-in equivalence)
# ---------------------------------------------------------------------------


def test_llm_and_regex_scorers_have_compatible_interface():
    """Both scorers should accept (response, ScoringContext) and return BehaviorScore."""
    regex = RefusalScorer()
    payload = json.dumps(
        {"refused": False, "confidence": 0.0, "refusal_type": "comply", "quoted_refusal_text": "", "rationale": "comply"}
    )
    llm = LLMRefusalScorer(FakeClient([payload]))
    ctx = ScoringContext(task_id="t", metadata={"user_prompt": "?"})
    a = regex.score("answer", ctx)
    b = llm.score("answer", ctx)
    assert a.__class__ is b.__class__
    assert a.name and b.name
    assert isinstance(a.passed, bool) and isinstance(b.passed, bool)


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "example_id, expected",
    [
        ("mmlu:anatomy:test:42", "knowledge"),
        ("gsm8k:test:0", "math"),
        ("gpqa:gpqa_main:train:7", "knowledge"),
        ("humaneval:HumanEval/4", "code"),
        ("mbpp:42", "code"),
        ("truthfulqa:validation:5", "truthfulness"),
        ("beavertails:30k_train:0", "safety"),
        ("synth_safety_42", "safety"),
        ("", "unknown"),
        ("totally_unknown_thing:9", "unknown"),
    ],
)
def test_task_family_inference(example_id: str, expected: str):
    assert task_family_from_example_id(example_id) == expected


def test_classify_openai_reasoning():
    tier = classify_model("openai:gpt-5.5")
    assert tier.vendor == "openai"
    assert tier.model_id == "gpt-5.5"
    assert tier.is_reasoning is True
    assert tier.reasoning_provenance == "closed_polished"
    assert reasoning_tier_bucket(tier) == "closed_polished_reasoning"


def test_classify_openai_non_reasoning():
    tier = classify_model("openai:gpt-4o-mini")
    assert tier.vendor == "openai"
    assert tier.is_reasoning is False
    assert tier.reasoning_provenance == "none"
    assert reasoning_tier_bucket(tier) == "non_reasoning"


def test_classify_anthropic_reasoning():
    tier = classify_model("anthropic:claude-opus-4-7")
    assert tier.vendor == "anthropic"
    assert tier.is_reasoning is True
    assert tier.reasoning_provenance == "closed_polished"


def test_classify_anthropic_non_reasoning():
    tier = classify_model("anthropic:claude-haiku-4-5-20251001")
    assert tier.vendor == "anthropic"
    assert tier.is_reasoning is False


def test_classify_open_reasoning_via_openrouter():
    tier = classify_model("openrouter:deepseek/deepseek-r1")
    assert tier.vendor == "openrouter"
    assert tier.is_reasoning is True
    assert tier.reasoning_provenance == "open_full"
    assert reasoning_tier_bucket(tier) == "open_full_reasoning"


def test_classify_unprefixed_label():
    tier = classify_model("gpt-5.5")
    assert tier.vendor == "openai"
    assert tier.is_reasoning is True


# ---------------------------------------------------------------------------
# rescore_cross_protocol.py — synthetic end-to-end
# ---------------------------------------------------------------------------


def _make_rollout(
    *,
    model: str,
    condition: str,
    example_id: str,
    response: str,
    rollout_index: int = 0,
    prompt: str = "test prompt",
    system_prompt: str = "",
) -> Dict[str, Any]:
    return {
        "model_name": model,
        "condition": condition,
        "example_id": example_id,
        "rollout_index": rollout_index,
        "prompt": prompt,
        "system_prompt": system_prompt,
        "response": response,
        "refusal": 1.0 if "can't" in response.lower() else 0.0,
    }


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def test_rescore_end_to_end_writes_all_artifacts(tmp_path):
    from scripts.rescore_cross_protocol import main as rescore_main

    run_dir = tmp_path / "run"
    rollouts: List[Dict[str, Any]] = []
    for i in range(8):
        for cond in ("baseline", "neutral", "inject", "remove"):
            response = (
                "I can't help with that." if cond == "inject" else "Here is the answer."
            )
            rollouts.append(
                _make_rollout(
                    model="openai:gpt-4o-mini",
                    condition=cond,
                    example_id=f"beavertails:30k_train:{i}",
                    response=response,
                    rollout_index=i,
                )
            )
        for cond in ("baseline", "neutral", "inject", "remove"):
            rollouts.append(
                _make_rollout(
                    model="openai:gpt-5.5",
                    condition=cond,
                    example_id=f"beavertails:30k_train:{i}",
                    response="Sure, here's an answer.",
                    rollout_index=i,
                )
            )
    _write_jsonl(run_dir / "rollouts.jsonl", rollouts)

    rescore_main([str(run_dir), "--in-place"])

    summary_path = run_dir / "cross_protocol_summary.json"
    report_path = run_dir / "cross_protocol_report.md"
    scored_path = run_dir / "rollouts_scored.jsonl"
    assert summary_path.exists()
    assert report_path.exists()
    assert scored_path.exists()

    summary = json.loads(summary_path.read_text())
    assert summary["rescored_records"] == len(rollouts)
    # gpt-4o-mini should show inflation: inject refusal > 0, neutral refusal == 0
    per_model = {row["model_name"]: row for row in summary["per_model"]}
    mini = per_model["openai:gpt-4o-mini"]
    assert mini["inject_refusal_rate"] > 0.5
    assert mini["neutral_refusal_rate"] == 0.0
    assert mini["inject_minus_neutral"] > 0.5
    assert mini["inject_n"] == 8

    # gpt-5.5 (no refusals at all) should have zero across the board.
    gpt55 = per_model["openai:gpt-5.5"]
    assert gpt55["inject_refusal_rate"] == 0.0

    # Tier cuts: gpt-4o-mini = non_reasoning/openai, gpt-5.5 = closed_polished_reasoning/openai
    tier_buckets = {
        (row["model_tier_bucket"], row["model_vendor"])
        for row in summary["reasoning_tier_cuts"]
    }
    assert ("non_reasoning", "openai") in tier_buckets
    assert ("closed_polished_reasoning", "openai") in tier_buckets

    # Task family breakdown should include the safety family for both models.
    family_pairs = {
        (row["model_name"], row["task_family"])
        for row in summary["task_family_breakdown"]
    }
    assert ("openai:gpt-4o-mini", "safety") in family_pairs
    assert ("openai:gpt-5.5", "safety") in family_pairs

    report = report_path.read_text()
    assert "Run configuration" in report
    assert "Model identity" in report
    assert "Reasoning-tier cuts" in report
    assert "Task-family breakdown" in report
    assert "Claims supported by this run" in report
    # LLM judge wasn't run; that section should say so explicitly.
    assert "LLM judge was not run" in report


def test_rescore_with_corpus_manifest_uses_templates(tmp_path):
    from scripts.rescore_cross_protocol import main as rescore_main

    run_dir = tmp_path / "run"
    rollouts = []
    for i in range(4):
        for cond in ("baseline", "neutral", "inject", "remove"):
            rollouts.append(
                _make_rollout(
                    model="anthropic:claude-haiku-4-5-20251001",
                    condition=cond,
                    example_id=f"beavertails:30k_train:{i}",
                    response="ok" if cond != "inject" else "I can't help with that.",
                    rollout_index=i,
                )
            )
    _write_jsonl(run_dir / "rollouts.jsonl", rollouts)

    manifest = {
        "n_prompts_requested": 4,
        "n_prompts_actual": 4,
        "n_rollouts": 1,
        "safety_only": True,
        "conditions": ["baseline", "inject", "remove", "neutral"],
        "prompt_templates": {
            "arxiv_inject_system": "INJECT TEMPLATE TEXT",
            "arxiv_remove_system": "REMOVE TEMPLATE TEXT",
            "arxiv_neutral_system": "NEUTRAL TEMPLATE TEXT",
            "goodfire_system": "GOODFIRE TEMPLATE TEXT",
        },
        "task_family_counts": {"safety": 4},
        "examples": [
            {
                "example_id": f"beavertails:30k_train:{i}",
                "task_family": "safety",
                "source": "beavertails",
                "label": "safety_eval",
            }
            for i in range(4)
        ],
        "models": [
            {
                "label": "anthropic:claude-haiku-4-5-20251001",
                "vendor": "anthropic",
                "model_id": "claude-haiku-4-5-20251001",
                "is_reasoning": False,
                "reasoning_provenance": "none",
                "notes": "",
            }
        ],
        "reasoning_effort_requested": "default",
        "judge_provider": "openai",
        "judge_model": "gpt-4o-mini",
    }
    (run_dir / "corpus_manifest.json").write_text(json.dumps(manifest))

    rescore_main([str(run_dir), "--in-place"])

    report = (run_dir / "cross_protocol_report.md").read_text()
    assert "INJECT TEMPLATE TEXT" in report
    assert "NEUTRAL TEMPLATE TEXT" in report
    # Manifest's reasoning_effort_requested should appear in run config.
    assert "Reasoning effort requested" in report


def test_rescore_runs_with_llm_judge_via_fake(tmp_path, monkeypatch):
    """LLM judge cross-check should populate scorer validation table."""
    from scripts import rescore_cross_protocol as rescore

    # Stub out build_default_llm_refusal_scorer to return a FakeClient-backed
    # scorer so the test doesn't need API keys.
    def _make_scorer(provider: str, model: str = None, **kwargs):
        # Mark every response containing "can't" as refused, everything else as comply.
        class JudgeClient:
            model = "fake-judge"

            def generate(self, transcript):
                text = transcript.user_prompt
                if "I can't" in text:
                    return json.dumps(
                        {
                            "refused": True,
                            "confidence": 0.9,
                            "refusal_type": "hard_refuse",
                            "quoted_refusal_text": "I can't",
                            "rationale": "fake judge",
                        }
                    )
                return json.dumps(
                    {
                        "refused": False,
                        "confidence": 0.9,
                        "refusal_type": "comply",
                        "quoted_refusal_text": "",
                        "rationale": "fake judge",
                    }
                )

        return LLMRefusalScorer(JudgeClient())

    monkeypatch.setattr(
        rescore, "build_default_llm_refusal_scorer", _make_scorer
    )

    run_dir = tmp_path / "run"
    rollouts = []
    for i in range(3):
        for cond in ("baseline", "neutral", "inject", "remove"):
            response = (
                "I can't help with that."
                if cond == "inject"
                else "Here is the answer."
            )
            rollouts.append(
                _make_rollout(
                    model="openai:gpt-4o-mini",
                    condition=cond,
                    example_id=f"beavertails:30k_train:{i}",
                    response=response,
                    rollout_index=i,
                )
            )
    _write_jsonl(run_dir / "rollouts.jsonl", rollouts)

    rescore.main(
        [
            str(run_dir),
            "--in-place",
            "--llm-judge-provider",
            "openai",
            "--llm-judge-model",
            "gpt-4o-mini",
        ]
    )

    summary = json.loads((run_dir / "cross_protocol_summary.json").read_text())
    overall = summary["scorer_validation"]["overall"]
    assert overall["n"] == len(rollouts)
    # Fake judge perfectly agrees with the regex on these synthetic rollouts.
    assert overall["agreement"] == 1.0

    scored = [
        json.loads(line)
        for line in (run_dir / "rollouts_scored.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert all(record["refusal_llm"] is not None for record in scored)


# ---------------------------------------------------------------------------
# validate_refusal_scorer.py — sample mode + score mode
# ---------------------------------------------------------------------------


def test_validate_sample_mode_writes_csv(tmp_path):
    from scripts.validate_refusal_scorer import main as validate_main

    run_dir = tmp_path / "run"
    rollouts = []
    for i in range(20):
        cond = ("baseline", "neutral", "inject", "remove")[i % 4]
        response = "I can't help with that." if i % 2 == 0 else "Sure, here you go."
        rollouts.append(
            _make_rollout(
                model="openai:gpt-4o-mini",
                condition=cond,
                example_id=f"beavertails:30k_train:{i}",
                response=response,
                rollout_index=i,
            )
        )
    _write_jsonl(run_dir / "rollouts.jsonl", rollouts)

    csv_path = run_dir / "labelling.csv"
    validate_main([str(run_dir), "--n-per-bucket", "2", "--output", str(csv_path)])

    assert csv_path.exists()
    with csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) > 0
    # Every CSV row should expose the required columns.
    expected = {
        "response_id",
        "model_name",
        "condition",
        "task_family",
        "example_id",
        "rollout_index",
        "prompt",
        "response",
        "refusal_regex",
        "refusal_llm",
        "llm_refusal_type",
        "human_refusal",
        "human_notes",
    }
    assert expected.issubset(rows[0].keys())
    # task_family should be inferred (beavertails → safety).
    assert all(row["task_family"] == "safety" for row in rows)


def test_validate_score_mode_computes_metrics(tmp_path):
    from scripts.validate_refusal_scorer import main as validate_main

    run_dir = tmp_path / "run"
    csv_path = run_dir / "labelling.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Hand-built CSV with 4 records: 2 correct, 1 false-positive, 1 miss.
    fieldnames = (
        "response_id",
        "model_name",
        "condition",
        "task_family",
        "example_id",
        "rollout_index",
        "prompt",
        "response",
        "refusal_regex",
        "refusal_llm",
        "llm_refusal_type",
        "human_refusal",
        "human_notes",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerow(
            {
                "response_id": "r0",
                "model_name": "openai:gpt-4o-mini",
                "condition": "inject",
                "task_family": "safety",
                "example_id": "beavertails:0",
                "rollout_index": "0",
                "prompt": "x",
                "response": "y",
                "refusal_regex": "1.0",
                "refusal_llm": "1.0",
                "llm_refusal_type": "hard_refuse",
                "human_refusal": "yes",
                "human_notes": "",
            }
        )
        writer.writerow(
            {
                "response_id": "r1",
                "model_name": "openai:gpt-4o-mini",
                "condition": "neutral",
                "task_family": "safety",
                "example_id": "beavertails:1",
                "rollout_index": "0",
                "prompt": "x",
                "response": "y",
                "refusal_regex": "0.0",
                "refusal_llm": "0.0",
                "llm_refusal_type": "comply",
                "human_refusal": "no",
                "human_notes": "",
            }
        )
        writer.writerow(
            {
                "response_id": "r2",
                "model_name": "openai:gpt-4o-mini",
                "condition": "inject",
                "task_family": "safety",
                "example_id": "beavertails:2",
                "rollout_index": "0",
                "prompt": "x",
                "response": "y",
                "refusal_regex": "1.0",
                "refusal_llm": "0.0",
                "llm_refusal_type": "comply",
                "human_refusal": "no",
                "human_notes": "regex false positive",
            }
        )
        writer.writerow(
            {
                "response_id": "r3",
                "model_name": "openai:gpt-4o-mini",
                "condition": "inject",
                "task_family": "safety",
                "example_id": "beavertails:3",
                "rollout_index": "0",
                "prompt": "x",
                "response": "y",
                "refusal_regex": "0.0",
                "refusal_llm": "1.0",
                "llm_refusal_type": "soft_decline",
                "human_refusal": "yes",
                "human_notes": "regex missed",
            }
        )

    validate_main([str(run_dir), "--score", "--input", str(csv_path)])

    metrics_path = csv_path.with_suffix(".metrics.json")
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text())
    assert metrics["labeled"] == 4
    regex = metrics["regex_vs_human"]
    # tp=1 (r0), fp=1 (r2), fn=1 (r3), tn=1 (r1) -> agreement 0.5
    assert regex["true_positive"] == 1
    assert regex["false_positive"] == 1
    assert regex["false_negative"] == 1
    assert regex["true_negative"] == 1
    assert abs(regex["agreement"] - 0.5) < 1e-6

    llm = metrics["llm_vs_human"]
    # LLM agrees with human on all 4: tp=2, fn=0, fp=0, tn=2
    assert llm["true_positive"] == 2
    assert llm["true_negative"] == 2
    assert llm["false_positive"] == 0
    assert llm["false_negative"] == 0
    assert llm["agreement"] == 1.0


# ---------------------------------------------------------------------------
# Track B - OpenRouter dispatch + open-reasoning token-budget bump
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_id, expected",
    [
        # DeepSeek family
        ("deepseek/deepseek-r1", True),
        ("deepseek/deepseek-r1-0528", True),
        ("deepseek-ai/deepseek-r1-distill", True),
        ("deepseek/deepseek-r2", True),
        ("deepseek/deepseek-v3.2-exp", True),
        ("deepseek/deepseek-v4", True),
        ("deepseek/deepseek-v4-thinking", True),
        ("deepseek-ai/deepseek-v4", True),
        # Qwen family (explicit + "thinking" heuristic)
        ("qwen/qwq-32b", True),
        ("qwen-qwq", True),
        ("qwq-max", True),
        ("qwen/qwen3-235b-a22b-thinking-2507", True),
        ("qwen/qwen3-30b-a3b-thinking-2507", True),
        ("qwen/qwen3-32b-thinking", True),
        # Kimi / Moonshot
        ("kimi-k2.5", True),
        ("moonshotai/kimi-k2", True),
        ("moonshotai/kimi-k2-0905", True),
        ("moonshotai/kimi-k2-thinking", True),
        # GLM / Zhipu thinking
        ("zai-org/glm-4.6", True),
        ("glm-4.5-thinking", True),
        # OpenThinker / Open Thoughts
        ("openthinker-32b", True),
        ("open-thoughts/openthinker3-7b", True),
        # OLMo
        ("olmo-3-think", True),
        ("olmo-2-think-7b", True),
        # Negative cases
        ("meta-llama/llama-3.3-70b-instruct", False),
        ("mistralai/mistral-large", False),
        ("gpt-4o-mini", False),
        ("anthropic/claude-haiku-4-5", False),
        ("", False),
    ],
)
def test_is_open_reasoning_model(model_id: str, expected: bool):
    assert _is_open_reasoning_model(model_id) is expected


def test_cross_protocol_discover_models_picks_up_openrouter(monkeypatch):
    """Setting OPENROUTER_API_KEY + CROSS_OPENROUTER_MODELS dispatches OpenRouter clients."""
    from scripts.run_cross_protocol_comparison import _discover_models

    # Wipe any provider keys that might add unrelated clients.
    for var in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "CROSS_OLLAMA_MODELS",
        "CROSS_OPENAI_MODELS",
        "CROSS_ANTHROPIC_MODELS",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")
    monkeypatch.setenv(
        "CROSS_OPENROUTER_MODELS",
        "deepseek/deepseek-r1, qwen/qwq-32b ,meta-llama/llama-3.3-70b-instruct",
    )
    models = _discover_models()
    names = [m.name for m in models]
    assert names == [
        "openrouter:deepseek/deepseek-r1",
        "openrouter:qwen/qwq-32b",
        "openrouter:meta-llama/llama-3.3-70b-instruct",
    ]
    assert all(isinstance(m, OpenRouterClient) for m in models)


def test_cross_protocol_discover_models_skips_openrouter_without_model_list(
    monkeypatch, caplog
):
    """OPENROUTER_API_KEY without CROSS_OPENROUTER_MODELS logs a warning and falls through."""
    from scripts.run_cross_protocol_comparison import _discover_models

    for var in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "CROSS_OLLAMA_MODELS",
        "CROSS_OPENROUTER_MODELS",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")
    with caplog.at_level("WARNING"):
        models = _discover_models()
    # Should fall back to scripted, and warn.
    assert any("CROSS_OPENROUTER_MODELS is empty" in r.message for r in caplog.records)
    assert all(not isinstance(m, OpenRouterClient) for m in models)


def test_openai_compatible_open_reasoning_token_budget_floor(monkeypatch):
    """Open-reasoning models routed through OpenRouter get max_tokens floored to 2048.

    Without the floor, the visible response shares a too-small budget with the
    reasoning trace and comes back near-empty. This test asserts the payload
    sent to the upstream API has the bumped budget.
    """
    captured: Dict[str, Any] = {}

    class FakeResponse:
        status_code = 200
        headers = {"x-request-id": "rid-1"}

        def json(self):
            return {
                "choices": [
                    {"message": {"content": "ok"}, "finish_reason": "stop"}
                ],
                "usage": {},
            }

    def fake_post(url, headers, payload, timeout_seconds, retry):
        captured["payload"] = dict(payload)
        return FakeResponse()

    import src.eval_awareness.providers as providers_module

    monkeypatch.setattr(providers_module, "_post_with_retries", fake_post)
    client = OpenRouterClient(
        name="openrouter:deepseek/deepseek-r1",
        base_url="https://openrouter.ai/api/v1",
        api_key="fake",
        model="deepseek/deepseek-r1",
    )
    transcript = TranscriptInput(user_prompt="hello")
    config = GenerationConfig(temperature=0.0, max_tokens=256)
    client.generate_with_config(transcript, config)
    payload = captured["payload"]
    # Reasoning models keep max_tokens (OpenRouter accepts it), but the floor
    # is 2048 so visible output isn't crowded out by the reasoning trace.
    assert "max_completion_tokens" not in payload
    assert payload["max_tokens"] == 2048
    # Temperature should still be passed (DeepSeek R1 accepts it).
    assert payload["temperature"] == 0.0


def test_openai_compatible_non_reasoning_token_budget_unchanged(monkeypatch):
    """Non-reasoning models keep the caller's max_tokens verbatim."""
    captured: Dict[str, Any] = {}

    class FakeResponse:
        status_code = 200
        headers = {}

        def json(self):
            return {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {},
            }

    def fake_post(url, headers, payload, timeout_seconds, retry):
        captured["payload"] = dict(payload)
        return FakeResponse()

    import src.eval_awareness.providers as providers_module

    monkeypatch.setattr(providers_module, "_post_with_retries", fake_post)
    client = OpenRouterClient(
        name="openrouter:meta-llama/llama-3.3-70b-instruct",
        base_url="https://openrouter.ai/api/v1",
        api_key="fake",
        model="meta-llama/llama-3.3-70b-instruct",
    )
    transcript = TranscriptInput(user_prompt="hello")
    config = GenerationConfig(temperature=0.0, max_tokens=256)
    client.generate_with_config(transcript, config)
    payload = captured["payload"]
    # Non-reasoning model: no floor applied.
    assert payload["max_tokens"] == 256
    assert "max_completion_tokens" not in payload


def test_corpus_manifest_classifies_openrouter_open_reasoning():
    """OpenRouter open-reasoning models classify into the open_full bucket."""
    tier = classify_model("openrouter:qwen/qwq-32b")
    assert tier.vendor == "openrouter"
    assert tier.is_reasoning is True
    assert tier.reasoning_provenance == "open_full"
    assert reasoning_tier_bucket(tier) == "open_full_reasoning"

    tier_non = classify_model("openrouter:meta-llama/llama-3.3-70b-instruct")
    assert tier_non.vendor == "openrouter"
    assert tier_non.is_reasoning is False
    assert reasoning_tier_bucket(tier_non) == "non_reasoning"


# ---------------------------------------------------------------------------
# --preflight-only mode (model smoke-check before the full eval)
# ---------------------------------------------------------------------------


class _StubModelOK:
    name = "stub:happy"
    model = "stub-happy-1"

    class _LastResponse:
        model = "stub-happy-1-actual"

    def __init__(self):
        self.last_response = _StubModelOK._LastResponse()

    def generate_with_config(self, transcript, config):
        return "pong"


class _StubModelFail:
    name = "stub:broken"
    model = "stub-broken-1"
    last_response = None

    def generate_with_config(self, transcript, config):
        raise RuntimeError("400: invalid api key")


def test_preflight_check_reports_per_model_results():
    from scripts.run_cross_protocol_comparison import (
        _preflight_check,
        _render_preflight_table,
    )

    results = _preflight_check([_StubModelOK(), _StubModelFail()])
    assert len(results) == 2
    assert results[0].ok is True
    assert results[0].returned_model == "stub-happy-1-actual"
    assert "pong" in results[0].response_snippet
    assert results[1].ok is False
    assert "invalid api key" in results[1].error
    table = _render_preflight_table(results)
    assert "stub:happy" in table
    assert "stub:broken" in table
    assert "FAIL" in table
    assert "OK" in table


def test_preflight_only_short_circuits_main(tmp_path, monkeypatch):
    """--preflight-only exits before the main run loop and writes preflight.json."""
    from scripts import run_cross_protocol_comparison as cross

    monkeypatch.setattr(cross, "_discover_models", lambda: [_StubModelOK()])
    out_dir = tmp_path / "preflight"
    cross.main(
        [
            "--output-dir",
            str(out_dir),
            "--preflight-only",
            "--n-prompts",
            "1",
            "--n-rollouts",
            "1",
            "--safety-only",
        ]
    )
    preflight_path = out_dir / "preflight.json"
    assert preflight_path.exists()
    payload = json.loads(preflight_path.read_text())
    assert len(payload) == 1
    assert payload[0]["ok"] is True
    assert payload[0]["name"] == "stub:happy"
    # Importantly, the main run loop should NOT have written its summary.
    assert not (out_dir / "cross_protocol_summary.json").exists()


def test_preflight_only_fails_loudly_on_broken_model(tmp_path, monkeypatch):
    from scripts import run_cross_protocol_comparison as cross

    monkeypatch.setattr(
        cross, "_discover_models", lambda: [_StubModelOK(), _StubModelFail()]
    )
    out_dir = tmp_path / "preflight"
    with pytest.raises(SystemExit) as excinfo:
        cross.main(
            [
                "--output-dir",
                str(out_dir),
                "--preflight-only",
                "--n-prompts",
                "1",
                "--n-rollouts",
                "1",
                "--safety-only",
            ]
        )
    assert "preflight FAILED" in str(excinfo.value)
    # JSON should still be written so the user can inspect what failed.
    assert (out_dir / "preflight.json").exists()
