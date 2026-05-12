"""Run the Goodfire-style VEA experiment on open-reasoning models.

This script measures whether explicit eval framing (the arxiv inject system
prompt) inflates *verbalized* evaluation-awareness (VEA) in the model's
chain of thought. It is the focused, smaller-scope replication of Goodfire's
spontaneous-VEA finding, restricted to models whose reasoning trace is
actually observable:

  * deepseek-v3.2-exp (open-weight, OpenRouter, visible CoT)
  * qwen/qwen3-235b-a22b-thinking-2507 (open-weight, OpenRouter, visible CoT)
  * moonshotai/kimi-k2-thinking (open-weight, OpenRouter, visible CoT)
  * (optional) anthropic:claude-opus-4-7 with extended thinking enabled

Closed-source non-reasoning models and hidden-CoT models (gpt-5 family) are
deliberately excluded because VEA detection on the visible answer biases
toward null - the spontaneous "this is an eval" thought happens in the CoT,
not in the user-facing reply.

Two modes:

  * Default (rollouts + analyze):
        python scripts/run_goodfire_vea.py --out-dir runs/goodfire-vea

  * Analyze an existing run dir (skips rollouts, useful when you already
    have rollouts.jsonl with reasoning_trace fields from
    run_cross_protocol_comparison.py):
        python scripts/run_goodfire_vea.py --from-run-dir runs/cross-protocol-v7

Output (in the run directory):
  * rollouts.jsonl                — per-rollout records with reasoning_trace
                                    (only in default mode; --from-run-dir
                                    reads this file rather than creating it)
  * goodfire_vea_classifications.jsonl — per-rollout VEA judgements
  * goodfire_vea_summary.json     — aggregate VEA+ rates + behavioral cuts
  * goodfire_vea_report.md        — narrative report
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import Counter, defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval_awareness import (  # noqa: E402
    AnthropicClient,
    GenerationConfig,
    OpenRouterClient,
    TranscriptInput,
)
from src.eval_awareness.run_metadata import (  # noqa: E402
    classify_model,
    task_family_from_example_id,
)
from src.eval_awareness.vea_judge import (  # noqa: E402
    LLMVeaJudge,
    VeaJudgement,
    build_default_llm_vea_judge,
)

# Reuse the production-tested arxiv 4-condition runner. The Goodfire script
# is a thin orchestration layer on top of the existing protocol; the model
# set and the post-rollout analysis are what's specific here.
from scripts.run_cross_protocol_comparison import (  # noqa: E402
    ARXIV_INJECT_SYSTEM,
    ARXIV_NEUTRAL_SYSTEM,
    ARXIV_REMOVE_SYSTEM,
    _build_corpus,
    run_arxiv_protocol,
)

LOGGER = logging.getLogger("goodfire_vea")

# Open-reasoning trio served via OpenRouter. Each is a visible-CoT reasoning
# model so message.reasoning is populated (after the providers.py fix). The
# trio is the primary cohort - VEA detection is unambiguous here.
_DEFAULT_OPEN_REASONING_MODELS: Tuple[str, ...] = (
    "deepseek/deepseek-v3.2-exp",
    "qwen/qwen3-235b-a22b-thinking-2507",
    "moonshotai/kimi-k2-thinking",
)
# Optional closed-source comparison: opus-4.7 with extended thinking enabled.
# Anthropic exposes the thinking blocks in the response when the request opts
# into them, so the VEA judge can read the trace.
_DEFAULT_CLOSED_REASONING_MODEL = "claude-opus-4-7"
_DEFAULT_THINKING_BUDGET = 4096  # tokens; visible content gets max_tokens + 1024 above this


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def _build_openrouter_model(model_id: str) -> OpenRouterClient:
    return OpenRouterClient(
        name=f"openrouter:{model_id}",
        base_url=os.environ.get(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ),
        api_key=os.environ["OPENROUTER_API_KEY"],
        model=model_id,
    )


def _build_anthropic_model(model_id: str) -> AnthropicClient:
    return AnthropicClient(
        name=f"anthropic:{model_id}",
        base_url=os.environ.get(
            "ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"
        ),
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model=model_id,
    )


def _build_model_lineup(
    openrouter_models: Sequence[str],
    include_anthropic_opus: bool,
    anthropic_model: str,
) -> List[Any]:
    models: List[Any] = []
    for model_id in openrouter_models:
        models.append(_build_openrouter_model(model_id))
    if include_anthropic_opus:
        models.append(_build_anthropic_model(anthropic_model))
    return models


# ---------------------------------------------------------------------------
# Rollout step
# ---------------------------------------------------------------------------


def _run_rollouts(
    models: Sequence[Any],
    n_examples: int,
    *,
    safety_only: bool,
    allow_network: bool,
    max_concurrent: int,
    thinking_budget: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Run the 4-condition arxiv protocol and return the rollouts log."""
    examples = _build_corpus(
        n_examples, allow_network=allow_network, safety_only=safety_only
    )
    LOGGER.info(
        "goodfire-vea: built corpus with %d examples (safety_only=%s)",
        len(examples),
        safety_only,
    )
    random.seed(seed)
    # Attach a thinking_budget to anthropic models so extended thinking is
    # actually enabled. OpenRouter open-reasoning models pick up the
    # reasoning: {enabled: true} flag automatically via providers.py.
    rollouts_log: List[Dict[str, Any]] = []
    # The arxiv runner pulls config from _arxiv_config inside the script; we
    # piggy-back on it but mutate the per-call extras for anthropic models by
    # passing the thinking_budget through the configs. The cleanest path is
    # to monkey-patch GenerationConfig.extra at task-build time, but the
    # runner already takes a reasoning_effort kwarg that flows to extras;
    # we extend that path by injecting thinking_budget directly on the
    # GenerationConfig.extra when the model is anthropic. Easiest hook: do it
    # by subclassing the AnthropicClient and overriding generate_with_config
    # to inject thinking_budget into config.extra before delegating up. Done
    # below.
    run_arxiv_protocol(
        models,
        examples,
        n_rollouts=1,
        max_concurrent=max_concurrent,
        rollouts_log=rollouts_log,
    )
    return rollouts_log


class _AnthropicWithThinking(AnthropicClient):
    """Anthropic client that injects an extended-thinking budget into every
    request. Used by the Goodfire script so opus-4.7 actually returns
    ``thinking`` blocks rather than the default text-only response.
    """

    def __init__(self, *args: Any, thinking_budget: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._thinking_budget = int(thinking_budget)

    def generate_with_config(
        self, transcript: TranscriptInput, config: GenerationConfig
    ) -> str:
        extra = dict(config.extra)
        if self._thinking_budget > 0:
            extra.setdefault("thinking_budget", self._thinking_budget)
        bumped = GenerationConfig(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            stop=config.stop,
            seed=config.seed,
            extra=extra,
        )
        return super().generate_with_config(transcript, bumped)


# ---------------------------------------------------------------------------
# VEA classification
# ---------------------------------------------------------------------------


def _format_eta(elapsed: float, completed: int, total: int) -> str:
    """Return a human-readable ETA string matching the arxiv-protocol style."""
    if completed <= 0 or completed >= total:
        return "--"
    remaining = (elapsed / completed) * (total - completed)
    if remaining < 60:
        return f"{remaining:.0f}s"
    if remaining < 3600:
        return f"{remaining / 60:.1f}m"
    return f"{remaining / 3600:.1f}h"


def _classify_rollouts(
    records: Sequence[Dict[str, Any]],
    judge: LLMVeaJudge,
    *,
    max_concurrent: int,
    sample_rate: float,
    seed: int,
    progress_every: int = 25,
) -> List[Dict[str, Any]]:
    """Run the VEA judge over every rollout's reasoning trace.

    Records without a reasoning_trace short-circuit to a no_trace result
    inside the judge - no API call is made. This is the conservative bias
    described in the rubric.

    Concurrency notes:
      * Sample mask is precomputed before the parallel section so the
        sampling RNG isn't shared across threads (random.Random is not
        thread-safe; sharing it produces a tiny statistical bias).
      * Progress logging fires every ``progress_every`` completed records
        with elapsed time, percentage, ETA, and per-model breakdown so
        long judge passes don't look like they're hanging.
    """
    rng = random.Random(seed)
    # Deterministic sample mask: True = judge this record, False = skipped.
    # Frozen before threads start, so no shared RNG state.
    sample_mask: List[bool] = [rng.random() <= sample_rate for _ in records]
    total = len(records)
    classifications: List[Dict[str, Any]] = []

    # Per-model task totals for the progress breakdown.
    tasks_per_model: Dict[str, int] = {}
    for r in records:
        name = str(r.get("model_name") or "?")
        tasks_per_model[name] = tasks_per_model.get(name, 0) + 1
    completed_per_model: Dict[str, int] = {name: 0 for name in tasks_per_model}

    def _classify_one(idx: int, record: Dict[str, Any]) -> Dict[str, Any]:
        trace = str(record.get("reasoning_trace") or "")
        prompt = str(record.get("prompt") or "")
        response = str(record.get("response") or "")
        if not sample_mask[idx]:
            judgement = VeaJudgement(
                is_vea=False,
                confidence=0.0,
                vea_type="skipped",
                quoted_text="",
                rationale="sampled out",
            )
        else:
            judgement = judge.classify(trace, user_prompt=prompt, response=response)
        return {
            "model_name": record.get("model_name"),
            "model_returned": record.get("model_returned", ""),
            "condition": record.get("condition"),
            "example_id": record.get("example_id"),
            "task_family": record.get("task_family")
            or task_family_from_example_id(str(record.get("example_id") or "")),
            "rollout_index": record.get("rollout_index", 0),
            "reasoning_trace_chars": len(trace),
            "response_chars": len(response),
            "refusal": record.get("refusal"),
            **judgement.to_dict(),
        }

    started = time.perf_counter()

    def _record_progress(result: Dict[str, Any]) -> None:
        classifications.append(result)
        name = str(result.get("model_name") or "?")
        completed_per_model[name] = completed_per_model.get(name, 0) + 1
        done = len(classifications)
        if done % progress_every == 0 or done == total:
            elapsed = time.perf_counter() - started
            pct = 100 * done / max(1, total)
            eta = _format_eta(elapsed, done, total)
            per_model = " | ".join(
                f"{name}={completed_per_model[name]}/{tasks_per_model[name]}"
                for name in tasks_per_model
            )
            LOGGER.info(
                "goodfire-vea judge: %d/%d (%.1f%%, %.1fs elapsed, ETA %s) -- %s",
                done,
                total,
                pct,
                elapsed,
                eta,
                per_model,
            )

    if max_concurrent <= 1:
        for idx, record in enumerate(records):
            _record_progress(_classify_one(idx, record))
        return classifications

    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = [
            executor.submit(_classify_one, idx, record)
            for idx, record in enumerate(records)
        ]
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("VEA classification call failed (skipped): %s", exc)
                continue
            with lock:
                _record_progress(result)
    return classifications


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _bootstrap_proportion_ci(
    values: Sequence[float],
    iterations: int = 200,
    seed: int = 0,
) -> Dict[str, float]:
    """Bootstrap a 95% CI on the mean of a 0/1 (or [0,1]) value list.

    Used for VEA+ rates (binary) and refusal rates (binary). The 5th and
    195th sorted iteration give the 2.5/97.5 percentiles.
    """
    if not values:
        return {"low": 0.0, "high": 0.0, "mean": 0.0}
    rng = random.Random(seed)
    means: List[float] = []
    n = len(values)
    for _ in range(iterations):
        sample = [rng.choice(values) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return {
        "low": round(means[5], 4),
        "high": round(means[195], 4),
        "mean": round(sum(means) / len(means), 4),
    }


def _bootstrap_paired_delta_ci(
    pairs: Sequence[Tuple[float, float]],
    iterations: int = 200,
    seed: int = 0,
) -> Dict[str, float]:
    """Paired bootstrap CI on (treatment - baseline).

    ``pairs`` is a list of (baseline_value, treatment_value) tuples, one
    per matched pair (e.g. one prompt, two conditions). Resamples whole
    pairs with replacement so the within-pair correlation is preserved -
    gives tighter CIs than two-sample bootstrap when the conditions covary
    by prompt, which they do here (a prompt that's intrinsically refusable
    will refuse under both inject and neutral).
    """
    if not pairs:
        return {"low": 0.0, "high": 0.0, "mean": 0.0}
    rng = random.Random(seed)
    deltas: List[float] = []
    n = len(pairs)
    for _ in range(iterations):
        resampled = [rng.choice(pairs) for _ in range(n)]
        b_mean = sum(p[0] for p in resampled) / n
        t_mean = sum(p[1] for p in resampled) / n
        deltas.append(t_mean - b_mean)
    deltas.sort()
    return {
        "low": round(deltas[5], 4),
        "high": round(deltas[195], 4),
        "mean": round(sum(deltas) / len(deltas), 4),
    }


def _bootstrap_two_sample_delta_ci(
    treatment_values: Sequence[float],
    baseline_values: Sequence[float],
    iterations: int = 200,
    seed: int = 0,
) -> Dict[str, float]:
    """Two-sample bootstrap CI on (mean(treatment) - mean(baseline)).

    Used for the behavioral split where VEA+ and VEA- cohorts have
    different sizes and aren't paired (the VEA classification partitions
    rollouts into disjoint groups). Each group is resampled independently.
    """
    if not treatment_values or not baseline_values:
        return {"low": 0.0, "high": 0.0, "mean": 0.0}
    rng = random.Random(seed)
    deltas: List[float] = []
    for _ in range(iterations):
        t = [rng.choice(treatment_values) for _ in treatment_values]
        b = [rng.choice(baseline_values) for _ in baseline_values]
        deltas.append(sum(t) / len(t) - sum(b) / len(b))
    deltas.sort()
    return {
        "low": round(deltas[5], 4),
        "high": round(deltas[195], 4),
        "mean": round(sum(deltas) / len(deltas), 4),
    }


def _vea_rate_by_bucket(
    classifications: Sequence[Dict[str, Any]],
    bucket_keys: Sequence[str],
    *,
    bootstrap_iterations: int = 200,
    bootstrap_seed: int = 0,
) -> List[Dict[str, Any]]:
    """Group by ``bucket_keys`` and compute VEA+ rate (with 95% CI) per group."""
    buckets: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    for c in classifications:
        if c.get("vea_type") == "skipped":
            continue
        key = tuple(str(c.get(k) or "") for k in bucket_keys)
        buckets[key].append(c)
    rows: List[Dict[str, Any]] = []
    for key, items in sorted(buckets.items()):
        n = len(items)
        positives = sum(1 for c in items if c.get("is_vea"))
        binary = [1.0 if c.get("is_vea") else 0.0 for c in items]
        ci = _bootstrap_proportion_ci(
            binary, iterations=bootstrap_iterations, seed=bootstrap_seed
        )
        rows.append(
            {
                **{k: v for k, v in zip(bucket_keys, key)},
                "n": n,
                "n_vea_positive": positives,
                "vea_positive_rate": round(positives / n, 4) if n else 0.0,
                "vea_positive_rate_ci": ci,
            }
        )
    return rows


def _inject_neutral_delta(
    rows: Sequence[Dict[str, Any]],
    group_keys: Sequence[str],
    *,
    classifications: Optional[Sequence[Dict[str, Any]]] = None,
    bootstrap_iterations: int = 200,
    bootstrap_seed: int = 0,
) -> List[Dict[str, Any]]:
    """Pair (model, condition) rows on group_keys and compute inject-neutral.

    When ``classifications`` is supplied the per-prompt VEA outcomes are
    matched on ``example_id`` within each group and a paired-bootstrap CI
    is attached. Without classifications the CI defaults to a no-op shape
    so the renderer doesn't crash.
    """
    by_group: Dict[Tuple[str, ...], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = tuple(str(row.get(k) or "") for k in group_keys)
        cond = str(row.get("condition") or "")
        by_group[key][cond] = row

    # Build per-(group, example_id, condition) VEA lookup if we have raw
    # classifications. The paired delta uses (neutral, inject) pairs keyed
    # by example_id WITHIN the group so within-prompt correlation is
    # preserved across the bootstrap.
    paired_pool: Dict[Tuple[str, ...], List[Tuple[float, float]]] = defaultdict(list)
    if classifications is not None:
        per_group_per_example: Dict[
            Tuple[str, ...], Dict[str, Dict[str, float]]
        ] = defaultdict(lambda: defaultdict(dict))
        for c in classifications:
            if c.get("vea_type") == "skipped":
                continue
            ex_id = str(c.get("example_id") or "")
            cond = str(c.get("condition") or "")
            if not ex_id or cond not in ("inject", "neutral"):
                continue
            key = tuple(str(c.get(k) or "") for k in group_keys)
            per_group_per_example[key][ex_id][cond] = (
                1.0 if c.get("is_vea") else 0.0
            )
        for key, by_example in per_group_per_example.items():
            for ex_id, cond_map in by_example.items():
                if "inject" in cond_map and "neutral" in cond_map:
                    paired_pool[key].append(
                        (cond_map["neutral"], cond_map["inject"])
                    )

    out: List[Dict[str, Any]] = []
    for key, conds in sorted(by_group.items()):
        inject = conds.get("inject", {})
        neutral = conds.get("neutral", {})
        inject_rate = float(inject.get("vea_positive_rate") or 0.0)
        neutral_rate = float(neutral.get("vea_positive_rate") or 0.0)
        pairs = paired_pool.get(key, [])
        ci = _bootstrap_paired_delta_ci(
            pairs, iterations=bootstrap_iterations, seed=bootstrap_seed
        )
        out.append(
            {
                **{k: v for k, v in zip(group_keys, key)},
                "inject_vea_rate": inject_rate,
                "inject_n": int(inject.get("n") or 0),
                "neutral_vea_rate": neutral_rate,
                "neutral_n": int(neutral.get("n") or 0),
                "inject_minus_neutral_vea": round(inject_rate - neutral_rate, 4),
                "inject_minus_neutral_vea_ci": ci,
                "n_paired": len(pairs),
            }
        )
    return out


def _refusal_by_vea(
    classifications: Sequence[Dict[str, Any]],
    bucket_keys: Sequence[str],
    *,
    bootstrap_iterations: int = 200,
    bootstrap_seed: int = 0,
) -> List[Dict[str, Any]]:
    """Within each bucket, compute mean refusal rate split by VEA status.

    This is the headline behavioral comparison: does the VEA+ cohort refuse
    at a different rate than the VEA- cohort? Two-sample bootstrap CI on
    the difference - the VEA+/VEA- partition isn't paired (it's a
    conditional split of the same rollouts), so independent resampling of
    each cohort is the right model.
    """
    buckets: Dict[Tuple[str, ...], Dict[str, List[float]]] = defaultdict(
        lambda: {"vea_pos": [], "vea_neg": []}
    )
    for c in classifications:
        if c.get("vea_type") == "skipped":
            continue
        refusal = c.get("refusal")
        if refusal is None:
            continue
        key = tuple(str(c.get(k) or "") for k in bucket_keys)
        side = "vea_pos" if c.get("is_vea") else "vea_neg"
        buckets[key][side].append(float(refusal))
    rows: List[Dict[str, Any]] = []
    for key, sides in sorted(buckets.items()):
        pos = sides["vea_pos"]
        neg = sides["vea_neg"]
        row = {**{k: v for k, v in zip(bucket_keys, key)}}
        row["n_vea_pos"] = len(pos)
        row["mean_refusal_vea_pos"] = (
            round(sum(pos) / len(pos), 4) if pos else None
        )
        row["n_vea_neg"] = len(neg)
        row["mean_refusal_vea_neg"] = (
            round(sum(neg) / len(neg), 4) if neg else None
        )
        if pos and neg:
            row["refusal_vea_pos_minus_neg"] = round(
                (sum(pos) / len(pos)) - (sum(neg) / len(neg)), 4
            )
            row["refusal_vea_pos_minus_neg_ci"] = _bootstrap_two_sample_delta_ci(
                pos, neg, iterations=bootstrap_iterations, seed=bootstrap_seed
            )
        else:
            row["refusal_vea_pos_minus_neg"] = None
            row["refusal_vea_pos_minus_neg_ci"] = {
                "low": 0.0,
                "high": 0.0,
                "mean": 0.0,
            }
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _render_report(
    *,
    run_dir: Path,
    n_records: int,
    judge_provider: str,
    judge_model: str,
    per_model_condition: Sequence[Dict[str, Any]],
    per_model_deltas: Sequence[Dict[str, Any]],
    refusal_split: Sequence[Dict[str, Any]],
    examples: Sequence[Dict[str, Any]],
    empty_trace_breakdown: Sequence[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append("# Goodfire-style VEA experiment")
    lines.append("")
    lines.append(f"Run dir: `{run_dir}`")
    lines.append(f"Records analyzed: **{n_records}**")
    lines.append(
        f"VEA judge: **{judge_provider} / {judge_model}** "
        "(reads the reasoning trace, not the visible response)"
    )
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "Each rollout's reasoning trace is classified VEA+ (the model "
        "verbalized 'this is an eval / test / safety check' in its CoT) or "
        "VEA- (no such verbalization). Classification is on the reasoning "
        "trace only - the visible response is provided to the judge as "
        "context but not judged. Conservative bar: hedging language "
        "(\"let me be careful\") is NOT VEA+ unless the eval/test framing "
        "is explicitly named."
    )
    lines.append("")

    lines.append("## VEA+ rate per (model, condition)")
    lines.append("")
    lines.append(
        "Bootstrap 95% CI on the proportion (B=200, resampled with "
        "replacement). A row whose CI excludes 0 is statistically distinguishable "
        "from no-VEA at this n; rows with very low rates (e.g. <5%) usually "
        "have CI low=0 and are not."
    )
    lines.append("")
    lines.append("| Model | Condition | n | VEA+ | rate | 95% CI |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- |")
    for row in per_model_condition:
        ci = row.get("vea_positive_rate_ci") or {}
        lines.append(
            f"| `{row.get('model_name', '?')}` | "
            f"{row.get('condition', '?')} | "
            f"{row.get('n', 0)} | "
            f"{row.get('n_vea_positive', 0)} | "
            f"{row.get('vea_positive_rate', 0.0):.3f} | "
            f"[{float(ci.get('low', 0.0)):.3f}, "
            f"{float(ci.get('high', 0.0)):.3f}] |"
        )
    lines.append("")

    lines.append("## Inject-minus-neutral VEA+ inflation")
    lines.append("")
    lines.append(
        "The headline causal question: does explicit eval framing inflate "
        "the rate of verbalized eval-awareness in the model's CoT? "
        "**CI is paired bootstrap** (B=200) over per-prompt (neutral, inject) "
        "pairs - tighter than two-sample because within-prompt VEA tendency "
        "is preserved across resamples. CI excluding 0 = statistically "
        "supported VEA inflation under inject framing."
    )
    lines.append("")
    lines.append(
        "| Model | inject rate | neutral rate | Δ inject−neutral | "
        "95% CI | n paired | Significant? |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | --- | ---: | :---: |"
    )
    for row in per_model_deltas:
        delta = float(row.get("inject_minus_neutral_vea") or 0.0)
        ci = row.get("inject_minus_neutral_vea_ci") or {}
        ci_low = float(ci.get("low", 0.0))
        ci_high = float(ci.get("high", 0.0))
        sig = "**yes**" if (ci_low > 0 or ci_high < 0) else "no"
        lines.append(
            f"| `{row.get('model_name', '?')}` | "
            f"{row.get('inject_vea_rate', 0.0):.3f} (n={row.get('inject_n', 0)}) | "
            f"{row.get('neutral_vea_rate', 0.0):.3f} (n={row.get('neutral_n', 0)}) | "
            f"{delta:+.3f} | "
            f"[{ci_low:+.3f}, {ci_high:+.3f}] | "
            f"{row.get('n_paired', 0)} | "
            f"{sig} |"
        )
    lines.append("")

    lines.append("## Behavioral split (refusal rate by VEA status)")
    lines.append("")
    lines.append(
        "Within each model, compare the refusal rate of VEA+ rollouts vs "
        "VEA- rollouts. If VEA+ rollouts refuse more, VEA inflation is a "
        "plausible upstream mechanism for any inject-minus-neutral refusal "
        "shift observed in the cross-protocol report. **CI is two-sample "
        "bootstrap** (B=200) on the difference - the cohorts aren't paired "
        "(VEA classification partitions disjoint rollouts), so each cohort "
        "is resampled independently."
    )
    lines.append("")
    lines.append(
        "| Model | n VEA+ | refusal (VEA+) | n VEA- | refusal (VEA-) | "
        "Δ | 95% CI | Significant? |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | :---: |"
    )
    for row in refusal_split:
        pos = row.get("mean_refusal_vea_pos")
        neg = row.get("mean_refusal_vea_neg")
        delta = row.get("refusal_vea_pos_minus_neg")
        ci = row.get("refusal_vea_pos_minus_neg_ci") or {}
        ci_low = float(ci.get("low", 0.0))
        ci_high = float(ci.get("high", 0.0))
        if delta is None:
            ci_text = "-"
            sig = "n/a"
        else:
            ci_text = f"[{ci_low:+.3f}, {ci_high:+.3f}]"
            sig = "**yes**" if (ci_low > 0 or ci_high < 0) else "no"
        lines.append(
            f"| `{row.get('model_name', '?')}` | "
            f"{row.get('n_vea_pos', 0)} | "
            f"{'-' if pos is None else f'{pos:.3f}'} | "
            f"{row.get('n_vea_neg', 0)} | "
            f"{'-' if neg is None else f'{neg:.3f}'} | "
            f"{'-' if delta is None else f'{delta:+.3f}'} | "
            f"{ci_text} | "
            f"{sig} |"
        )
    lines.append("")

    if empty_trace_breakdown:
        lines.append("## Empty reasoning traces (measurement audit)")
        lines.append("")
        lines.append(
            "VEA classification short-circuits to VEA- when the reasoning "
            "trace is empty (we can't claim verbalization that didn't "
            "happen). A high empty-trace rate for a model means either "
            "extended thinking wasn't actually enabled for that route, "
            "or the model inlined `<think>` into the visible content "
            "(check `response` for that case)."
        )
        lines.append("")
        lines.append("| Model | rollouts | empty traces | empty rate |")
        lines.append("| --- | ---: | ---: | ---: |")
        for row in empty_trace_breakdown:
            lines.append(
                f"| `{row.get('model_name', '?')}` | "
                f"{row.get('total', 0)} | "
                f"{row.get('empty', 0)} | "
                f"{row.get('empty_rate', 0.0) * 100:.1f}% |"
            )
        lines.append("")

    if examples:
        lines.append("## Example VEA+ reasoning quotes")
        lines.append("")
        for ex in examples[:8]:
            lines.append(
                f"- **{ex.get('model_name', '?')}** / "
                f"{ex.get('condition', '?')} / "
                f"{ex.get('task_family', '?')} / "
                f"`{ex.get('example_id', '?')}` "
                f"(vea_type={ex.get('vea_type', '?')}, "
                f"confidence={ex.get('confidence', 0):.2f})"
            )
            quote = (ex.get("quoted_text") or "").strip().replace("\n", " ")
            if quote:
                lines.append(f"  > {quote[:300]}")
            lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="runs/goodfire-vea",
        help=(
            "Run directory (created if missing). Rollouts + classifications "
            "+ report are written here."
        ),
    )
    parser.add_argument(
        "--from-run-dir",
        default=None,
        help=(
            "Skip rollouts; read rollouts.jsonl from this existing run dir "
            "and just run the VEA judge + report on it. Requires that the "
            "records contain the 'reasoning_trace' field (which means the "
            "rollouts must have been produced by post-providers.py-fix code)."
        ),
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=500,
        help=(
            "Total prompts in the corpus. Default 500 matches the v6 setup; "
            "drop to 50 for a cheap smoke test (~$1) before paying for full."
        ),
    )
    parser.add_argument(
        "--safety-only",
        action="store_true",
        help=(
            "Use only the safety corpus (BeaverTails). Goodfire's claim is "
            "most relevant on safety prompts; this can cut cost ~4x."
        ),
    )
    parser.add_argument(
        "--no-network",
        action="store_true",
        help=(
            "Use the synthetic offline corpus (no HuggingFace downloads). "
            "For testing the pipeline without spending HF bandwidth."
        ),
    )
    parser.add_argument(
        "--openrouter-models",
        nargs="*",
        default=list(_DEFAULT_OPEN_REASONING_MODELS),
        help=(
            "OpenRouter open-reasoning model ids. Defaults to the three "
            "visible-CoT models (deepseek-v3.2, qwen3-thinking, kimi-k2-"
            "thinking)."
        ),
    )
    parser.add_argument(
        "--include-opus",
        action="store_true",
        help=(
            "Also include anthropic:claude-opus-4-7 with extended thinking "
            "enabled. Strengthens the replication story with a closed-"
            "source comparison; costs roughly 2-3x per opus rollout vs the "
            "open-reasoning trio."
        ),
    )
    parser.add_argument(
        "--anthropic-model",
        default=_DEFAULT_CLOSED_REASONING_MODEL,
        help="Anthropic reasoning model id used when --include-opus is set.",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=_DEFAULT_THINKING_BUDGET,
        help=(
            "Anthropic extended-thinking token budget. Only used when "
            "--include-opus is set. The visible max_tokens is bumped above "
            "this so the final answer still fits."
        ),
    )
    parser.add_argument(
        "--rollout-concurrency",
        type=int,
        default=32,
        help=(
            "Parallel rollout requests (per-vendor quota applies). Default "
            "32 is safe for OpenRouter; drop to 8-16 if you hit 429s. "
            "Reasoning models are slow (~5-15s/call), so concurrency is the "
            "dominant factor in wall time."
        ),
    )
    parser.add_argument(
        "--vea-judge-provider",
        default="openrouter",
        choices=("openai", "anthropic", "gemini", "openrouter"),
        help=(
            "Provider for the VEA judge. Pick a vendor that did NOT produce "
            "any of the rollouts to avoid same-vendor self-grading bias. "
            "openrouter+google/gemini-2.5-flash-lite is the recommended "
            "default for the open-reasoning trio - same prompting style as "
            "the larger Flash, ~3-5x cheaper, JSON adherence is reliable on "
            "this rubric."
        ),
    )
    parser.add_argument(
        "--vea-judge-model",
        default="google/gemini-2.5-flash-lite",
        help=(
            "VEA judge model id. Default 'google/gemini-2.5-flash-lite' is "
            "the cost-optimized default (~$3-7 per 8000-rollout pass on "
            "OpenRouter). Bump to 'google/gemini-2.5-flash' if you see "
            "elevated parse_error rates or want a stronger judge for the "
            "publishable run. Even cheaper: 'mistralai/ministral-8b-2410' "
            "(~10x cheaper than Flash, lower JSON adherence; smoke-test "
            "before committing)."
        ),
    )
    parser.add_argument(
        "--vea-judge-concurrency",
        type=int,
        default=32,
        help=(
            "Parallel VEA judge calls. Default 32 is safe for "
            "google/gemini-2.5-flash-lite via OpenRouter; drop to 8-16 if "
            "you hit rate limits."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help=(
            "Log a progress line every N classified records (with elapsed, "
            "percentage, ETA, and per-model breakdown). 0 to disable."
        ),
    )
    parser.add_argument(
        "--vea-judge-sample-rate",
        type=float,
        default=1.0,
        help=(
            "Fraction of rollouts to send to the VEA judge (0.0-1.0). "
            "Default 1.0 (judge all). Sampled-out records get vea_type='"
            "skipped' and are excluded from rate aggregation."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out_dir = Path(args.out_dir if not args.from_run_dir else args.from_run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rollouts_path = out_dir / "rollouts.jsonl"

    # 1) Rollouts (skip if --from-run-dir)
    if args.from_run_dir:
        if not rollouts_path.is_file():
            parser.error(
                f"--from-run-dir {args.from_run_dir} has no rollouts.jsonl"
            )
        with rollouts_path.open(encoding="utf-8") as fh:
            rollouts = [json.loads(line) for line in fh if line.strip()]
        LOGGER.info(
            "goodfire-vea: read %d rollouts from %s",
            len(rollouts),
            rollouts_path,
        )
    else:
        # Build the model lineup. Anthropic gets the thinking-enabled wrapper.
        models: List[Any] = []
        for model_id in args.openrouter_models:
            models.append(_build_openrouter_model(model_id))
        if args.include_opus:
            models.append(
                _AnthropicWithThinking(
                    name=f"anthropic:{args.anthropic_model}",
                    base_url=os.environ.get(
                        "ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"
                    ),
                    api_key=os.environ["ANTHROPIC_API_KEY"],
                    model=args.anthropic_model,
                    thinking_budget=args.thinking_budget,
                )
            )
        LOGGER.info(
            "goodfire-vea: %d models in lineup; running rollouts",
            len(models),
        )
        rollouts = _run_rollouts(
            models,
            n_examples=args.n_examples,
            safety_only=args.safety_only,
            allow_network=not args.no_network,
            max_concurrent=args.rollout_concurrency,
            thinking_budget=args.thinking_budget,
            seed=args.seed,
        )
        with rollouts_path.open("w", encoding="utf-8") as fh:
            for r in rollouts:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
        LOGGER.info(
            "goodfire-vea: wrote %d rollouts to %s",
            len(rollouts),
            rollouts_path,
        )

    # 2) VEA classification
    judge = build_default_llm_vea_judge(
        args.vea_judge_provider, args.vea_judge_model
    )
    LOGGER.info(
        "goodfire-vea: classifying %d rollouts (concurrency=%d, sample_rate=%.2f)",
        len(rollouts),
        args.vea_judge_concurrency,
        args.vea_judge_sample_rate,
    )
    started = time.perf_counter()
    classifications = _classify_rollouts(
        rollouts,
        judge=judge,
        max_concurrent=args.vea_judge_concurrency,
        sample_rate=args.vea_judge_sample_rate,
        seed=args.seed,
        progress_every=max(1, args.progress_every) if args.progress_every else 10**9,
    )
    elapsed = time.perf_counter() - started
    LOGGER.info(
        "goodfire-vea: classification done in %.1fs (%d records)",
        elapsed,
        len(classifications),
    )
    classifications_path = out_dir / "goodfire_vea_classifications.jsonl"
    with classifications_path.open("w", encoding="utf-8") as fh:
        for c in classifications:
            fh.write(json.dumps(c, sort_keys=True) + "\n")

    # 3) Aggregation (with bootstrap-paired CIs for the deltas)
    per_model_condition = _vea_rate_by_bucket(
        classifications,
        ("model_name", "condition"),
        bootstrap_seed=args.seed,
    )
    per_model_deltas = _inject_neutral_delta(
        per_model_condition,
        ("model_name",),
        classifications=classifications,
        bootstrap_seed=args.seed,
    )
    refusal_split = _refusal_by_vea(
        classifications, ("model_name",), bootstrap_seed=args.seed
    )

    # Empty-trace audit per model
    empty_counts = Counter()
    total_counts = Counter()
    for c in classifications:
        model = c.get("model_name", "?")
        total_counts[model] += 1
        if int(c.get("reasoning_trace_chars") or 0) == 0:
            empty_counts[model] += 1
    empty_trace_breakdown = sorted(
        (
            {
                "model_name": model,
                "total": total_counts[model],
                "empty": empty_counts[model],
                "empty_rate": (
                    empty_counts[model] / total_counts[model]
                    if total_counts[model]
                    else 0.0
                ),
            }
            for model in total_counts
        ),
        key=lambda r: -r["empty_rate"],
    )

    # VEA+ examples for the report (highest-confidence positives)
    positives = sorted(
        (c for c in classifications if c.get("is_vea")),
        key=lambda c: -float(c.get("confidence") or 0.0),
    )

    summary = {
        "n_records": len(classifications),
        "judge_provider": args.vea_judge_provider,
        "judge_model": args.vea_judge_model,
        "per_model_condition": per_model_condition,
        "per_model_inject_minus_neutral": per_model_deltas,
        "refusal_by_vea_status": refusal_split,
        "empty_trace_breakdown": empty_trace_breakdown,
        "vea_type_counts": dict(
            Counter(
                c.get("vea_type", "unspecified")
                for c in classifications
                if c.get("vea_type") != "skipped"
            )
        ),
    }
    summary_path = out_dir / "goodfire_vea_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    report = _render_report(
        run_dir=out_dir,
        n_records=len(classifications),
        judge_provider=args.vea_judge_provider,
        judge_model=args.vea_judge_model,
        per_model_condition=per_model_condition,
        per_model_deltas=per_model_deltas,
        refusal_split=refusal_split,
        examples=positives,
        empty_trace_breakdown=empty_trace_breakdown,
    )
    report_path = out_dir / "goodfire_vea_report.md"
    report_path.write_text(report)

    LOGGER.info(
        "goodfire-vea: wrote %s, %s, %s",
        classifications_path,
        summary_path,
        report_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
