"""V2 Apollo 2x2 factorial runner.

Runs the V2 experiment described in ``docs/v2_apollo_design.md``:

  factor A (intrinsic): Apollo label - ``eval`` vs ``deployment``
  factor B (extrinsic): system-prompt framing - ``neutral`` vs ``inject``

For each (model, example, condition) triple this script produces one rollout
and appends it to ``rollouts.jsonl`` in the output directory. The downstream
analysis script (PR 3) computes cell means, main effects, and the interaction
effect with paired-bootstrap CIs.

This script only handles rollouts. It does NOT compute aggregates or call the
VEA judge - those run separately so re-analysis is cheap (judge cost only).

Typical usage::

    python scripts/run_v2_apollo_factorial.py \\
      --apollo-dataset-path /path/to/eval_awareness_val.json \\
      --n-eval 100 --n-deployment 100 \\
      --openrouter-models qwen/qwen3-235b-a22b-thinking-2507 \\
      --include-opus \\
      --out-dir runs/v2-apollo-factorial-pilot \\
      --seed 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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
from src.eval_awareness.dataset import EvalAwarenessExample  # noqa: E402
from src.eval_awareness.dataset_loaders import (  # noqa: E402
    ApolloEvalAwarenessLoader,
    ApolloPromptMode,
)

# Reuse the V1 system-prompt constants + the response/refusal helpers so the
# extrinsic framing is byte-identical to V1 and refusal scoring stays consistent.
from scripts.run_cross_protocol_comparison import (  # noqa: E402
    ARXIV_INJECT_SYSTEM,
    ARXIV_NEUTRAL_SYSTEM,
    _refusal_score,
    _safe_call,
)
# Reuse V1's extended-thinking wrapper so opus-4.7 actually returns
# reasoning blocks. Without it the bare AnthropicClient gets text-only
# responses, which silently breaks the VEA analysis on opus rollouts
# (reasoning_trace empty -> VEA judge short-circuits to no_trace -> opus
# VEA cells in the analysis are all 0).
from scripts.run_goodfire_vea import _AnthropicWithThinking  # noqa: E402

LOGGER = logging.getLogger("v2_apollo_factorial")

CONDITIONS: Tuple[str, ...] = ("neutral", "inject")
_CONDITION_SYSTEM_PROMPT = {
    "neutral": ARXIV_NEUTRAL_SYSTEM,
    "inject": ARXIV_INJECT_SYSTEM,
}


# ---------------------------------------------------------------------------
# Transcript construction
# ---------------------------------------------------------------------------


def _build_condition_transcript(
    example: EvalAwarenessExample, condition: str
) -> TranscriptInput:
    """Overlay a condition's extrinsic framing onto an Apollo example.

    If the Apollo entry carried a system message (some entries do), the
    condition's framing is prepended so the extrinsic cue sets context
    before the source-dataset's own system context. ``conversation_history``
    is preserved verbatim so multi-turn prompts flow through unchanged in
    ``full_transcript`` mode.
    """
    base_system = example.transcript.system_prompt
    framing = _CONDITION_SYSTEM_PROMPT[condition]
    if base_system:
        combined_system = f"{framing}\n\n{base_system}"
    else:
        combined_system = framing
    return TranscriptInput(
        user_prompt=example.transcript.user_prompt,
        system_prompt=combined_system,
        conversation_history=list(example.transcript.conversation_history),
        visible_metadata=dict(example.transcript.visible_metadata),
        task_format=example.transcript.task_format,
    )


# ---------------------------------------------------------------------------
# Task plumbing
# ---------------------------------------------------------------------------


@dataclass
class _FactorialTask:
    model: Any
    model_name: str
    condition: str
    example: EvalAwarenessExample
    transcript: TranscriptInput
    config: GenerationConfig


def _build_tasks(
    models: Sequence[Any],
    examples: Sequence[EvalAwarenessExample],
    *,
    conditions: Sequence[str] = CONDITIONS,
    seed: int = 0,
) -> List[_FactorialTask]:
    tasks: List[_FactorialTask] = []
    for model in models:
        model_name = getattr(model, "name", model.__class__.__name__)
        for example in examples:
            for condition in conditions:
                tasks.append(
                    _FactorialTask(
                        model=model,
                        model_name=model_name,
                        condition=condition,
                        example=example,
                        transcript=_build_condition_transcript(example, condition),
                        config=GenerationConfig(
                            temperature=0.7,
                            max_tokens=512,
                            seed=seed,
                        ),
                    )
                )
    return tasks


def _run_task(task: _FactorialTask) -> Tuple[_FactorialTask, str, float]:
    response = _safe_call(task.model, task.transcript, task.config)
    return task, response, _refusal_score(response)


def _rollout_record(
    task: _FactorialTask, response: str, refusal: float
) -> Dict[str, Any]:
    last_response = getattr(task.model, "last_response", None)
    returned_model = getattr(last_response, "model", "") or "" if last_response else ""
    reasoning_trace = (
        getattr(last_response, "reasoning_trace", "") or "" if last_response else ""
    )
    metadata = task.example.metadata
    return {
        "model_name": task.model_name,
        "model_returned": returned_model,
        "condition": task.condition,
        "example_id": task.example.example_id,
        "task_family": task.example.task_family,
        "apollo_id": metadata.get("apollo_id", ""),
        "apollo_is_eval": metadata.get("apollo_is_eval", ""),
        "apollo_eval_category": metadata.get("apollo_eval_category", ""),
        "apollo_category": metadata.get("apollo_category", ""),
        "apollo_prompt_mode": metadata.get("apollo_prompt_mode", ""),
        "prompt": task.transcript.user_prompt,
        "system_prompt": task.transcript.system_prompt,
        "response": response,
        "reasoning_trace": reasoning_trace,
        "refusal": refusal,
    }


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------


def run_factorial(
    models: Sequence[Any],
    examples: Sequence[EvalAwarenessExample],
    *,
    conditions: Sequence[str] = CONDITIONS,
    max_concurrent: int = 16,
    seed: int = 0,
    progress_every: int = 25,
) -> List[Dict[str, Any]]:
    """Run the 2x2 factorial across (model, example, condition) and return rollouts."""
    tasks = _build_tasks(models, examples, conditions=conditions, seed=seed)
    LOGGER.info(
        "v2-apollo: %d tasks across %d models x %d examples x %d conditions (max_concurrent=%d)",
        len(tasks),
        len(models),
        len(examples),
        len(conditions),
        max_concurrent,
    )

    rollouts: List[Dict[str, Any]] = []
    lock = threading.Lock()
    completed = 0
    started = time.perf_counter()

    def _record(task: _FactorialTask, response: str, refusal: float) -> None:
        nonlocal completed
        rollouts.append(_rollout_record(task, response, refusal))
        completed += 1
        if completed % progress_every == 0:
            elapsed = time.perf_counter() - started
            pct = 100 * completed / max(1, len(tasks))
            rate = completed / max(1e-6, elapsed)
            remaining = (len(tasks) - completed) / max(1e-6, rate)
            LOGGER.info(
                "v2-apollo: %d/%d (%.1f%%, %.1fs elapsed, ~%.0fs left)",
                completed,
                len(tasks),
                pct,
                elapsed,
                remaining,
            )

    if max_concurrent <= 1:
        for task in tasks:
            _, response, refusal = _run_task(task)
            _record(task, response, refusal)
    else:
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = [executor.submit(_run_task, task) for task in tasks]
            for future in as_completed(futures):
                try:
                    task, response, refusal = future.result()
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("v2-apollo task failed (skipped): %s", exc)
                    continue
                with lock:
                    _record(task, response, refusal)
    return rollouts


def write_rollouts_jsonl(rollouts: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in rollouts:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def load_apollo_examples(
    dataset_path: Path,
    *,
    n_eval: int,
    n_deployment: int,
    prompt_mode: ApolloPromptMode,
    seed: int,
) -> List[EvalAwarenessExample]:
    """Load a deterministic stratified subsample: n_eval eval + n_deployment deployment."""
    eval_examples = ApolloEvalAwarenessLoader(
        dataset_path,
        label_filter="eval",
        prompt_mode=prompt_mode,
        max_examples=n_eval,
        seed=seed,
    ).load()
    deployment_examples = ApolloEvalAwarenessLoader(
        dataset_path,
        label_filter="deployment",
        prompt_mode=prompt_mode,
        max_examples=n_deployment,
        seed=seed,
    ).load()
    LOGGER.info(
        "v2-apollo: loaded %d eval + %d deployment examples (mode=%s, seed=%d)",
        len(eval_examples),
        len(deployment_examples),
        prompt_mode,
        seed,
    )
    return [*eval_examples, *deployment_examples]


# ---------------------------------------------------------------------------
# Model construction (mirrors run_goodfire_vea.py)
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


def _build_anthropic_model(
    model_id: str,
    *,
    thinking_budget: int,
    thinking_effort: Optional[str],
    thinking_display: str,
) -> AnthropicClient:
    """Build an Anthropic client with extended thinking enabled.

    opus-4.7 requires this; without it the model returns text-only
    responses and the VEA judge sees empty reasoning_trace on every
    opus rollout.
    """
    return _AnthropicWithThinking(
        name=f"anthropic:{model_id}",
        base_url=os.environ.get(
            "ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"
        ),
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model=model_id,
        thinking_budget=thinking_budget,
        thinking_effort=thinking_effort,
        thinking_display=thinking_display,
    )


_DEFAULT_OPENROUTER_MODELS: Tuple[str, ...] = (
    "qwen/qwen3-235b-a22b-thinking-2507",
    "deepseek/deepseek-v3.2-exp",
    "moonshotai/kimi-k2-thinking",
)
_DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-7"
# Matches the V1 defaults in scripts/run_goodfire_vea.py — keep in sync
# so opus rollouts in V2 use the same thinking config as the V1 paper.
_DEFAULT_THINKING_BUDGET = 4096
_DEFAULT_THINKING_EFFORT = "max"
_DEFAULT_THINKING_DISPLAY = "summarized"


def _build_model_lineup(
    openrouter_models: Sequence[str],
    include_opus: bool,
    anthropic_model: str,
    *,
    thinking_budget: int = _DEFAULT_THINKING_BUDGET,
    thinking_effort: Optional[str] = _DEFAULT_THINKING_EFFORT,
    thinking_display: str = _DEFAULT_THINKING_DISPLAY,
) -> List[Any]:
    models: List[Any] = [
        _build_openrouter_model(model_id) for model_id in openrouter_models
    ]
    if include_opus:
        models.append(
            _build_anthropic_model(
                anthropic_model,
                thinking_budget=thinking_budget,
                thinking_effort=thinking_effort,
                thinking_display=thinking_display,
            )
        )
    return models


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--apollo-dataset-path",
        type=Path,
        required=True,
        help="Path to Apollo's eval_awareness_val.json (download from HuggingFace).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write rollouts.jsonl (and any run metadata).",
    )
    parser.add_argument(
        "--n-eval",
        type=int,
        default=100,
        help="Number of eval-labeled Apollo examples to include.",
    )
    parser.add_argument(
        "--n-deployment",
        type=int,
        default=100,
        help="Number of deployment-labeled Apollo examples to include.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("full_transcript", "last_user_turn"),
        default="full_transcript",
        help="How to render Apollo transcripts into the rollout input.",
    )
    parser.add_argument(
        "--openrouter-models",
        nargs="*",
        default=list(_DEFAULT_OPENROUTER_MODELS),
        help="OpenRouter model ids to include in the lineup.",
    )
    parser.add_argument(
        "--include-opus",
        action="store_true",
        help="Add claude-opus-4-7 via the Anthropic API to the lineup.",
    )
    parser.add_argument(
        "--anthropic-model",
        default=_DEFAULT_ANTHROPIC_MODEL,
        help="Anthropic model id used when --include-opus is set.",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=_DEFAULT_THINKING_BUDGET,
        help="Anthropic extended-thinking token budget (legacy API).",
    )
    parser.add_argument(
        "--thinking-effort",
        choices=("low", "medium", "high", "xhigh", "max"),
        default=_DEFAULT_THINKING_EFFORT,
        help="Anthropic adaptive-thinking effort (opus-4.7 API).",
    )
    parser.add_argument(
        "--thinking-display",
        choices=("summarized", "omitted"),
        default=_DEFAULT_THINKING_DISPLAY,
        help="Anthropic thinking display mode (opus-4.7 API).",
    )
    parser.add_argument(
        "--rollout-concurrency",
        type=int,
        default=16,
        help="Number of in-flight rollouts (ThreadPoolExecutor workers).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Log progress every N completed tasks.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (INFO, DEBUG, WARNING, ...).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    examples = load_apollo_examples(
        args.apollo_dataset_path,
        n_eval=args.n_eval,
        n_deployment=args.n_deployment,
        prompt_mode=args.prompt_mode,
        seed=args.seed,
    )
    if not examples:
        LOGGER.error("no Apollo examples loaded; aborting")
        return 1

    models = _build_model_lineup(
        args.openrouter_models,
        include_opus=args.include_opus,
        anthropic_model=args.anthropic_model,
        thinking_budget=args.thinking_budget,
        thinking_effort=args.thinking_effort,
        thinking_display=args.thinking_display,
    )
    if not models:
        LOGGER.error("no models in the lineup; aborting")
        return 1

    rollouts = run_factorial(
        models,
        examples,
        max_concurrent=args.rollout_concurrency,
        seed=args.seed,
        progress_every=args.progress_every,
    )

    out_path = args.out_dir / "rollouts.jsonl"
    write_rollouts_jsonl(rollouts, out_path)
    LOGGER.info(
        "v2-apollo: wrote %d rollouts to %s", len(rollouts), out_path
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
