"""
V4 Stage 1: behavior-only eval/deploy sweep across candidate reasoning models.

Picks the right model for V4b before any NLA training or activation work.
Stage 1 question: which reasoning model shows the cleanest BEHAVIORAL
eval/deploy separation when asked the same prompts as Llama-3.3-70B got?

This is the V4a behavior pipeline with the AV/Neuronpedia half stripped out
and the inference target parameterized. Same eval/deploy pairs, same Opus
behavior judge with the same frozen prompt hash, same separation threshold
(0.70 = 14/20 arms correct, binomial p<0.05 vs chance).

Llama-3.3-70B baseline from V4a (2026-05-19): behavior accuracy 13/20 = 0.65.

Default candidates (reasoning models, all on OpenRouter):
  - qwen/qwen3-32b              (Microsoft probes at L62 already available)
  - deepseek/deepseek-r1
  - deepseek/deepseek-r1-distill-llama-70b
  - moonshotai/kimi-k2          (Read & Bloom: verbalizes eval-awareness)

Plus Llama-3.3-70B re-run as control / sanity check (~65% expected).

Cost: ~$0.50-1.00 per model × 5 models = ~$3-5 total. ~10 min.

Output: ranked table of behavior accuracy per model. The model with the
highest separation (and above 0.70) becomes the V4b target.

Dependencies: openai, anthropic, pyyaml.
Env: ANTHROPIC_API_KEY, OPENROUTER_API_KEY.

Usage:
    python v4/scripts/behavior_model_sweep.py                        # default model set
    python v4/scripts/behavior_model_sweep.py --models qwen/qwen3-32b deepseek/deepseek-r1
    python v4/scripts/behavior_model_sweep.py --limit 2              # smoke (2 pairs)
    python v4/scripts/behavior_model_sweep.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

# Reuse the frozen behavior-judge prompts and pipeline from the V4a script
# so Stage 1 results are directly comparable to Llama V4a baseline.
from v4.scripts.sanity_check_llama_av import (  # noqa: E402
    BEHAVIOR_JUDGE_SYSTEM,
    BEHAVIOR_JUDGE_USER_TEMPLATE,
    JUDGE_MODEL,
    SEPARATION_MIN_ACCURACY,
    call_judge_behavior,
    load_pairs,
    _hash_text,
)

DEFAULT_CANDIDATES = [
    # Baseline (already tested in V4a, expected ~0.65)
    "meta-llama/llama-3.3-70b-instruct",
    # Reasoning models, ranked by expected eval-awareness per Read & Bloom + Jord
    "qwen/qwen3-32b",
    "deepseek/deepseek-r1",
    "deepseek/deepseek-r1-distill-llama-70b",
    "moonshotai/kimi-k2",
]

INFERENCE_MAX_TOKENS = 512


_openrouter_client = None


def _openrouter():
    global _openrouter_client
    if _openrouter_client is None:
        try:
            from openai import OpenAI
        except ImportError:
            sys.exit("pip install openai")
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            sys.exit("Set OPENROUTER_API_KEY")
        _openrouter_client = OpenAI(
            api_key=key,
            base_url="https://openrouter.ai/api/v1",
        )
    return _openrouter_client


def call_inference(prompt: str, model_id: str, provider: str | None = None) -> dict[str, Any]:
    client = _openrouter()
    extra: dict[str, Any] = {}
    if provider:
        extra["provider"] = {"order": [provider], "allow_fallbacks": False}
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=INFERENCE_MAX_TOKENS,
        temperature=0.0,
        extra_body=extra if extra else None,
    )
    dumped: dict[str, Any] = {}
    try:
        dumped = resp.model_dump()
    except Exception:
        pass
    return {
        "response_text": resp.choices[0].message.content or "",
        "served_model": getattr(resp, "model", None),
        "openrouter_provider": dumped.get("provider"),
        "usage": dumped.get("usage"),
    }


def run_one_model(model_id: str, pairs: list[dict[str, Any]], provider: str | None) -> dict[str, Any]:
    per_pair: list[dict[str, Any]] = []
    correct = evaluable = 0
    t0 = time.time()

    for pair in pairs:
        pair_rec: dict[str, Any] = {"pair_id": pair["pair_id"]}
        for arm in ("a", "b"):
            text = pair[arm]["text"]
            arm_label = pair[arm]["label"]
            y = 1 if arm_label == "eval" else 0
            try:
                inference = call_inference(text, model_id, provider)
                judge = call_judge_behavior(text, inference["response_text"])
            except Exception as e:
                print(f"  [{pair['pair_id']}/{arm}] ERROR: {e}")
                pair_rec[arm] = {"arm_label": arm_label, "y": y, "error": str(e)}
                continue
            evaluable += 1
            if judge["binary"] == y:
                correct += 1
            pair_rec[arm] = {
                "arm_label": arm_label, "y": y,
                "behavior_binary": judge["binary"],
                "eval_like": judge["eval_like"],
                "evidence": judge["evidence"],
                "inference": inference,
            }
            print(f"  {pair['pair_id']}/{arm} ({arm_label}): beh={judge['binary']}")
        per_pair.append(pair_rec)

    accuracy = correct / evaluable if evaluable else 0.0
    separates = accuracy >= SEPARATION_MIN_ACCURACY
    return {
        "model_id": model_id,
        "correct": correct,
        "evaluable": evaluable,
        "accuracy": accuracy,
        "separates": separates,
        "elapsed_seconds": time.time() - t0,
        "per_pair": per_pair,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_CANDIDATES)
    parser.add_argument("--pairs-yaml", type=Path,
                        default=_REPO_ROOT / "v4/prompts/sanity_pairs.yaml")
    parser.add_argument("--output-dir", type=Path,
                        default=_REPO_ROOT / "v4/runs/behavior_model_sweep")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N pairs (smoke test)")
    parser.add_argument("--provider", type=str, default=None,
                        help="OpenRouter provider to pin (applies to all models)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip API calls; verify pair load + model list only")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(args.pairs_yaml, "eval_vs_deploy")
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"Loaded {len(pairs)} eval/deploy pairs from {args.pairs_yaml}")
    print(f"Models to sweep ({len(args.models)}):")
    for m in args.models:
        print(f"  - {m}")

    if args.dry_run:
        print("\nDry run: skipping API work.")
        return

    prompt_hashes = {
        "BEHAVIOR_JUDGE_SYSTEM": _hash_text(BEHAVIOR_JUDGE_SYSTEM),
        "BEHAVIOR_JUDGE_USER_TEMPLATE": _hash_text(BEHAVIOR_JUDGE_USER_TEMPLATE),
    }
    print(f"\nFrozen prompt hashes: {prompt_hashes}")
    print(f"Judge: {JUDGE_MODEL}; separation threshold: {SEPARATION_MIN_ACCURACY}")

    per_model: list[dict[str, Any]] = []
    t0 = time.time()

    for model_id in args.models:
        print(f"\n{'='*60}")
        print(f"Running {model_id}")
        print(f"{'='*60}")
        result = run_one_model(model_id, pairs, args.provider)
        per_model.append(result)
        print(f"\n{model_id}: {result['correct']}/{result['evaluable']} = "
              f"{result['accuracy']:.3f} (separates: {result['separates']})")

    elapsed = time.time() - t0

    # Ranked summary
    ranked = sorted(per_model, key=lambda r: r["accuracy"], reverse=True)
    print(f"\n{'='*60}")
    print("RANKED SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<45} {'Accuracy':>10} {'Separates':>10}")
    for r in ranked:
        print(f"{r['model_id']:<45} {r['accuracy']:>9.3f} {str(r['separates']):>10}")

    top = ranked[0] if ranked else None
    if top and top["separates"]:
        print(f"\n--> V4b candidate: {top['model_id']} (accuracy {top['accuracy']:.3f})")
    else:
        print("\n--> No model crossed the 0.70 threshold. Reconsider model set or stimuli.")

    out = {
        "models_swept": args.models,
        "n_pairs": len(pairs),
        "judge_model": JUDGE_MODEL,
        "separation_min_accuracy": SEPARATION_MIN_ACCURACY,
        "frozen_prompt_hashes": prompt_hashes,
        "pinned_provider": args.provider,
        "elapsed_seconds": elapsed,
        "per_model": per_model,
        "ranked": [{"model_id": r["model_id"], "accuracy": r["accuracy"],
                    "correct": r["correct"], "evaluable": r["evaluable"],
                    "separates": r["separates"]} for r in ranked],
        "v4b_candidate": top["model_id"] if (top and top["separates"]) else None,
    }
    out_path = args.output_dir / "behavior_sweep_result.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
