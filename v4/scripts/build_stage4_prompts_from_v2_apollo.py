"""
Stage 4 prompt assembly from V2 Apollo rollouts.

V1 rollouts (BeaverTails/MMLU/GSM8K) aren't on this machine. V2 Apollo
rollouts ARE here, so for the initial Stage 4 pipeline validation we
use those: alignment-family prompts in the "neutral" condition (no
inject framing), which is the deploy-equivalent natural prompt.

Each unique example_id appears multiple times in the rollouts (once
per model × condition). We deduplicate on example_id, keep only
condition="neutral", and emit a JSONL with the schema expected by
run_steered_generation.py.

Output JSONL fields per record:
    prompt_id      : example_id from V2 (apollo:<...>:<id>)
    text           : the user prompt text
    family         : "apollo_alignment"  (single-family for now)
    v1_condition   : "deploy"           (V2 neutral ≡ V1 deploy framing)
    apollo_category, apollo_subcategory, apollo_is_eval : preserved for analysis

Usage:
    python build_stage4_prompts_from_v2_apollo.py \\
        --rollouts runs/v2-apollo-factorial-prereg/rollouts.jsonl \\
        --limit 125 \\
        --output v4/runs/stage4/v1_prompts_apollo_125.jsonl

Or pull ALL unique Apollo neutral prompts (typical: ~125 of them in V2):
    python build_stage4_prompts_from_v2_apollo.py \\
        --rollouts runs/v2-apollo-factorial-prereg/rollouts.jsonl \\
        --output v4/runs/stage4/v1_prompts_apollo_all.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True,
                        help="Path to V2 Apollo rollouts.jsonl")
    parser.add_argument("--limit", type=int, default=None,
                        help="Take only first N unique example_ids (after dedup)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    print(f"Reading {args.rollouts}")
    seen: dict[str, dict] = {}
    n_lines = 0
    n_neutral = 0
    with open(args.rollouts) as f:
        for line in f:
            r = json.loads(line)
            n_lines += 1
            if r.get("condition") != "neutral":
                continue
            n_neutral += 1
            eid = r.get("example_id")
            if not eid or eid in seen:
                continue
            seen[eid] = {
                "prompt_id": eid,
                "text": r["prompt"],
                "family": "apollo_alignment",
                "v1_condition": "deploy",
                "apollo_category": r.get("apollo_category", ""),
                "apollo_subcategory": r.get("apollo_subcategory", ""),
                "apollo_is_eval": r.get("apollo_is_eval", ""),
                "system_prompt": r.get("system_prompt", ""),
            }
    print(f"  read {n_lines} rollouts, {n_neutral} in 'neutral' condition, "
          f"{len(seen)} unique example_ids")

    records = list(seen.values())
    if args.limit:
        records = records[: args.limit]
        print(f"  taking first {args.limit}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fout:
        for r in records:
            fout.write(json.dumps(r) + "\n")
    print(f"\nSaved {len(records)} prompts to {args.output}")
    print(f"  text length: min={min(len(r['text']) for r in records)} "
          f"max={max(len(r['text']) for r in records)} "
          f"median={sorted(len(r['text']) for r in records)[len(records)//2]}")


if __name__ == "__main__":
    main()
