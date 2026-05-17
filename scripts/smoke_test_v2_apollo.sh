#!/usr/bin/env bash
# Tiny end-to-end smoke test of the V2 Apollo pipeline.
#
# Runs 5 eval-labeled + 5 deployment-labeled Apollo prompts through 1 model
# (qwen3-235b-thinking) under both extrinsic conditions, judges them for VEA,
# and runs the analysis script. Confirms the loader handles real Apollo data,
# the runner produces well-formed rollouts.jsonl, the VEA judge works, and
# the analysis script produces a summary + report.
#
# Cost: ~$1. Wall time: ~5 min.
#
# This is NOT a confirmatory run. The pre-registered confirmatory command is
# in docs/pre_registration_v2_apollo_2x2.md §6.
#
# Usage:
#   APOLLO_DATA=~/data/eval_awareness_val.json \
#   OPENROUTER_API_KEY=sk-or-... \
#   bash scripts/smoke_test_v2_apollo.sh

set -euo pipefail

# --- prereqs ---------------------------------------------------------------

APOLLO_DATA="${APOLLO_DATA:-$HOME/data/apollo/dataset.json}"
OUT_DIR="${OUT_DIR:-runs/v2-apollo-smoke}"

if [ ! -f "$APOLLO_DATA" ]; then
    echo "ERROR: Apollo dataset not found at $APOLLO_DATA" >&2
    echo "  Download from https://huggingface.co/datasets/jjpn2/eval_awareness" >&2
    echo "  Then set APOLLO_DATA=/path/to/eval_awareness_val.json" >&2
    exit 1
fi

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "ERROR: OPENROUTER_API_KEY must be set" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Apollo dataset: $APOLLO_DATA"
echo "==> Apollo sha256:  $(shasum -a 256 "$APOLLO_DATA" | awk '{print $1}')"
echo "==> Output dir:     $OUT_DIR"
echo "==> Model:          qwen/qwen3-235b-a22b-thinking-2507"
echo "==> Plan:           5 eval + 5 deployment prompts × 2 conditions = 20 rollouts"
echo "==> Estimated:      ~\$1, ~5 min"
echo

# --- 1) rollouts ----------------------------------------------------------

echo "==> [1/3] Running rollouts..."
python scripts/run_v2_apollo_factorial.py \
    --apollo-dataset-path "$APOLLO_DATA" \
    --n-eval 5 --n-deployment 5 \
    --prompt-mode full_transcript \
    --openrouter-models qwen/qwen3-235b-a22b-thinking-2507 \
    --rollout-concurrency 8 \
    --out-dir "$OUT_DIR" \
    --seed 0

# --- 2) VEA judging --------------------------------------------------------

echo
echo "==> [2/3] Running VEA judge..."
python scripts/run_goodfire_vea.py \
    --from-run-dir "$OUT_DIR" \
    --vea-judge-provider openrouter \
    --vea-judge-model google/gemini-2.5-flash-lite \
    --vea-judge-concurrency 8 \
    --seed 0

# --- 3) analysis ----------------------------------------------------------

echo
echo "==> [3/3] Running analysis (strict VEA, B=500, min_cell_n=3)..."
python scripts/analyze_v2_factorial.py \
    --run-dir "$OUT_DIR" \
    --classifications-path "$OUT_DIR/goodfire_vea_classifications.jsonl" \
    --strict-vea \
    --bootstrap-samples 500 \
    --min-cell-n 3 \
    --seed 0 \
    --out-suffix .smoke

# --- assertions -----------------------------------------------------------

echo
echo "==> Validating outputs..."
ROLLOUTS_PATH="$OUT_DIR/rollouts.jsonl"
CLS_PATH="$OUT_DIR/goodfire_vea_classifications.jsonl"
SUMMARY_PATH="$OUT_DIR/v2_apollo_factorial_summary.smoke.json"
REPORT_PATH="$OUT_DIR/v2_apollo_factorial_report.smoke.md"

OUT_DIR="$OUT_DIR" SUMMARY_PATH="$SUMMARY_PATH" \
    ROLLOUTS_PATH="$ROLLOUTS_PATH" CLS_PATH="$CLS_PATH" \
    python - <<'PYEOF'
import json, os, sys

rollouts = [json.loads(l) for l in open(os.environ["ROLLOUTS_PATH"]) if l.strip()]
cls = [json.loads(l) for l in open(os.environ["CLS_PATH"]) if l.strip()]
summary = json.loads(open(os.environ["SUMMARY_PATH"]).read())

errors = []

if len(rollouts) != 20:
    errors.append(f"expected 20 rollouts, got {len(rollouts)}")

if len(cls) != 20:
    errors.append(f"expected 20 VEA classifications, got {len(cls)}")

required_rollout_keys = {
    "model_name", "condition", "example_id", "apollo_id",
    "apollo_is_eval", "apollo_eval_category", "apollo_category",
    "response", "refusal",
}
missing = required_rollout_keys - set(rollouts[0].keys()) if rollouts else required_rollout_keys
if missing:
    errors.append(f"rollouts missing fields: {sorted(missing)}")

conditions = {r["condition"] for r in rollouts}
if conditions != {"neutral", "inject"}:
    errors.append(f"expected conditions {{neutral, inject}}, got {conditions}")

labels = {r["apollo_is_eval"] for r in rollouts}
if labels != {"true", "false"}:
    errors.append(f"expected labels {{true, false}}, got {labels}")

if not summary.get("per_model"):
    errors.append("summary has no per_model entries")
else:
    pm = summary["per_model"][0]
    cells = pm.get("cells", {})
    expected_cells = {"neutral_eval", "inject_eval", "neutral_deployment", "inject_deployment"}
    if set(cells.keys()) != expected_cells:
        errors.append(f"summary cells {set(cells.keys())} != {expected_cells}")
    for k, c in cells.items():
        if c.get("n", 0) != 5:
            errors.append(f"cell {k} has n={c.get('n')}, expected 5")
        if "vea_mean" not in c:
            errors.append(f"cell {k} missing vea_mean (VEA join failed?)")
    if pm.get("verdict") == "insufficient_data":
        errors.append("verdict is insufficient_data — VEA join or cell count broke")

# trace presence check — for reasoning models, at least some traces should be non-empty
n_with_trace = sum(1 for r in rollouts if r.get("reasoning_trace"))
if n_with_trace == 0:
    errors.append("no rollout has a reasoning_trace — provider config likely wrong")

# refusal scoring sanity
refusal_vals = {r["refusal"] for r in rollouts}
if not refusal_vals.issubset({0.0, 1.0}):
    errors.append(f"refusal not binary: {sorted(refusal_vals)}")

if errors:
    print("FAIL:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("PASS: all 20 rollouts + 20 classifications + 4 populated cells + verdict produced")
PYEOF

echo
echo "==> Smoke test complete."
echo "    Report:  $REPORT_PATH"
echo "    Summary: $SUMMARY_PATH"
echo
echo "    Inspect the report:"
echo "        cat $REPORT_PATH"
echo
echo "    If everything looks right, fire the pre-registered run from"
echo "    docs/pre_registration_v2_apollo_2x2.md §6 next."
