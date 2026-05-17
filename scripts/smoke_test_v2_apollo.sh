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
#   # Default: 1 OpenRouter model (qwen3) only — fastest, cheapest
#   APOLLO_DATA=~/data/apollo/dataset.json \
#   OPENROUTER_API_KEY=sk-or-... \
#   bash scripts/smoke_test_v2_apollo.sh
#
#   # Verify opus thinking config: 1 Anthropic model (opus) only
#   APOLLO_DATA=~/data/apollo/dataset.json \
#   ANTHROPIC_API_KEY=sk-ant-... \
#   SMOKE_MODE=opus bash scripts/smoke_test_v2_apollo.sh
#
#   # Both models (rare; only needed if you change the inject prompt)
#   APOLLO_DATA=~/data/apollo/dataset.json \
#   OPENROUTER_API_KEY=sk-or-... ANTHROPIC_API_KEY=sk-ant-... \
#   SMOKE_MODE=both bash scripts/smoke_test_v2_apollo.sh

set -euo pipefail

# --- prereqs ---------------------------------------------------------------

APOLLO_DATA="${APOLLO_DATA:-$HOME/data/apollo/dataset.json}"
# SMOKE_MODE: qwen (default, OpenRouter only) | opus (Anthropic only) | both
SMOKE_MODE="${SMOKE_MODE:-qwen}"
OUT_DIR="${OUT_DIR:-runs/v2-apollo-smoke-${SMOKE_MODE}}"

if [ ! -f "$APOLLO_DATA" ]; then
    echo "ERROR: Apollo dataset not found at $APOLLO_DATA" >&2
    echo "  Download from https://huggingface.co/datasets/jjpn2/eval_awareness" >&2
    echo "  Then set APOLLO_DATA=/path/to/dataset.json" >&2
    exit 1
fi

case "$SMOKE_MODE" in
    qwen|both)
        if [ -z "${OPENROUTER_API_KEY:-}" ]; then
            echo "ERROR: OPENROUTER_API_KEY must be set for SMOKE_MODE=$SMOKE_MODE" >&2
            exit 1
        fi
        ;;
esac

case "$SMOKE_MODE" in
    opus|both)
        if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
            echo "ERROR: ANTHROPIC_API_KEY must be set for SMOKE_MODE=$SMOKE_MODE" >&2
            exit 1
        fi
        ;;
esac

# Pick the model flags for run_v2_apollo_factorial.py based on SMOKE_MODE
MODEL_FLAGS=()
EXPECTED_MODELS=1
case "$SMOKE_MODE" in
    qwen)
        MODEL_FLAGS=(--openrouter-models qwen/qwen3-235b-a22b-thinking-2507)
        ;;
    opus)
        MODEL_FLAGS=(
            --openrouter-models  # empty list (drop all OpenRouter models)
            --include-opus
            --anthropic-model claude-opus-4-7
        )
        ;;
    both)
        MODEL_FLAGS=(
            --openrouter-models qwen/qwen3-235b-a22b-thinking-2507
            --include-opus
            --anthropic-model claude-opus-4-7
        )
        EXPECTED_MODELS=2
        ;;
    *)
        echo "ERROR: SMOKE_MODE must be one of: qwen | opus | both (got $SMOKE_MODE)" >&2
        exit 1
        ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EXPECTED_ROLLOUTS=$((10 * 2 * EXPECTED_MODELS))  # 10 examples × 2 conditions × N models

echo "==> Apollo dataset: $APOLLO_DATA"
echo "==> Apollo sha256:  $(shasum -a 256 "$APOLLO_DATA" | awk '{print $1}')"
echo "==> Output dir:     $OUT_DIR"
echo "==> Smoke mode:     $SMOKE_MODE ($EXPECTED_MODELS model(s))"
echo "==> Plan:           5 eval + 5 deployment prompts × 2 conditions × $EXPECTED_MODELS models = $EXPECTED_ROLLOUTS rollouts"
echo "==> Estimated:      ~\$$EXPECTED_MODELS, ~5 min"
echo

# --- 1) rollouts ----------------------------------------------------------

echo "==> [1/3] Running rollouts..."
python scripts/run_v2_apollo_factorial.py \
    --apollo-dataset-path "$APOLLO_DATA" \
    --n-eval 5 --n-deployment 5 \
    --prompt-mode full_transcript \
    "${MODEL_FLAGS[@]}" \
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
    EXPECTED_ROLLOUTS="$EXPECTED_ROLLOUTS" EXPECTED_MODELS="$EXPECTED_MODELS" \
    python - <<'PYEOF'
import json, os, sys

rollouts = [json.loads(l) for l in open(os.environ["ROLLOUTS_PATH"]) if l.strip()]
cls = [json.loads(l) for l in open(os.environ["CLS_PATH"]) if l.strip()]
summary = json.loads(open(os.environ["SUMMARY_PATH"]).read())
expected_rollouts = int(os.environ["EXPECTED_ROLLOUTS"])
expected_models = int(os.environ["EXPECTED_MODELS"])

errors = []

if len(rollouts) != expected_rollouts:
    errors.append(f"expected {expected_rollouts} rollouts, got {len(rollouts)}")

if len(cls) != expected_rollouts:
    errors.append(f"expected {expected_rollouts} VEA classifications, got {len(cls)}")

required_rollout_keys = {
    "model_name", "condition", "example_id", "apollo_id",
    "apollo_is_eval", "apollo_eval_type", "apollo_category",
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

if not summary.get("per_model") or len(summary["per_model"]) != expected_models:
    errors.append(
        f"summary has {len(summary.get('per_model', []))} per_model entries, expected {expected_models}"
    )

for pm in summary.get("per_model", []):
    cells = pm.get("cells", {})
    expected_cells = {"neutral_eval", "inject_eval", "neutral_deployment", "inject_deployment"}
    if set(cells.keys()) != expected_cells:
        errors.append(f"[{pm['model_name']}] summary cells {set(cells.keys())} != {expected_cells}")
    for k, c in cells.items():
        if c.get("n", 0) != 5:
            errors.append(f"[{pm['model_name']}] cell {k} has n={c.get('n')}, expected 5")
        if "vea_mean" not in c:
            errors.append(f"[{pm['model_name']}] cell {k} missing vea_mean (VEA join failed?)")
    if pm.get("verdict") == "insufficient_data":
        errors.append(f"[{pm['model_name']}] verdict is insufficient_data")

# Per-model trace check — every model in the lineup should have at least
# some non-empty reasoning_trace. If a model returns 0 traces it's a
# provider config bug (e.g. opus without --thinking-effort).
by_model_traces = {}
for r in rollouts:
    m = r["model_name"]
    by_model_traces.setdefault(m, 0)
    if r.get("reasoning_trace"):
        by_model_traces[m] += 1
for m, n in by_model_traces.items():
    if n == 0:
        errors.append(
            f"[{m}] zero rollouts have a reasoning_trace -- provider/thinking config likely wrong"
        )

refusal_vals = {r["refusal"] for r in rollouts}
if not refusal_vals.issubset({0.0, 1.0}):
    errors.append(f"refusal not binary: {sorted(refusal_vals)}")

if errors:
    print("FAIL:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    trace_summary = ", ".join(f"{m.split(':')[-1]}={n}" for m, n in by_model_traces.items())
    print(f"PASS: {expected_rollouts} rollouts + classifications + cells/verdict ok. traces: {trace_summary}")
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
