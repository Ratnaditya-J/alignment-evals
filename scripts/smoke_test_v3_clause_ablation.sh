#!/usr/bin/env bash
# Tiny end-to-end smoke test of the V3 inject-clause ablation pipeline.
#
# Runs 5 alignment-eval Apollo prompts through 1-4 models under the 5 V3
# conditions (neutral, eval_only, reviewer_only, comparison_only,
# full_inject), judges them for VEA, and runs the analysis script.
# Confirms the V3 runner produces well-formed rollouts.jsonl, the VEA
# judge tolerates the new condition labels, the analyzer's attribution
# decision tree runs without error, and the JSON + MD outputs land.
#
# Cost: ~$0.50-2.50 depending on SMOKE_MODE. Wall time: ~5 min.
#
# This is NOT a confirmatory run. The pre-registered confirmatory command
# will live in docs/pre_registration_v3_clause_ablation.md (PR 4).
#
# Usage:
#   # Default: qwen3 only (cheapest)
#   APOLLO_DATA=~/data/apollo/dataset.json \
#   OPENROUTER_API_KEY=sk-or-... \
#   bash scripts/smoke_test_v3_clause_ablation.sh
#
#   # Validate opus thinking config: opus only
#   APOLLO_DATA=~/data/apollo/dataset.json \
#   ANTHROPIC_API_KEY=sk-ant-... \
#   SMOKE_MODE=opus bash scripts/smoke_test_v3_clause_ablation.sh
#
#   # Headline lineup: qwen3 + opus
#   APOLLO_DATA=~/data/apollo/dataset.json \
#   OPENROUTER_API_KEY=sk-or-... ANTHROPIC_API_KEY=sk-ant-... \
#   SMOKE_MODE=both bash scripts/smoke_test_v3_clause_ablation.sh
#
#   # Full lineup including null baselines (only needed once to validate
#   # --include-baselines plumbing)
#   APOLLO_DATA=~/data/apollo/dataset.json \
#   OPENROUTER_API_KEY=sk-or-... ANTHROPIC_API_KEY=sk-ant-... \
#   SMOKE_MODE=baselines bash scripts/smoke_test_v3_clause_ablation.sh

set -euo pipefail

# --- prereqs ---------------------------------------------------------------

APOLLO_DATA="${APOLLO_DATA:-$HOME/data/apollo/dataset.json}"
# SMOKE_MODE: qwen (default) | opus | both | baselines
SMOKE_MODE="${SMOKE_MODE:-qwen}"
OUT_DIR="${OUT_DIR:-runs/v3-clause-ablation-smoke-${SMOKE_MODE}}"
N_PROMPTS="${N_PROMPTS:-5}"

if [ ! -f "$APOLLO_DATA" ]; then
    echo "ERROR: Apollo dataset not found at $APOLLO_DATA" >&2
    echo "  Download from https://huggingface.co/datasets/jjpn2/eval_awareness" >&2
    echo "  Then set APOLLO_DATA=/path/to/dataset.json" >&2
    exit 1
fi

case "$SMOKE_MODE" in
    qwen|both|baselines)
        if [ -z "${OPENROUTER_API_KEY:-}" ]; then
            echo "ERROR: OPENROUTER_API_KEY must be set for SMOKE_MODE=$SMOKE_MODE" >&2
            exit 1
        fi
        ;;
esac

case "$SMOKE_MODE" in
    opus|both|baselines)
        if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
            echo "ERROR: ANTHROPIC_API_KEY must be set for SMOKE_MODE=$SMOKE_MODE" >&2
            exit 1
        fi
        ;;
esac

# Pick the model flags for run_v3_clause_ablation.py based on SMOKE_MODE
MODEL_FLAGS=()
EXPECTED_MODELS=1
case "$SMOKE_MODE" in
    qwen)
        MODEL_FLAGS=(--openrouter-models qwen/qwen3-235b-a22b-thinking-2507)
        ;;
    opus)
        MODEL_FLAGS=(
            --openrouter-models  # empty list (drop OpenRouter models)
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
    baselines)
        MODEL_FLAGS=(
            --openrouter-models qwen/qwen3-235b-a22b-thinking-2507
            --include-opus
            --include-baselines
            --anthropic-model claude-opus-4-7
        )
        EXPECTED_MODELS=4  # qwen3 + opus + deepseek + kimi
        ;;
    *)
        echo "ERROR: SMOKE_MODE must be one of: qwen | opus | both | baselines (got $SMOKE_MODE)" >&2
        exit 1
        ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EXPECTED_CONDITIONS=5
EXPECTED_ROLLOUTS=$((N_PROMPTS * EXPECTED_CONDITIONS * EXPECTED_MODELS))

echo "==> Apollo dataset: $APOLLO_DATA"
echo "==> Apollo sha256:  $(shasum -a 256 "$APOLLO_DATA" | awk '{print $1}')"
echo "==> Output dir:     $OUT_DIR"
echo "==> Smoke mode:     $SMOKE_MODE ($EXPECTED_MODELS model(s))"
echo "==> Plan:           $N_PROMPTS alignment-eval prompts × 5 conditions × $EXPECTED_MODELS models = $EXPECTED_ROLLOUTS rollouts"
echo "==> Estimated:      ~\$$EXPECTED_MODELS, ~5 min"
echo

# --- 1) rollouts ----------------------------------------------------------

echo "==> [1/3] Running rollouts..."
python scripts/run_v3_clause_ablation.py \
    --apollo-dataset-path "$APOLLO_DATA" \
    --n-prompts "$N_PROMPTS" \
    --prompt-mode full_transcript \
    "${MODEL_FLAGS[@]}" \
    --rollout-concurrency 8 \
    --out-dir "$OUT_DIR" \
    --seed 2

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
python scripts/analyze_v3_clause_ablation.py \
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
SUMMARY_PATH="$OUT_DIR/v3_clause_ablation_summary.smoke.json"
REPORT_PATH="$OUT_DIR/v3_clause_ablation_report.smoke.md"

OUT_DIR="$OUT_DIR" SUMMARY_PATH="$SUMMARY_PATH" \
    ROLLOUTS_PATH="$ROLLOUTS_PATH" CLS_PATH="$CLS_PATH" \
    EXPECTED_ROLLOUTS="$EXPECTED_ROLLOUTS" EXPECTED_MODELS="$EXPECTED_MODELS" \
    N_PROMPTS="$N_PROMPTS" \
    python - <<'PYEOF'
import json, os, sys

rollouts = [json.loads(l) for l in open(os.environ["ROLLOUTS_PATH"]) if l.strip()]
cls = [json.loads(l) for l in open(os.environ["CLS_PATH"]) if l.strip()]
summary = json.loads(open(os.environ["SUMMARY_PATH"]).read())
expected_rollouts = int(os.environ["EXPECTED_ROLLOUTS"])
expected_models = int(os.environ["EXPECTED_MODELS"])
n_prompts = int(os.environ["N_PROMPTS"])

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
expected_conditions = {"neutral", "eval_only", "reviewer_only", "comparison_only", "full_inject"}
if conditions != expected_conditions:
    errors.append(f"expected conditions {expected_conditions}, got {conditions}")

# V3 is alignment-eval ONLY, by construction
labels = {r["apollo_is_eval"] for r in rollouts}
if labels != {"true"}:
    errors.append(f"V3 should have only eval-labeled prompts; got apollo_is_eval={labels}")

eval_types = {r["apollo_eval_type"] for r in rollouts}
if eval_types != {"alignment"}:
    errors.append(
        f"V3 should have only alignment eval_type; got apollo_eval_type={eval_types}"
    )

if not summary.get("per_model") or len(summary["per_model"]) != expected_models:
    errors.append(
        f"summary has {len(summary.get('per_model', []))} per_model entries, expected {expected_models}"
    )

expected_cells = {"neutral", "eval_only", "reviewer_only", "comparison_only", "full_inject"}
for pm in summary.get("per_model", []):
    cells = pm.get("cells", {})
    if set(cells.keys()) != expected_cells:
        errors.append(f"[{pm['model_name']}] summary cells {set(cells.keys())} != {expected_cells}")
    for k, c in cells.items():
        if c.get("n", 0) != n_prompts:
            errors.append(f"[{pm['model_name']}] cell {k} has n={c.get('n')}, expected {n_prompts}")
        if "vea_mean" not in c:
            errors.append(f"[{pm['model_name']}] cell {k} missing vea_mean (VEA join failed?)")
    # Attribution should be reachable; with n_prompts=5 and min_cell_n=3 it
    # may legitimately land on any of the 6 categories. Just verify it's set
    # to a known value and not blank.
    attribution = pm.get("attribution", "")
    valid_attributions = {
        "insufficient_data",
        "null_at_full",
        "additive",
        "synergistic",
        "unclassified",
    }
    if not (attribution in valid_attributions or attribution.startswith("single_clause:")):
        errors.append(f"[{pm['model_name']}] unexpected attribution {attribution!r}")

# Per-singleton effect structure check
for pm in summary.get("per_model", []):
    for clause in ("eval_only", "reviewer_only", "comparison_only"):
        eff = pm.get("singleton_effects", {}).get(clause, {})
        if "point" not in eff or "ci" not in eff or "p_value" not in eff:
            errors.append(
                f"[{pm['model_name']}] singleton:{clause} missing point/ci/p_value"
            )
    full = pm.get("full_inject_effect", {})
    if "point" not in full or "ci" not in full:
        errors.append(f"[{pm['model_name']}] full_inject_effect missing point/ci")
    residual = pm.get("additivity_residual", {})
    if "point" not in residual or "ci" not in residual:
        errors.append(f"[{pm['model_name']}] additivity_residual missing point/ci")

# Holm should have (#models * 3 singletons) rows; full_inject excluded.
holm = summary.get("holm_bonferroni", [])
expected_holm_rows = expected_models * 3
if len(holm) != expected_holm_rows:
    errors.append(f"expected {expected_holm_rows} Holm rows, got {len(holm)}")
if any("full_inject" in r.get("effect", "") for r in holm):
    errors.append("Holm family should exclude full_inject (it's the V1-replication anchor)")

# Cross-model direction comparison row count = 3 singletons + 1 anchor
cross = summary.get("cross_model_direction_comparison", [])
if len(cross) != 4:
    errors.append(f"expected 4 cross-model rows (3 singletons + full_inject), got {len(cross)}")

# Per-model trace check — every model in the lineup should produce at least
# some non-empty reasoning_trace. Zero traces from a thinking model is a
# provider config bug (this caught V2's opus thinking-empty regression).
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
    attribution_summary = ", ".join(
        f"{pm['model_name'].split(':')[-1]}={pm['attribution']}"
        for pm in summary["per_model"]
    )
    print(
        f"PASS: {expected_rollouts} rollouts + classifications + 5 conditions × {expected_models} models. "
        f"traces: {trace_summary}. attributions: {attribution_summary}"
    )
PYEOF

echo
echo "==> V3 smoke test complete."
echo "    Report:  $REPORT_PATH"
echo "    Summary: $SUMMARY_PATH"
echo
echo "    Inspect the report:"
echo "        cat $REPORT_PATH"
echo
echo "    If everything looks right, draft the pre-registration"
echo "    (docs/pre_registration_v3_clause_ablation.md, PR 4) next."
