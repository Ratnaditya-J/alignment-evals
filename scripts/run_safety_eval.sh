#!/usr/bin/env bash
#
# One-shot wrapper for a publishable safety-only cross-protocol run.
#
# Runs the full pipeline:
#   1. Sanity-check the four env vars used by _discover_models.
#   2. run_cross_protocol_comparison.py with --allow-network --safety-only
#      so BeaverTails actually loads (NOT the 2-prompt synthetic fallback).
#      Saves rollouts.jsonl.
#   3. analyze_run.sh -> regex rescore + LLM-judge cross-check.
#   4. Final rescore with --primary-scorer llm so the headline tables in
#      cross_protocol_report.md are LLM-judge-driven (not regex), which
#      is the publishable signal given how miscalibrated the regex is on
#      modern third-person-refusal phrasing.
#
# Usage:
#   ./scripts/run_safety_eval.sh                                # defaults to runs/cross-protocol-v5
#   ./scripts/run_safety_eval.sh runs/cross-protocol-v6         # custom output dir
#   N_PROMPTS=200 ./scripts/run_safety_eval.sh runs/smoke       # smaller corpus
#
# Required env (set these before running):
#   OPENAI_API_KEY
#   ANTHROPIC_API_KEY        (also used as the LLM judge - cross-vendor)
#   OPENROUTER_API_KEY
#   CROSS_OPENAI_MODELS              e.g. "gpt-4o-mini,gpt-5.5"
#   CROSS_ANTHROPIC_MODELS           e.g. "claude-haiku-4-5-20251001,claude-opus-4-7"
#   CROSS_OPENROUTER_MODELS          e.g. "deepseek/deepseek-v3.2-exp,..."
#
# Optional env:
#   N_PROMPTS               default 500
#   N_ROLLOUTS              default 1
#   MAX_CONCURRENT          default 16
#   REASONING_EFFORT        default high
#   JUDGE_PROVIDER          default anthropic
#   JUDGE_MODEL             default claude-haiku-4-5-20251001
#
set -euo pipefail

OUT_DIR="${1:-runs/cross-protocol-v5}"
N_PROMPTS="${N_PROMPTS:-500}"
N_ROLLOUTS="${N_ROLLOUTS:-1}"
MAX_CONCURRENT="${MAX_CONCURRENT:-16}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-anthropic}"
JUDGE_MODEL="${JUDGE_MODEL:-claude-haiku-4-5-20251001}"
PY="${PYTHON:-python}"

# ---------------------------------------------------------------------------
# Pre-flight env check - fail fast if anything is missing so the user
# doesn't kick off a 30-minute run and discover later that ANTHROPIC_API_KEY
# wasn't exported.
# ---------------------------------------------------------------------------

echo "=== Safety eval pipeline -> $OUT_DIR ==="
echo

missing=0
for var in OPENAI_API_KEY ANTHROPIC_API_KEY OPENROUTER_API_KEY \
           CROSS_OPENAI_MODELS CROSS_ANTHROPIC_MODELS CROSS_OPENROUTER_MODELS; do
  if [ -z "${!var:-}" ]; then
    echo "MISSING: $var"
    missing=$((missing + 1))
  else
    val="${!var}"
    if [[ "$var" == *API_KEY* ]]; then
      echo "OK: $var (${#val} chars)"
    else
      echo "OK: $var=$val"
    fi
  fi
done

if [ "$missing" -gt 0 ]; then
  echo
  echo "ERROR: $missing required env var(s) missing. Set them and re-run." >&2
  exit 1
fi

echo
echo "Run config:"
echo "  output dir       = $OUT_DIR"
echo "  n_prompts        = $N_PROMPTS"
echo "  n_rollouts       = $N_ROLLOUTS"
echo "  max_concurrent   = $MAX_CONCURRENT"
echo "  reasoning_effort = $REASONING_EFFORT"
echo "  judge            = $JUDGE_PROVIDER / $JUDGE_MODEL"
echo

# ---------------------------------------------------------------------------
# Step 1 - the main eval (rollouts.jsonl)
# ---------------------------------------------------------------------------

echo "--- Step 1/3: cross-protocol run (~30-45 min, ~\$70 on 7 models x 500 prompts) ---"
echo

"$PY" scripts/run_cross_protocol_comparison.py \
    --output-dir "$OUT_DIR" \
    --allow-network \
    --safety-only \
    --n-prompts "$N_PROMPTS" \
    --n-rollouts "$N_ROLLOUTS" \
    --save-rollouts \
    --judge-provider "$JUDGE_PROVIDER" \
    --judge-model "$JUDGE_MODEL" \
    --reasoning-effort "$REASONING_EFFORT" \
    --max-concurrent "$MAX_CONCURRENT"

# ---------------------------------------------------------------------------
# Step 2 - analyze wrapper does regex rescore + LLM-judge cross-check
# ---------------------------------------------------------------------------

echo
echo "--- Step 2/3: analyze (regex rescore + LLM-judge cross-check) ---"
echo

./scripts/analyze_run.sh "$OUT_DIR" --no-open

# ---------------------------------------------------------------------------
# Step 3 - flip primary scorer to LLM. The judge scores from step 2 are
# already in rollouts_scored.jsonl, so this is a free re-aggregation pass.
# ---------------------------------------------------------------------------

echo
echo "--- Step 3/3: flip --primary-scorer to LLM (free, reuses prior judge scores) ---"
echo

"$PY" scripts/rescore_cross_protocol.py "$OUT_DIR" \
    --primary-scorer llm \
    --in-place

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

echo
echo "=== Done ==="
echo "Report:           $OUT_DIR/cross_protocol_report.md"
echo "Summary JSON:     $OUT_DIR/cross_protocol_summary.json"
echo "Scored rollouts:  $OUT_DIR/rollouts_scored.jsonl"
echo
echo "Headline tables in the report now use the LLM judge (with the tighter"
echo "request-vs-question rubric) as the primary scorer. Regex is shown as"
echo "the cross-check."

if command -v open >/dev/null 2>&1; then
  open "$OUT_DIR/cross_protocol_report.md" 2>/dev/null || true
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$OUT_DIR/cross_protocol_report.md" 2>/dev/null || true
fi
