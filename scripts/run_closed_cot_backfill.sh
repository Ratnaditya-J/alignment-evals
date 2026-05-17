#!/usr/bin/env bash
# Closed-CoT backfill: run openai/gpt-5.5 + google/gemini-2.5-pro on V2 and V3
# WITHOUT re-running the existing 4-model lineup.
#
# Reads V2's existing rollouts.jsonl + V3's existing rollouts.jsonl (the main
# pre-reg runs), appends new closed-CoT rollouts, and re-runs the analyzers on
# the combined files. Holm-Bonferroni interpretation is two-family per the
# backfill pre-reg's §5: V2-main + V2-backfill corrected separately, V3-main +
# V3-backfill corrected separately.
#
# Pre-registered in docs/pre_registration_closed_cot_backfill.md §6.
#
# Cost: ~$50-60. Wall time: ~15-25 min.
#
# Usage:
#   APOLLO_DATA=~/data/apollo/dataset.json \
#   OPENROUTER_API_KEY=sk-or-... \
#   V2_RUN_DIR=runs/v2-apollo-factorial-prereg \
#   V3_RUN_DIR=runs/v3-clause-ablation-prereg \
#   bash scripts/run_closed_cot_backfill.sh
#
# Optional env:
#   SKIP_V2=1 to skip V2 backfill (run only V3)
#   SKIP_V3=1 to skip V3 backfill (run only V2)

set -euo pipefail

# --- prereqs ---------------------------------------------------------------

APOLLO_DATA="${APOLLO_DATA:-$HOME/data/apollo/dataset.json}"
V2_RUN_DIR="${V2_RUN_DIR:-runs/v2-apollo-factorial-prereg}"
V3_RUN_DIR="${V3_RUN_DIR:-runs/v3-clause-ablation-prereg}"
SKIP_V2="${SKIP_V2:-0}"
SKIP_V3="${SKIP_V3:-0}"

BACKFILL_MODELS=(openai/gpt-5.5 google/gemini-2.5-pro)

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "ERROR: OPENROUTER_API_KEY must be set" >&2
    exit 1
fi

if [ ! -f "$APOLLO_DATA" ]; then
    echo "ERROR: Apollo dataset not found at $APOLLO_DATA" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Match V2 / V3 pre-reg §6 discipline: ensure we're on latest main before
# any rollouts so the recorded commit-hash and the executed code agree.
echo "==> git: ensuring latest main"
git checkout main
git pull origin main
echo

# Validate main run dirs exist before doing anything destructive
if [ "$SKIP_V2" != "1" ]; then
    if [ ! -f "$V2_RUN_DIR/rollouts.jsonl" ]; then
        echo "ERROR: V2 main rollouts not found at $V2_RUN_DIR/rollouts.jsonl" >&2
        echo "       The V2 main pre-reg run must complete before backfill." >&2
        echo "       Set SKIP_V2=1 to skip V2 backfill." >&2
        exit 1
    fi
fi

if [ "$SKIP_V3" != "1" ]; then
    if [ ! -f "$V3_RUN_DIR/rollouts.jsonl" ]; then
        echo "ERROR: V3 main rollouts not found at $V3_RUN_DIR/rollouts.jsonl" >&2
        echo "       The V3 main pre-reg run must complete before backfill." >&2
        echo "       Set SKIP_V3=1 to skip V3 backfill." >&2
        exit 1
    fi
fi

echo "==> Apollo dataset: $APOLLO_DATA"
echo "==> Apollo sha256:  $(shasum -a 256 "$APOLLO_DATA" | awk '{print $1}')"
echo "==> Backfill models: ${BACKFILL_MODELS[*]}"
[ "$SKIP_V2" != "1" ] && echo "==> V2 main run dir: $V2_RUN_DIR"
[ "$SKIP_V3" != "1" ] && echo "==> V3 main run dir: $V3_RUN_DIR"
echo

# --- V2 backfill -----------------------------------------------------------

if [ "$SKIP_V2" != "1" ]; then
    V2_BACKFILL_DIR="$V2_RUN_DIR.backfill-closed-cot"
    echo "==> [V2 1/3] Running V2 backfill rollouts (gpt-5.5 + gemini-2.5-pro on V2's 200 prompts)..."
    python scripts/run_v2_apollo_factorial.py \
        --apollo-dataset-path "$APOLLO_DATA" \
        --n-eval 100 --n-deployment 100 \
        --prompt-mode full_transcript \
        --openrouter-models "${BACKFILL_MODELS[@]}" \
        --rollout-concurrency 16 \
        --out-dir "$V2_BACKFILL_DIR" \
        --seed 0

    echo
    echo "==> [V2 2/3] Appending backfill rollouts to V2 main rollouts file..."
    BACKFILL_LINES=$(wc -l < "$V2_BACKFILL_DIR/rollouts.jsonl")
    cat "$V2_BACKFILL_DIR/rollouts.jsonl" >> "$V2_RUN_DIR/rollouts.jsonl"
    echo "    Appended $BACKFILL_LINES lines to $V2_RUN_DIR/rollouts.jsonl"

    echo
    echo "==> [V2 3/3] Re-running V2 analyzer on combined rollouts (strict VEA, .with-backfill suffix)..."
    python scripts/analyze_v2_factorial.py \
        --run-dir "$V2_RUN_DIR" \
        --classifications-path "$V2_RUN_DIR/goodfire_vea_classifications.jsonl" \
        --strict-vea \
        --bootstrap-samples 2000 \
        --seed 0 \
        --out-suffix .strict.with-backfill
    echo
fi

# --- V3 backfill -----------------------------------------------------------

if [ "$SKIP_V3" != "1" ]; then
    V3_BACKFILL_DIR="$V3_RUN_DIR.backfill-closed-cot"
    echo "==> [V3 1/3] Running V3 backfill rollouts (gpt-5.5 + gemini-2.5-pro on V3 baselines tier, N=100)..."
    python scripts/run_v3_clause_ablation.py \
        --apollo-dataset-path "$APOLLO_DATA" \
        --n-prompts 100 \
        --prompt-mode full_transcript \
        --openrouter-models "${BACKFILL_MODELS[@]}" \
        --rollout-concurrency 16 \
        --out-dir "$V3_BACKFILL_DIR" \
        --seed 2

    echo
    echo "==> [V3 2/3] Appending backfill rollouts to V3 main rollouts file..."
    BACKFILL_LINES=$(wc -l < "$V3_BACKFILL_DIR/rollouts.jsonl")
    cat "$V3_BACKFILL_DIR/rollouts.jsonl" >> "$V3_RUN_DIR/rollouts.jsonl"
    echo "    Appended $BACKFILL_LINES lines to $V3_RUN_DIR/rollouts.jsonl"

    echo
    echo "==> [V3 3/3] Re-running V3 analyzer on combined rollouts (strict VEA, .with-backfill suffix)..."
    python scripts/analyze_v3_clause_ablation.py \
        --run-dir "$V3_RUN_DIR" \
        --classifications-path "$V3_RUN_DIR/goodfire_vea_classifications.jsonl" \
        --strict-vea \
        --bootstrap-samples 2000 \
        --seed 0 \
        --out-suffix .strict.with-backfill
    echo
fi

# --- summary ---------------------------------------------------------------

echo "==> Closed-CoT backfill complete."
[ "$SKIP_V2" != "1" ] && echo "    V2 with-backfill summary: $V2_RUN_DIR/v2_apollo_factorial_summary.strict.with-backfill.json"
[ "$SKIP_V2" != "1" ] && echo "    V2 with-backfill report:  $V2_RUN_DIR/v2_apollo_factorial_report.strict.with-backfill.md"
[ "$SKIP_V3" != "1" ] && echo "    V3 with-backfill summary: $V3_RUN_DIR/v3_clause_ablation_summary.strict.with-backfill.json"
[ "$SKIP_V3" != "1" ] && echo "    V3 with-backfill report:  $V3_RUN_DIR/v3_clause_ablation_report.strict.with-backfill.md"
echo
echo "    Reminder: Holm-Bonferroni is two-family per the backfill pre-reg §5."
echo "    The 4-original-model effects keep their existing Holm decisions."
echo "    The 2-backfill-model effects are corrected independently as a"
echo "    separate 2 × N-effects family."
