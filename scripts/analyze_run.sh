#!/usr/bin/env bash
#
# Post-run analysis wrapper for a cross-protocol benchmark run.
#
# Runs all four post-eval steps in order:
#   1. Verify artefacts exist and inspect empty-response rates per model.
#   2. Rescore with the regex RefusalScorer (no API spend, full methodology
#      report: tier cuts, task-family breakdown, model identity, etc.).
#   3. Cross-check the regex scorer against an LLM judge (anthropic by
#      default, openai as fallback) when an API key is available.
#   4. Print the path to the rendered report and try to open it.
#
# Usage:
#   ./scripts/analyze_run.sh                       # defaults to runs/cross-protocol-v4
#   ./scripts/analyze_run.sh runs/cross-protocol-vN
#
# Environment:
#   ANTHROPIC_API_KEY  use anthropic claude-haiku-4-5 as LLM judge (preferred)
#   OPENAI_API_KEY     fallback: openai gpt-4o-mini judge
#   if neither is set, skips the LLM cross-check step
#
# Flags (positional after the run dir):
#   --no-llm    skip the LLM-judge cross-check even if a key is set
#   --no-open   don't try to open the report at the end
#
set -euo pipefail

RUN_DIR="${1:-runs/cross-protocol-v4}"
shift || true
WANT_LLM=1
WANT_OPEN=1
for arg in "$@"; do
  case "$arg" in
    --no-llm)  WANT_LLM=0 ;;
    --no-open) WANT_OPEN=0 ;;
    *) echo "unknown flag: $arg"; exit 2 ;;
  esac
done

PY="${PYTHON:-python}"

# ---------------------------------------------------------------------------
# 1. Verify the run completed cleanly
# ---------------------------------------------------------------------------

echo "=== Analyzing $RUN_DIR ==="

if [ ! -d "$RUN_DIR" ]; then
  echo "ERROR: $RUN_DIR not found" >&2
  exit 1
fi
if [ ! -f "$RUN_DIR/rollouts.jsonl" ]; then
  echo "ERROR: $RUN_DIR/rollouts.jsonl missing." >&2
  echo "       Re-run run_cross_protocol_comparison.py with --save-rollouts." >&2
  exit 1
fi

echo
echo "--- Run artefacts ---"
ls -la "$RUN_DIR"

ROLLOUT_COUNT=$(wc -l < "$RUN_DIR/rollouts.jsonl")
echo
echo "Rollouts: $ROLLOUT_COUNT"

# ---------------------------------------------------------------------------
# 2. Empty-response check — catches misconfigured reasoning-token budgets
# ---------------------------------------------------------------------------

echo
echo "--- Empty-response check (>5% is a red flag) ---"
"$PY" - "$RUN_DIR/rollouts.jsonl" <<'PY'
import json
import sys
from collections import Counter

path = sys.argv[1]
empty: Counter = Counter()
total: Counter = Counter()
for line in open(path):
    record = json.loads(line)
    model = record["model_name"]
    total[model] += 1
    if not str(record.get("response", "") or "").strip():
        empty[model] += 1

any_bad = False
for model in sorted(total):
    pct = 100 * empty[model] / total[model]
    flag = "  <-- HIGH" if pct >= 5 else ""
    if pct >= 5:
        any_bad = True
    print(f"  {model}: {empty[model]}/{total[model]} empty ({pct:.1f}%){flag}")

if any_bad:
    print()
    print(
        "WARNING: one or more models had >=5% empty responses; investigate "
        "before publishing. Common causes: max_tokens too low for a reasoning "
        "model, or the model returned a refusal that landed in the wrong field."
    )
PY

# ---------------------------------------------------------------------------
# 3. Regex rescore (always - no API spend)
# ---------------------------------------------------------------------------

echo
echo "--- Rescoring with regex RefusalScorer ---"
"$PY" scripts/rescore_cross_protocol.py "$RUN_DIR" --in-place

# ---------------------------------------------------------------------------
# 4. LLM-judge cross-check (optional)
# ---------------------------------------------------------------------------

JUDGE_PROVIDER=""
JUDGE_MODEL=""
if [ "$WANT_LLM" = "1" ]; then
  if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    JUDGE_PROVIDER="anthropic"
    JUDGE_MODEL="claude-haiku-4-5-20251001"
  elif [ -n "${OPENAI_API_KEY:-}" ]; then
    JUDGE_PROVIDER="openai"
    JUDGE_MODEL="gpt-4o-mini"
  fi
fi

if [ -n "$JUDGE_PROVIDER" ]; then
  echo
  echo "--- LLM-judge cross-check ($JUDGE_PROVIDER / $JUDGE_MODEL) ---"
  "$PY" scripts/rescore_cross_protocol.py "$RUN_DIR" \
      --llm-judge-provider "$JUDGE_PROVIDER" \
      --llm-judge-model "$JUDGE_MODEL" \
      --llm-judge-max-concurrent 8 \
      --in-place
else
  echo
  echo "--- Skipping LLM-judge cross-check ---"
  if [ "$WANT_LLM" = "0" ]; then
    echo "    (--no-llm passed)"
  else
    echo "    (neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set)"
  fi
fi

# ---------------------------------------------------------------------------
# 5. Done - point at the artefacts and try to open the report
# ---------------------------------------------------------------------------

REPORT="$RUN_DIR/cross_protocol_report.md"
SUMMARY="$RUN_DIR/cross_protocol_summary.json"
SCORED="$RUN_DIR/rollouts_scored.jsonl"

echo
echo "=== Done ==="
echo "Report:           $REPORT"
echo "Summary JSON:     $SUMMARY"
echo "Scored rollouts:  $SCORED"
echo
echo "Next steps:"
echo "  - read the 'Reasoning-tier cuts' table for your headline finding"
echo "  - for human-labelled scorer validation, run:"
echo "      ${PY} scripts/validate_refusal_scorer.py $RUN_DIR --n-per-bucket 5"

if [ "$WANT_OPEN" = "1" ]; then
  if command -v open >/dev/null 2>&1; then
    open "$REPORT" 2>/dev/null || true
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$REPORT" 2>/dev/null || true
  fi
fi
