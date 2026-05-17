# Pre-registration: closed-CoT backfill for V2 and V3

**Pre-registered:** 2026-05-17
**Author:** Ratnaditya Jonnalagadda
**Supplements:**
  - `docs/pre_registration_v2_apollo_2x2.md` (V2 main pre-reg)
  - `docs/pre_registration_v3_clause_ablation.md` (V3 main pre-reg)
**Status:** filed before running. Results section below is empty until execution.

---

## 1. Motivation

V1's behavioral cross-protocol experiment included two closed-CoT
models — `openai/gpt-5.5` and `google/gemini-2.5-pro` family
representatives — alongside the open-reasoning trio and Anthropic
models. V1's Goodfire/mediation analysis dropped them because no
visible CoT means no VEA judgement, which is required for the
mediation question.

V2 (Apollo factorial) and V3 (inject-clause ablation) inherited the
V1-Goodfire 4-model lineup without methodological justification —
the **primary** outcomes of V2 (refusal × intrinsic eval label) and
V3 (refusal × clause attribution) are behavioral, not mediation-based.
Closed-CoT models can contribute to both primary outcomes; only the
secondary VEA cells would be empty.

This supplement adds `openai/gpt-5.5` and `google/gemini-2.5-pro` as
**behavioral-only** participants in V2 and V3, run as a separate
backfill family with its own Holm correction. The original V2 and V3
pre-registrations stay intact (their model lineups, samples, Holm
families, and headline claims are unaffected by this backfill).

## 2. What this supplement does and does not do

**This supplement DOES:**

- Add 2 closed-CoT models (gpt-5.5, gemini-2.5-pro) to V2 and V3
  result tables as a separate "closed-CoT" subset.
- Pre-register that the backfill rollouts are appended to the main
  rollouts.jsonl files of V2 and V3 (after the main runs complete).
- Pre-register that Holm-Bonferroni for the backfill subset is a
  SEPARATE family of `2 models × N effects`, not merged with the
  original V2 / V3 families.
- Re-use V2 and V3's exact prompt sets (same seeds, same selection
  rules — see §3 below).

**This supplement DOES NOT:**

- Modify the V2 main pre-reg's 4-model lineup or its 12-hypothesis
  Holm family.
- Modify the V3 main pre-reg's 4-model lineup or its 12-hypothesis
  Holm family.
- Re-run the original V2 or V3 models. The backfill writes new rollouts
  for the 2 new models only; existing rollouts on opus / qwen3 /
  deepseek / kimi are not touched.
- Attempt VEA judgement on closed-CoT model outputs. The VEA secondary
  outcome is reported as `n/a` for these 2 models. The analyzers
  already handle missing-VEA per the `has_vea` gate.

## 3. Sample design

### 3.1 V2 backfill subsample

Identical to V2 main pre-reg §3.2:
- n_eval = 100 alignment + capability mix, n_deployment = 100
- Selection: `ApolloEvalAwarenessLoader(seed=0, max_examples=100)`
  invoked twice with `label_filter="eval"` and
  `label_filter="deployment"`
- **Same prompts** as V2 main run, so V2's appended results are
  directly comparable cell-by-cell.

Rollouts: 200 prompts × 2 conditions × 2 models = **800 rollouts**.

### 3.2 V3 backfill subsample

Identical to V3 main pre-reg §3.2 baselines tier:
- N = 100 alignment-eval prompts (the baselines tier sample size)
- Selection: top 100 by `sha256(f"v3:{seed}:{example_id}")` with
  seed = 2 (same hash + same seed as V3 main baselines = strict subset
  of V3 active prompts; baselines, deepseek, kimi, and these 2
  closed-CoT models all share the same 100 prompts)

Rollouts: 100 prompts × 5 conditions × 2 models = **1,000 rollouts**.

### 3.3 Why N=100 (baselines tier) for V3 closed-CoT

The V3 main pre-reg's "active tier" (use-all) was justified by V1+V2's
small alignment-stratum effects on opus and qwen3 specifically. V1 and
V2 don't give us priors expecting larger effects from gpt-5.5 or
gemini-2.5-pro — V1's behavioral result on gpt-5.5 was null. N=100
matches the deepseek + kimi tier and saves ~$60. If the backfill
surfaces a non-null result at N=100, that itself is worth pre-registering
a follow-up at higher N.

### 3.4 Models

| Identifier (CLI form) | OpenRouter model ID | Notes |
|---|---|---|
| `openrouter:openai/gpt-5.5` | `openai/gpt-5.5` | closed-CoT (reasoning not exposed) |
| `openrouter:google/gemini-2.5-pro` | `google/gemini-2.5-pro` | closed-CoT (reasoning not exposed) |

Both reached via OpenRouter as the meta-provider; no OpenAI / Google
SDK changes required. The V2 and V3 runners already support arbitrary
OpenRouter model IDs via `--openrouter-models`.

### 3.5 Conditions

Identical to the respective main pre-regs:
- V2 backfill: `neutral` + `inject` (V2 §3.4)
- V3 backfill: all 5 V3 conditions (V3 §3.4)

System-prompt overlay rules unchanged.

## 4. VEA classification

**Skipped** for both backfills. Closed-CoT models do not expose CoT
in a form compatible with V1+V2's VEA judging (no verbatim reasoning
trace). The analyzers will read empty `reasoning_trace` fields and
emit VEA cells with `n=0` for these models. This is the expected
behavior and not an error.

## 5. Pre-specified analysis

### 5.1 V2 backfill analysis

Append V2 backfill rollouts.jsonl to V2 main rollouts.jsonl. Re-run
`scripts/analyze_v2_factorial.py` on the combined file. The analyzer
will produce per-model cells, main effects, and verdicts for all 6
models (4 original + 2 backfill).

**Holm-Bonferroni is computed in two separate families:**
- Family A (V2 main, unchanged): 4 models × 3 effects = 12 hypotheses
  — exactly as the V2 main pre-reg §5.6 specified. Decisions on these
  12 are NOT modified by the backfill.
- Family B (V2 backfill, new): 2 models × 3 effects = 6 hypotheses,
  corrected independently at family-wise α = 0.05.

The combined summary JSON will report both families. The original V2
paper §5.5 numbers are unchanged.

### 5.2 V3 backfill analysis

Append V3 backfill rollouts.jsonl to V3 main rollouts.jsonl. Re-run
`scripts/analyze_v3_clause_ablation.py` on the combined file. The
analyzer produces per-model cells, per-singleton effects, and
attribution categories for all 6 models.

**Holm-Bonferroni is computed in two separate families:**
- Family A (V3 main, unchanged): 4 models × 3 singletons = 12
  hypotheses.
- Family B (V3 backfill, new): 2 models × 3 singletons = 6 hypotheses,
  corrected independently.

The analyzer's existing `_apply_holm_bonferroni` function corrects
across all rows. To enforce the two-family split, the post-analysis
report manually partitions the holm_bonferroni list by model name
(see §6 commands).

### 5.3 Cross-model direction comparison

V3's `cross_model_direction_comparison` analysis (V3 §5.7) is
descriptive (not a hypothesis test). The 2 new models are added to
the per-clause sign table; no correction needed.

## 6. Pre-registered commands

Run AFTER the V2 main run is complete (already complete as of
2026-05-17) and AFTER the V3 main run is complete (scheduled per the
V3 main pre-reg §6).

```bash
cd ~/alignment-evals && \
git checkout main && \
git pull origin main && \
APOLLO_DATA=~/data/apollo/dataset.json && \
V2_RUN_DIR=runs/v2-apollo-factorial-prereg && \
V3_RUN_DIR=runs/v3-clause-ablation-prereg && \

# ============================================================================
# V2 BACKFILL: gpt-5.5 + gemini-2.5-pro on V2's same 200 prompts (seed=0)
# ============================================================================
python scripts/run_v2_apollo_factorial.py \
  --apollo-dataset-path "$APOLLO_DATA" \
  --n-eval 100 --n-deployment 100 \
  --prompt-mode full_transcript \
  --openrouter-models openai/gpt-5.5 google/gemini-2.5-pro \
  --rollout-concurrency 16 \
  --out-dir "$V2_RUN_DIR.backfill-closed-cot" \
  --seed 0 && \

# Append backfill rollouts to V2's main rollouts file
cat "$V2_RUN_DIR.backfill-closed-cot/rollouts.jsonl" >> "$V2_RUN_DIR/rollouts.jsonl" && \

# Re-run V2 analyzer on the combined file (existing VEA classifications
# still cover the 4 main models; the 2 backfill models will show empty
# VEA cells, which is the pre-registered expected behavior)
python scripts/analyze_v2_factorial.py \
  --run-dir "$V2_RUN_DIR" \
  --classifications-path "$V2_RUN_DIR/goodfire_vea_classifications.jsonl" \
  --strict-vea \
  --bootstrap-samples 2000 \
  --seed 0 \
  --out-suffix .strict.with-backfill && \

# ============================================================================
# V3 BACKFILL: gpt-5.5 + gemini-2.5-pro on V3 baselines-tier prompts (N=100)
# ============================================================================
python scripts/run_v3_clause_ablation.py \
  --apollo-dataset-path "$APOLLO_DATA" \
  --n-prompts 100 \
  --prompt-mode full_transcript \
  --openrouter-models openai/gpt-5.5 google/gemini-2.5-pro \
  --rollout-concurrency 16 \
  --out-dir "$V3_RUN_DIR.backfill-closed-cot" \
  --seed 2 && \

# Append backfill rollouts to V3's main rollouts file
cat "$V3_RUN_DIR.backfill-closed-cot/rollouts.jsonl" >> "$V3_RUN_DIR/rollouts.jsonl" && \

# Re-run V3 analyzer on the combined file
python scripts/analyze_v3_clause_ablation.py \
  --run-dir "$V3_RUN_DIR" \
  --classifications-path "$V3_RUN_DIR/goodfire_vea_classifications.jsonl" \
  --strict-vea \
  --bootstrap-samples 2000 \
  --seed 0 \
  --out-suffix .strict.with-backfill
```

For convenience, this command sequence is also wrapped in
`scripts/run_closed_cot_backfill.sh` (see §7).

Expected cost: ~$24 (V2 backfill) + ~$30 (V3 backfill) = **~$50-60
total**. Expected wall time: ~15-25 minutes.

## 7. Convenience wrapper script

`scripts/run_closed_cot_backfill.sh` orchestrates the §6 commands
with the same env-var requirements as the V3 smoke script. Usage:

```bash
APOLLO_DATA=~/data/apollo/dataset.json \
OPENROUTER_API_KEY=sk-or-... \
V2_RUN_DIR=runs/v2-apollo-factorial-prereg \
V3_RUN_DIR=runs/v3-clause-ablation-prereg \
bash scripts/run_closed_cot_backfill.sh
```

The script verifies the main run directories exist before starting
(refuses to run on a missing V2 or V3 main run).

## 8. Recorded at execution time

*To be filled in at execution time.*

- **Dataset file path:** _TBD_
- **Dataset sha256:** _TBD_ (should match V2 main + V3 main records)
- **V2 backfill rollouts completed:** _TBD_ / 800
- **V3 backfill rollouts completed:** _TBD_ / 1000
- **Models in backfill rollouts:** `openrouter:openai/gpt-5.5`, `openrouter:google/gemini-2.5-pro`
- **`run_v2_apollo_factorial.py` (backfill) exit code:** _TBD_
- **`run_v3_clause_ablation.py` (backfill) exit code:** _TBD_
- **`analyze_v2_factorial.py` (with-backfill) exit code:** _TBD_
- **`analyze_v3_clause_ablation.py` (with-backfill) exit code:** _TBD_

## 9. Deviations and amendments

This supplement is a behavioral-only extension of V2 and V3. The
deviations policy is:

- If the backfill fails partially, restart the affected lineup from
  scratch. No partial backfill claims allowed.
- The V2 and V3 main pre-regs' deviations rules are NOT triggered by
  this supplement: it adds new rollouts on new models without
  modifying existing main-pre-reg parameters or rollouts.
- Any change to §3 (subsample, models, conditions) or §5 (analysis
  pipeline) of this supplement requires a new supplement filing
  before execution.

## 10. Results

*To be filled in after execution.*

---

*Filed 2026-05-17. This document is committed to the repository
**before** the commands in §6 are executed; see the git log for proof
of timing.*
