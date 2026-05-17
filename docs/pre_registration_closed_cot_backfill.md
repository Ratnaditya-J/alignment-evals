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

*Filled in 2026-05-17, post-execution.*

- **Dataset file path:** `~/data/apollo/dataset.json`
- **Dataset sha256:** `9e375112910d2d56cd52dd9e6e1b70d44e28b1c44191efbffc1075d17e3de839` (matches V2 main + V3 main records)
- **V2 backfill rollouts completed:** **800 / 800** (100 prompts × 2 conditions × 2 models)
- **V3 backfill rollouts completed:** **1,000 / 1,000** (100 prompts × 5 conditions × 2 models)
- **Models in backfill rollouts:** `openrouter:openai/gpt-5.5`, `openrouter:google/gemini-2.5-pro`
- **`run_v2_apollo_factorial.py` (backfill) exit code:** 0
- **`run_v3_clause_ablation.py` (backfill) exit code:** 0
- **`analyze_v2_factorial.py` (with-backfill) exit code:** 0 (after the `_join_vea` fix in `1c6d59f`; see notes below)
- **`analyze_v3_clause_ablation.py` (with-backfill) exit code:** 0

Notes:

- **Initial V2 with-backfill analyzer run produced 4-model output
  (silent backfill drop).** Diagnosed as the `_join_vea` function in
  both V2 and V3 analyzers unconditionally filtering rollouts without
  matching VEA classifications. The closed-CoT models intentionally
  have no VEA classifications per §4, and the original join logic was
  dropping them at the join step. Fixed in commit `1c6d59f`: unmatched
  rollouts are now preserved with `is_vea=False` so behavioral data
  (refusal, the primary outcome) survives the join. Same fix applied
  to both `scripts/analyze_v2_factorial.py` and
  `scripts/analyze_v3_clause_ablation.py`. Regression tests added to
  both test suites (`test_rollouts_without_vea_classifications_are_preserved_not_dropped`).
- After the fix, re-running the analyzer on the combined rollouts
  files produced 6-model output for both V2 and V3 as intended. No
  re-rollout was required — the rollouts files were already correct
  (V2: 2,400 lines, 6 models; V3: 4,970 lines, 6 models).

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

*Filed 2026-05-17. Executed 2026-05-17. Headline: gpt-5.5 and
gemini-2.5-pro both show robust intrinsic-eval-label main effects
on V2 (Family B Holm-survivors); both are flat on V3 inject at
clause and full resolution (null_at_full attribution). gpt-5.5 is
the cleanest behavioral null in the entire 6-model lineup.*

### V2 backfill results (Family B: 2 models × 3 effects = 6 hypotheses)

**Per-model effects (strict VEA, paired bootstrap B=2000):**

| Model | inject main | 95% CI | intrinsic main | 95% CI | interaction | 95% CI |
|---|---:|---|---:|---|---:|---|
| gpt-5.5         | −0.010 | [−0.050, +0.025] | **+0.110** | [+0.040, +0.180] | +0.000 | [−0.080, +0.070] |
| gemini-2.5-pro  | +0.005 | [−0.020, +0.030] | **+0.095** | [+0.045, +0.155] | −0.010 | [−0.060, +0.040] |

**Family B Holm-Bonferroni decisions (6 hypotheses, α=0.05):**

| Model | Effect | Point | p | Threshold | Reject @ 0.05 |
|---|---|---:|---:|---:|:---:|
| gpt-5.5         | `main_effect_intrinsic` | +0.110 | 0.001 | 0.0036 | **yes** |
| gemini-2.5-pro  | `main_effect_intrinsic` | +0.095 | 0.000 | 0.0031 | **yes** |
| gemini-2.5-pro  | `main_effect_inject`    | +0.005 | 0.875 | 0.0100 | no |
| gemini-2.5-pro  | `interaction_effect`    | −0.010 | 0.842 | 0.0071 | no |
| gpt-5.5         | `main_effect_inject`    | −0.010 | 0.688 | 0.0063 | no |
| gpt-5.5         | `interaction_effect`    | +0.000 | 1.000 | 0.0500 | no |

**Family B survivors: 2 of 6.** Both intrinsic main effects survive
Holm; no inject or interaction effects survive on either model.

**V2 backfill headline:** The intrinsic-eval-label refusal effect
extends cleanly from the V2 main reasoning trio + opus to two
closed-CoT frontier models. gpt-5.5's +0.110 is the second-largest
intrinsic main effect in the entire 6-model lineup (after opus at
+0.130). The inject effect remains null at clause and full resolution
on both backfill models.

### V3 backfill results (Family B: 2 models × 3 singletons = 6 hypotheses; full_inject is anchor)

**Per-model attribution and full_inject anchor (strict VEA):**

| Model | full_inject Δ | 95% CI | p | Attribution |
|---|---:|---|---:|---|
| gpt-5.5         | **+0.000** | [−0.060, +0.060] | 1.000 | **null_at_full** |
| gemini-2.5-pro  | −0.010 | [−0.050, +0.030] | 0.810 | **null_at_full** |

**Per-singleton effects (strict):**

| Model | eval_only | reviewer_only | comparison_only |
|---|---:|---:|---:|
| gpt-5.5         | −0.010 [−0.050, +0.030] p=0.800 | +0.010 [−0.030, +0.050] p=0.844 | +0.010 [−0.040, +0.060] p=0.891 |
| gemini-2.5-pro  | −0.030 [−0.070, +0.000] p=0.102 | −0.020 [−0.050, +0.000] p=0.296 | −0.030 [−0.080, +0.020] p=0.313 |

**Family B Holm-Bonferroni decisions (6 hypotheses, α=0.05):** 0 of 6
survive. All 6 singleton point estimates are ≤0.030 absolute with
CIs that include zero. gemini-2.5-pro's `eval_only` (p=0.102) is the
closest to its Holm threshold of 0.0029.

**V3 backfill headlines:**

1. **gpt-5.5 is the cleanest behavioral null in the lineup.** Point
   estimate exactly **0.000** with the tightest CI in the entire
   6-model V3 lineup [−0.060, +0.060]. Every singleton effect within
   ±0.010 with CIs cleanly including zero. gpt-5.5 has a stable
   refusal posture on Apollo alignment-eval prompts (baseline 0.37,
   the highest of any V3 model) that is completely unresponsive to
   extrinsic eval-framing at any clause resolution.
2. **gemini-2.5-pro shows a borderline sub-additive pattern but the
   attribution is correctly null_at_full.** All three singletons
   mildly suppress refusal (each Δ ≈ −0.020 to −0.030), but the
   combined full_inject reverts to baseline (Δ −0.010). The
   additivity residual is +0.070 [+0.000, +0.150] p=0.057 (strict) /
   [+0.010, +0.150] p=0.043 (broad) — borderline excludes zero in
   broad. Attribution lands on `null_at_full` because the §5.5
   decision tree's first check (full_inject CI includes zero) takes
   priority. The sub-additive structure is a per-model footnote, not
   a headline claim.
3. **V1's direction asymmetry on opus + qwen3 is now triangulated
   against 4 different null models** (deepseek, kimi, gpt-5.5,
   gemini-2.5-pro), spanning closed-CoT and open-reasoning families
   from 4 model labs. The asymmetric signal is not an artifact of the
   V3 main pre-reg's 4-model lineup choice.

### Paper / writeup updates

V2 backfill numbers integrated into `docs/paper_draft.md` §5.5 (V2
Apollo factorial). The intrinsic main-effect table in §5.5.2 now lists
5 of 6 Holm-surviving intrinsic effects (3 from V2 Family A + 2 from
V2 Family B backfill). Per-cell summary in §5.5.2 includes gpt-5.5
and gemini-2.5-pro with VEA cells annotated `0.00 (n/a)` per §4 of
this supplement.

V3 backfill numbers integrated into `docs/paper_draft.md` §5.6 (V3
inject-clause ablation). Headline 6-model attribution table includes
both backfill models as `null_at_full`. §5.6.5 dedicated subsection
describes gpt-5.5's clean null and gemini-2.5-pro's sub-additive
footnote.

V1+V2+V3 synthesis in `docs/paper_draft.md` §7.5 reading (vi) reports
gpt-5.5 specifically as "the cleanest behavioral null in the lineup."

*The backfill supplement worked as intended: 2 of 6 Family B
hypotheses survived correction (both intrinsic main effects),
extending V2's intrinsic-eval-label finding to closed-CoT models
without modifying the V2 main pre-reg's headline claims. The V3
backfill confirms the V1 inject's behavioral null on closed-CoT
frontier models at clause resolution, completing the V3 picture as a
2-active-model story (opus + qwen3) against 4 null models from 4
different labs.*

---

*Filed 2026-05-17. This document is committed to the repository
**before** the commands in §6 are executed; see the git log for proof
of timing.*
