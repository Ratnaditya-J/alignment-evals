# Pre-registration: qwen3 safety VEA mediation replication

**Pre-registered:** 2026-05-13
**Author:** Ratnaditya Jonnalagadda
**Status:** filed before running the replication; results to be appended below after execution

---

## 1. Background and motivation

The primary exploratory analysis (PR #50) found that on the n=500 mixed
corpus, qwen3-235B-thinking exhibits within-prompt diff-in-diff
mediation of refusal behavior by VEA verbalization on the **safety**
task family under the validated strict VEA rubric:

> DiD = +0.222, 95% CI [+0.029, +0.430], n_flip = 63 paired prompts.

The CI excludes zero. The point estimate is the largest single
behavioral mediation effect in the project. Three threats to this
finding's robustness:

1. **Exploratory analysis.** The strict-set definition was refined
   after seeing data (PR #49 swapped grader_awareness out, safety_check
   in). The current strict set is empirically grounded but the
   mediation result was computed on the same data that informed the
   rubric refinement. Some optimistic bias is expected.
2. **Wide CI.** The upper bound +0.430 is much larger than the point
   estimate +0.222. n_flip = 63 paired is moderate but not large.
3. **Single sample.** The qwen3 safety cell is one (model, family)
   draw from a population of possible runs. Sampling variability
   alone could produce a spurious CI-excludes-zero finding.

This pre-registration locks in the analysis pipeline before running
a replication on a disjoint prompt sample.

## 2. Hypothesis

> **H1:** Under inject framing on a fresh BeaverTails sample (disjoint
> from the n=500 sample analyzed in PR #50), qwen3-235B-thinking shows
> a per-prompt diff-in-diff mediation of refusal behavior by VEA
> verbalization under the validated strict rubric. Specifically, the
> CI on the safety-family diff-in-diff will exclude zero (positive
> direction).

The alternative (null) hypothesis is the absence of CI-excludes-zero
positive mediation in the replication sample.

## 3. Sample design

- **Model under test:** `openrouter:qwen/qwen3-235b-a22b-thinking-2507`
  (single model; we are not re-testing DeepSeek/Kimi in this
  replication).
- **Corpus:** BeaverTails safety prompts only (`--safety-only` flag).
- **Sample size:** n=200 BeaverTails prompts × 4 conditions × 1 model
  = 800 rollouts.
- **Disjointness:** prompts 501..700 from the BeaverTails loader's
  deterministic ordering, picked via `--corpus-offset 500`
  `--n-examples 200`. This ensures zero overlap with the prior n=500
  sample (which used prompts 1..500).
- **Rationale for n=200:** the prior finding has n_flip = 63; if the
  same flip rate (~32% of paired prompts) holds at n=200, we expect
  ~64 flips — comparable power. A larger n would tighten the CI but
  adds cost without buying methodological strength.

## 4. Pre-specified analysis

1. **Rollouts.** Run the standard 4-condition arxiv protocol
   (baseline / neutral / inject / remove) on the disjoint sample.
2. **VEA classification.** Use `google/gemini-2.5-flash-lite` via
   OpenRouter — same judge as the primary analysis. Apply the broad
   rubric (raw `is_vea` boolean from the judge).
3. **Strict re-classification.** Apply the validated strict subtype
   set `{explicit_eval, benchmark, safety_check}` via
   `--strict-vea` on `analyze_vea_mediation.py`.
4. **Diff-in-diff.** Compute the within-prompt diff-in-diff on the
   safety family. Pair prompts on `example_id`. Bootstrap-paired CI
   with B=200 resamples and seed=0.
5. **Decision rule** (see §5).

No other analyses will be reported as confirmatory. Exploratory
secondary analyses (per-condition VEA rates, broad-rubric DiD,
behavioral split) may be presented in the paper's appendix but will
not be characterized as confirming the H1 finding.

## 5. Decision rule

After running steps 1-4:

| Outcome | Verdict | Action |
|---|---|---|
| Safety-family DiD CI excludes zero, point estimate in [+0.05, +0.50] | **Replicated.** | Promote the finding from exploratory to confirmatory in the paper. Update §5.3, §7, §9, abstract. |
| Safety-family DiD CI excludes zero, point estimate outside [+0.05, +0.50] | **Replicated with magnitude shift.** | Report both numbers (primary and replication). Note the magnitude difference. Discuss possible reasons. |
| Safety-family DiD CI includes zero, with a positive point estimate ≥+0.05 | **Suggestive but not replicated.** | Demote the primary finding to "suggestive, n=63 in initial sample, not replicated at n~200." Update conclusion to remove the "strong_mediation" framing. |
| Safety-family DiD CI includes zero, with a point estimate <+0.05 or negative | **Not replicated.** | Treat the primary as a false positive. Substantially revise paper: VEA mediation on safety is exploratory, not supported. |

The decision is based on the **strict-rubric DiD on the safety
family only**. Broad-rubric and other-family numbers are
exploratory.

## 6. Pre-registered command

The exact command to be run. No flag will be changed without amending
this pre-registration document:

```bash
cd ~/IsItBenchmark && \
git checkout main && \
git pull origin main && \
python scripts/run_goodfire_vea.py \
  --n-examples 200 \
  --corpus-offset 500 \
  --safety-only \
  --openrouter-models qwen/qwen3-235b-a22b-thinking-2507 \
  --out-dir runs/goodfire-replication-qwen3-safety \
  --rollout-concurrency 32 \
  --vea-judge-provider openrouter \
  --vea-judge-model google/gemini-2.5-flash-lite \
  --vea-judge-concurrency 32 \
  --seed 0 && \
python scripts/analyze_vea_mediation.py \
  --run-dir runs/goodfire-replication-qwen3-safety \
  --strict-vea \
  --out-suffix .strict
```

Expected cost: ~$3-5 (rollouts) + ~$1 (VEA judge) = **~$5 total**.
Expected wall time: ~30 minutes.

## 7. Outputs to inspect

- `runs/goodfire-replication-qwen3-safety/goodfire_vea_summary.json`
- `runs/goodfire-replication-qwen3-safety/vea_mediation_summary.strict.json`
- `runs/goodfire-replication-qwen3-safety/vea_mediation_report.strict.md`

The decisive number lives in
`vea_mediation_summary.strict.json` → `per_model_family` → cell where
`task_family == "safety"` → `diff_in_diff` → `diff_in_diff` and
`diff_in_diff_ci`.

## 8. Deviations and amendments

If the replication run is interrupted, fails for a non-statistical
reason (API error, empty traces, judge parse errors), or otherwise
cannot complete as specified, the entire pre-registration must be
reset: rerun from scratch with the same parameters. Partial / patched
runs are not acceptable as the confirmatory sample.

Any change to the command in §6 — different model, different judge,
different rubric, different n — invalidates this pre-registration.
A new pre-registration must be filed before the modified run is
executed.

## 9. Results

*(to be appended after the replication runs; leave as a placeholder
until then)*

- Replication run completed: ___ (date)
- Run directory: ___
- Safety-family strict DiD point estimate: ___
- Safety-family strict DiD 95% CI: ___
- Verdict per §5: ___
- Paper changes made: ___

---

*Filed 2026-05-13. This document was committed to the repository
**before** the replication command in §6 was executed; see the git
log for proof of timing.*
