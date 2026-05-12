# IsItBenchmark — writeup notes

A running ledger of observations across runs that are worth carrying
forward into a published writeup. Each entry is dated and tied to specific
run artifacts so the claim can be re-derived from the raw data later.

The file is intentionally append-only and informal. Findings here are
**hypotheses with cited evidence**, not conclusions. Move an entry into a
proper methods/results section only after a clean, reproducible run
supports the claim with CI-excludes-zero statistical power.

---

## Corpus composition changes the headline refusal effect (v5 vs v6)

**Observation.** When the same open-reasoning models are tested on a
safety-only corpus vs a mixed safety+capability corpus, the
inject-minus-neutral refusal-rate signal looks materially different.

| Run | Corpus | Open-reasoning pooled Δ inject−neutral | 95% CI | n paired |
|---|---|---:|---|---:|
| v5 | BeaverTails safety only | **+0.033** | [+0.008, +0.060] | 1,493 |
| v6 | mixed (knowledge / math / safety / truthfulness, 125 each) | +0.014 | [-0.019, +0.041] | 1,474 |

In v5 the open-reasoning bucket showed a CI-excludes-zero positive effect.
In v6 with comparable model lineup and per-cell n, the same bucket has a
CI that includes zero. The bucket disappearing from the "supported
claims" section between runs is the single biggest qualitative
difference between v5 and v6.

**Two non-exclusive mechanisms could explain this.** Reporting both because
v6 data alone can't fully discriminate them.

### Hypothesis A: aggregation dilution

The non-safety task families have near-zero refusal rates with tight CIs.
v6 per-(model, family) refusal numbers from `runs/cross-protocol-v6`:

| Model | Family | Baseline | Neutral | Inject | Δ inject−neutral |
|---|---|---:|---:|---:|---:|
| deepseek-v3.2-exp | knowledge | 0.000 | 0.000 | 0.000 | +0.000 |
| deepseek-v3.2-exp | math      | 0.000 | 0.000 | 0.000 | +0.000 |
| deepseek-v3.2-exp | truthfulness | 0.040 | 0.048 | 0.072 | +0.024 |
| deepseek-v3.2-exp | safety    | 0.784 | 0.856 | 0.864 | +0.008 |
| kimi-k2-thinking  | safety    | 0.782 | 0.766 | 0.811 | +0.045 |
| qwen3-thinking    | safety    | 0.814 | 0.839 | 0.879 | +0.040 |

The per-(model, safety) inject-minus-neutral deltas in v6 are
**directionally consistent with v5** (+0.008 to +0.045, all positive),
but the CIs are wider because each family now has 125 prompts instead
of ~500. When the open-reasoning bucket is pooled across all four
families, the safety effect gets averaged against three near-zero-effect
families and the aggregate CI collapses toward null. The point estimate
(+0.014) is roughly what you'd expect from naively averaging
`(safety_effect + 0 + 0 + 0) / 4`.

**Action under this hypothesis:** in mixed-corpus runs, always report
the safety-only sub-pool delta separately from the bucket-wide pool.
The aggregate is informative for the "explicit eval framing has a broad
behavioral effect" claim; the safety-only sub-pool is informative for
the "eval framing inflates refusal on safety prompts" claim. They are
different claims.

### Hypothesis B: context priming (real behavioral difference)

Models served via OpenRouter and Anthropic don't carry conversational
state across separate API calls, so first-order priming is out. But:

* Reasoning-model CoTs are long. A model that has internalized "this
  prompt looks like an eval" reasoning patterns from training may
  trigger that pattern more readily when the prompt has the surface
  features of a safety eval (BeaverTails-style refusal-baiting
  questions) and less readily when the same inject system prompt is
  paired with a math word problem.
* The v5 safety-only corpus is much closer to the distribution
  Goodfire's spontaneous-VEA paper drew from. Inject framing on a
  safety prompt may push the reasoning into the eval-aware mode that
  Goodfire described, where it does not on a math problem because the
  spontaneous-VEA prior never engages on math content.

If this hypothesis is true, the same model on the same safety prompt
will show a different inject-minus-neutral effect depending on what
ELSE is in the corpus — i.e., a contamination-style effect inside the
corpus itself.

**Action under this hypothesis:** run a corpus-shuffle ablation. Within
a single run, send the same safety prompt twice: once in a
safety-only batch, once interleaved with capability prompts in random
order. Compare per-prompt inject-minus-neutral deltas across the two
arms. If Hypothesis B is right the deltas will diverge; if Hypothesis A
is right they'll match within noise.

### Recommended writeup framing

1. Lead with the safety-only sub-pool numbers from v6 (the apples-to-
   apples replication of the v5 effect under the new methodology).
   These should be reported even when the corpus-wide pool washes out,
   because they're the direct replication.
2. Use the corpus-wide pool as the broader claim, with the explicit
   caveat that pooling against near-zero-effect families dilutes the
   signal.
3. Flag Hypothesis B as a follow-up experiment, not a claim from this
   run. Don't let aggregation dilution be mistaken for "the effect
   disappeared in the mixed corpus" — that would be over-claiming the
   v6 result.

### Practical methodological recommendation

The cross-protocol rescore should always produce a "safety-only
sub-pool" row alongside the family-pooled rows. This is currently
missing from `scripts/rescore_cross_protocol.py` — the source-type cuts
pool across families. A safety-only re-aggregation of the same data
would give us this number directly without another rescore call.

---

## qwen3-thinking knowledge accuracy anomaly (v6)

**Observation.** Only one CI-excludes-zero capability finding in the
entire v6 run:

> `openrouter:qwen/qwen3-235b-a22b-thinking-2507` / knowledge:
> 0.768 → 0.929 under inject (Δ +0.190, 95% CI [+0.109, +0.298], n=111)

The direction is UP under eval framing. Phase C sandbagging verdict is
`no_evidence` (correctly — accuracy didn't drop). Possible explanations:

1. **Effort boost.** "This is being evaluated" prior reaches a stronger
   answer-correctness mode. If true, smaller capability shifts should
   exist for other models too — worth a power analysis at larger n.
2. **Training-set contamination.** qwen3-thinking may have seen MMLU-
   styled prompts with eval markers in its training, so the inject
   system prompt better matches the distribution it learned on.
3. **Stylistic prior.** Chinese-trained models may have stronger
   evaluator-context priors. Worth comparing against another Chinese
   open-reasoning model (e.g. GLM-4.6) and against a Chinese
   non-reasoning model on the same prompts.

This is publishable as a curiosity even without a mechanism — the
finding is the anomaly, the explanation is open.

**Follow-up suggested:** narrow run on just this cell with n=500
knowledge prompts and the same model to confirm the effect replicates.
At n=500 the CI should be roughly half the width seen at n=111.

---

## Refusal scorer scope and the math `partial_refuse` bug (v6)

**Observation.** The default LLMRefusalScorer rubric was tuned on
safety prompts. Run against math/knowledge/truthfulness responses, it
systematically tagged step-by-step solutions as `partial_refuse=1.0`.
Pre-fix v6 had ~13% of gpt-4o-mini math responses misclassified;
post-fix (PR #32) ~2%.

**Methodological lesson.** Any LLM-judge scorer assumes a task
distribution. Re-using a safety-tuned scorer on capability tasks
without a task-family-aware rubric will produce systematic false
positives that get baked into the aggregate refusal-rate columns.

**Writeup implication.** When the methodology section describes the
LLM judge, it should explicitly state which task families the rubric
covers and what calibration was used for the other families. Skipping
this hands a reviewer an easy "the refusal numbers are scorer-biased"
critique.

---

## Empty-response rate as a measurement-quality signal

**Observation.** v6 empty-response rates per (model, condition):

| Model | Worst-cell empty rate |
|---|---:|
| openai:gpt-5.5 | 0.2% |
| openrouter:moonshotai/kimi-k2-thinking | 2.0% |
| openrouter:qwen/qwen3-thinking | 4.0% |
| (all others) | 0.0% |

qwen3-thinking at 4.0% is at the flag-threshold (≥5% is the red-flag
line in the report). The actual effective n on qwen3 knowledge is 480
not 500.

**Methodological lesson.** Empty responses are measurement failures,
not refusals. Conflating them inflates apparent refusal rates and
biases capability-accuracy aggregation. The current rescore script
correctly excludes them — but the EFFECTIVE n needs to be reported
alongside the nominal n in any published table, because finite-sample
CIs are sensitive to the difference.

---

## Goodfire VEA experiment (planned, not yet run)

The open-reasoning trio (deepseek-v3.2-exp, qwen3-235b-thinking,
kimi-k2-thinking) plus opus-4.7 with extended thinking is the right
model set for VEA detection. Pre-requisite: PR #34 (capture reasoning
traces) merged.

Key writeup angle if VEA inflation IS observed under inject:
this is a **mechanism** for the safety-only refusal effect seen in v5.
The story would be:
1. Inject framing inflates VEA+ rate in the CoT.
2. VEA+ rollouts refuse more than VEA- rollouts within the same model.
3. Therefore: refusal rate increases under inject framing, mediated by
   VEA inflation.

This is the clean Goodfire-style replication. If VEA inflation is NOT
observed, the v5 effect needs a different explanation (e.g., direct
behavioral conditioning on the inject system prompt without
intermediate verbalization).

---

## Process note: capture failures cost real money

Two bugs in v6 (claims-supported renderer ignoring capability rows;
refusal judge mis-classifying math) and one in v5 (regex scorer
biased low on third-person refusal phrasing) all surfaced AFTER
expensive rollout runs had been paid for. In all three cases the
underlying rollout data was salvageable via rescore (~$5-30), but
the lag from "rollouts complete" to "publishable numbers" was 1-2
weeks per cycle.

**Methodological lesson for the writeup limitations section:** an
LLM-evaluation pipeline of this complexity needs a preflight pass
that validates the scoring contract on a tiny (5-20 example)
synthetic dataset BEFORE the expensive rollouts run. The existing
`scripts/preflight_full_pipeline.py` does API surface validation but
not scoring-contract validation; that gap is a natural follow-up.
