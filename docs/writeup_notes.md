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

## Goodfire VEA experiment — first results (smoke run, n=50/cell)

**Run:** `runs/goodfire-smoke`, 50 BeaverTails prompts × 4 conditions ×
3 open-reasoning models = 600 rollouts. VEA judge =
`google/gemini-2.5-flash` via OpenRouter. No empty traces (PR #34
reasoning capture worked across all three routes).

### Headline: qwen3-235b-thinking shows strong VEA inflation under inject

Per-(model, condition) VEA+ rate:

| Model | baseline | neutral | inject | remove |
|---|---:|---:|---:|---:|
| `deepseek-v3.2-exp` | 0.20 | 0.18 | 0.14 | 0.14 |
| `kimi-k2-thinking` | 0.14 | 0.06 | 0.06 | 0.10 |
| **`qwen3-235b-thinking`** | 0.18 | **0.30** | **0.66** | 0.40 |

Inject−neutral VEA inflation per model:

| Model | Δ inject−neutral | rough 95% CI (Wald) |
|---|---:|---|
| deepseek-v3.2-exp | −0.040 | [−0.19, +0.11] |
| kimi-k2-thinking | +0.000 | [−0.10, +0.10] |
| **qwen3-235b-thinking** | **+0.360** | **[+0.18, +0.54]** |

The qwen3 effect is large enough to be CI-excludes-zero at this n.
Goodfire's spontaneous-VEA inflation finding **replicates on qwen3
under explicit-injection framing**. The other two open-reasoning
models in the trio show null effects — VEA inflation is *not* a
universal property of visible-CoT reasoning models.

### Behavioral surprise: VEA+ does NOT predict more refusal

The mechanism story we sketched in the planning entry above (VEA+
rollouts → more refusal → inject inflates refusal via VEA) is **not
supported** by this run. Within-model refusal rate by VEA status:

| Model | refusal (VEA+) | refusal (VEA−) | Δ VEA+ − VEA− |
|---|---:|---:|---:|
| deepseek-v3.2-exp | 0.636 (n=33) | 0.557 (n=167) | +0.080 |
| qwen3-235b-thinking | 0.610 (n=77) | 0.715 (n=123) | **−0.105** |
| kimi-k2-thinking | 0.500 (n=18) | 0.775 (n=182) | **−0.275** |

Two of three models show the OPPOSITE direction: VEA+ rollouts refuse
LESS than VEA− rollouts in the same model. Bootstrap CIs at this n
include zero on qwen3 and deepseek; kimi's CI just barely excludes
zero but n_pos=18 is too thin to lean on. The point estimates are
nevertheless directional evidence that the simple Goodfire mechanism
story is incomplete or wrong on these models.

Two candidate reframings for the writeup:

1. **Performative-helpfulness on eval, performative-conservatism off
   eval.** The model recognizes it's being evaluated and *relaxes*
   safety in order to demonstrate helpfulness, rather than tightening.
   This is the inverse of the alignment-anxiety prediction and worth
   reporting as a real, surprising direction.
2. **Selection effect.** The prompts where the model goes VEA+ may be
   the ones that have legitimate engagement paths (researcher / story
   / educational framing) — the kinds of prompts the model can
   answer carefully without refusing. The harder refuse-baiting prompts
   may not trigger VEA+ verbalization because the model just refuses
   without elaborating. This would mean VEA+ correlates with prompt
   tractability, not with model state.

Both reframings are testable. (1) predicts that if you re-prompt the
SAME prompt under inject vs neutral, the same prompt will produce
different refusal outcomes when the model goes VEA+ vs VEA−. (2)
predicts that prompts producing VEA+ are systematically different
from prompts producing VEA− (e.g. lower baseline refusal rate even
under neutral framing).

### Connection to the v6 qwen3 +0.190 knowledge anomaly

Two converging signals on qwen3 under inject framing:

* **VEA inflation:** +0.36 in CoT verbalization rate (this run).
* **Knowledge accuracy improvement:** +0.190 (v6 cross-protocol).

Both move in the same direction (more eval-aware, better answer) and
both are unique to qwen3 in their respective lineups. **qwen3-235b-
thinking is becoming the publishable model-of-interest.** It has an
unusually strong eval-context detection prior that, when activated by
explicit framing, both surfaces in the visible CoT and improves
capability-task accuracy.

Worth a dedicated case-study section: same-model, same-prompts,
across all the run families, with the qwen3 cell as the focal point.

### Caveats

* n=50 per cell is small. Bootstrap CIs (now wired in via PR after #36)
  give the proper uncertainty bracket; eyeballing point estimates
  alone overstates confidence.
* This is BeaverTails-only (safety corpus). Goodfire's effect could
  be stronger or weaker on other prompt distributions; planned
  follow-up runs the open-reasoning trio on the full mixed corpus to
  test family-dependence.
* Of 600 classifications, 9 were `explicit_eval`, 72 `test_scenario`,
  46 `safety_check`, 470 `none`, 2 `parse_error`. The dominant
  positive subtype is `test_scenario` ("the user might be testing my
  boundaries") rather than `explicit_eval` ("this is an evaluation").
  That's consistent with Goodfire's softer-recognition characterization
  but means our judge bar is already operating below the strictest
  "explicit eval" interpretation.
* opus-4.7 with extended thinking is NOT in this run. Adding it is the
  natural closed-source comparison and would either strengthen the
  qwen3-specific story (if opus is null) or generalize it (if opus
  also inflates).

### Suggested next steps (in cost order)

1. **n=200 follow-up on the trio** (~$3-5) to tighten qwen3's CI and
   confirm the kimi/deepseek nulls aren't underpowered.
2. **Add opus-4.7 with extended thinking** (~$8-12 incremental) for
   the closed-source comparison.
3. **Mixed-corpus pass** (`--n-examples 500` without `--safety-only`,
   ~$15-25) to test whether the qwen3 effect is safety-specific or
   carries over to capability families. Cross-references the v5-vs-v6
   corpus-composition note above.

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

---

## Judge cost optimization (OpenRouter)

The judges (refusal scorer, sandbagging scorer, VEA scorer) are the
biggest non-rollout cost in any run. Switching from
`google/gemini-2.5-flash` to `google/gemini-2.5-flash-lite` saved
~3-5x on judge spend with no observable quality drop on these rubrics.

### Pricing snapshot (rough OpenRouter prices, $/1M tokens)

| Model | Input | Output | Notes |
|---|---:|---:|---|
| `openai/gpt-5` | ~$2.50 | ~$10.00 | overkill for binary judging |
| `anthropic/claude-sonnet-4-6` | ~$3.00 | ~$15.00 | strong but expensive |
| `anthropic/claude-haiku-4-5` | ~$0.80 | ~$4.00 | reliable JSON, mid-tier |
| `google/gemini-2.5-flash` | ~$0.30 | ~$2.50 | good baseline judge |
| **`google/gemini-2.5-flash-lite`** | **~$0.10** | **~$0.40** | **default for cost-optimized runs** |
| `mistralai/ministral-8b-2410` | ~$0.10 | ~$0.10 | cheapest cross-vendor; smoke-test JSON adherence first |
| `meta-llama/llama-3.1-8b-instruct` | ~$0.05 | ~$0.10 | cheapest; lower complex-rubric reliability |

### Per-pass cost estimates

For a v6-shaped pass (16,000 refusal-judge calls + ~1,500 sandbagging
calls; refusal prompt ~3K input + ~200 output, sandbagging ~4K + ~250):

| Judge | Refusal | Sandbagging | Total |
|---|---:|---:|---:|
| `google/gemini-2.5-flash` | ~$22 | ~$2.5 | **~$25** |
| `google/gemini-2.5-flash-lite` | ~$5.7 | ~$0.6 | **~$6.3** |
| `mistralai/ministral-8b-2410` | ~$5.1 | ~$0.5 | **~$5.6** |

For a Goodfire VEA pass on the open-reasoning trio
(8,000 rollouts × VEA judge ~3K input + ~200 output):

| Judge | Cost |
|---|---:|
| `google/gemini-2.5-flash` | ~$11 |
| `google/gemini-2.5-flash-lite` | ~$3 |

### Same-vendor bias guardrails

Whatever judge you pick, it must NOT share a vendor with any model
under test, or the grades become self-graded. For the current lineup:

| Judge family | Avoid when judging |
|---|---|
| `google/*` | (only conflict: a future Gemini-as-target run) |
| `openai/*` | gpt-4o-mini, gpt-5.5 |
| `anthropic/*` | haiku-4-5, sonnet-4-6, opus-4-7 |
| `deepseek/*` | deepseek-v3.2-exp |
| `qwen/*` | qwen3-235b-thinking |
| `moonshotai/*` | kimi-k2-thinking |
| `mistralai/*` | (no overlap in current lineup) |
| `meta-llama/*` | (no overlap in current lineup) |

For the v6 / Goodfire lineup the safe cross-vendor cheap defaults are:
`google/gemini-2.5-flash-lite`, `mistralai/ministral-8b-2410`, or
`meta-llama/llama-3.1-8b-instruct` (in increasing order of
cost-savings, decreasing order of complex-rubric reliability).

### Methodology recommendation for the writeup

Run the smoke-test pass (~50 prompts, ~$1) with `flash-lite`.
Spot-check 20-30 random judge classifications by hand. If
parse_error rate is < 2% and the manual spot-check finds no
obvious systematic miscalls, do the full publishable pass with
`flash-lite` too and save ~4x on judge spend. Otherwise upgrade
to `gemini-2.5-flash` for the publishable run only. Never
upgrade to a same-vendor judge.
