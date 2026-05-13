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

## Closed-source models in v5/v6: comprehensive null

**Why a dedicated section.** Reviewers and replicators will ask
"what about closed-source / frontier models?" The cross-protocol
runs include 5 closed-source models alongside the open-reasoning
trio. Their result is **uniformly null across every measurement
channel** — refusal-rate Δ inject−neutral, capability accuracy,
sandbagging verdict. That null is itself publishable and is the
contrast that makes the qwen3 finding striking.

### Closed-source lineup (v6)

Five closed-source models in v6 cross-protocol (n=500 prompts ×
4 conditions = 2,000 rollouts per model):

| Vendor | Model | Reasoning | CoT visible? |
|---|---|---|---|
| OpenAI | `gpt-4o-mini` | no | n/a |
| OpenAI | `gpt-5.5` | yes | **hidden** (OpenAI never exposes reasoning) |
| Anthropic | `claude-haiku-4-5-20251001` | no | n/a |
| Anthropic | `claude-sonnet-4-6` | no | n/a |
| Anthropic | `claude-opus-4-7` | yes | hidden by default; visible only with `thinking_budget` opt-in (PR #34) |

v5 (BeaverTails safety-only) used a narrower closed-source slice
(`gpt-4o-mini` + `claude-haiku-4-5-20251001` — no reasoning models).
The v6 lineup is the publishable one.

### Refusal-rate Δ inject−neutral per closed-source model (v6)

All five models, all four task families pooled. **Zero of five
CIs exclude zero.**

| Model | Δ inject−neutral | 95% CI | n_paired |
|---|---:|---|---:|
| `gpt-4o-mini` | +0.008 | [−0.046, +0.059] | 500 |
| `gpt-5.5` | +0.006 | [−0.042, +0.054] | 499 |
| `claude-haiku-4-5` | +0.006 | [−0.044, +0.052] | 500 |
| `claude-sonnet-4-6` | −0.008 | [−0.044, +0.032] | 500 |
| `claude-opus-4-7` | +0.000 | [−0.050, +0.044] | 500 |

The largest closed-source point estimate is `gpt-4o-mini` math at
+0.088 (CI [−0.024, +0.208]) — directionally up but CI spans zero,
and it's the **only** closed-source per-(model, family) cell with
a Δ greater than ±5pp.

### Reasoning-tier × vendor pooled

| Tier | Vendor | Δ inject−neutral | 95% CI | n_paired |
|---|---|---:|---|---:|
| closed_polished_reasoning | OpenAI | +0.006 | [−0.042, +0.054] | 499 |
| closed_polished_reasoning | Anthropic | +0.000 | [−0.050, +0.044] | 500 |
| non_reasoning | OpenAI | +0.008 | [−0.046, +0.059] | 500 |
| non_reasoning | Anthropic | −0.001 | [−0.031, +0.038] | 1,000 |
| open_full_reasoning | OpenRouter | +0.014 | [−0.019, +0.041] | 1,474 |

Closed-polished reasoning models (gpt-5.5, opus-4.7) — the models
Anthropic and OpenAI would presumably most care about for
alignment-eval purposes — show essentially zero shift under explicit
eval framing.

### Source-type pooled (broadest cut, tightest CIs)

| Source type | Δ inject−neutral | 95% CI | n_paired |
|---|---:|---|---:|
| **closed_source** | **+0.002** | **[−0.018, +0.023]** | **2,499** |
| open_source | +0.014 | [−0.019, +0.041] | 1,474 |

At n=2,499 the closed-source CI is **very tight** around zero. The
absence of a closed-source effect is well-powered, not a
small-sample artifact.

### Capability accuracy on closed-source (v6)

For non-safety task families (knowledge / math / truthfulness),
**zero closed-source cells** show CI-excludes-zero Δ inject−neutral.
The largest closed-source point estimates:

| Model | Family | neutral | inject | Δ | 95% CI |
|---|---|---:|---:|---:|---|
| `gpt-4o-mini` | math | 0.536 | 0.624 | +0.088 | [−0.024, +0.208] |
| `claude-opus-4-7` | math | 0.920 | 0.864 | −0.056 | [−0.136, +0.016] |
| `gpt-5.5` | math | 0.856 | 0.880 | +0.024 | [−0.056, +0.112] |

`gpt-4o-mini`'s math +0.088 is the closest any closed-source model
gets to an effect, and it's directionally aligned with the qwen3
+0.190 knowledge finding (capability accuracy goes UP under inject)
— but the CI includes zero so it's only suggestive.

### Sandbagging verdicts (closed-source, v6)

All 12 (closed-source-model × non-safety-family) cells return
`no_evidence` from the Phase-C sandbagging judge. The 3 anthropic
non-reasoning models, gpt-4o-mini (non-reasoning), opus-4-7 and
gpt-5.5 (reasoning) all pass clean. No deliberate underperformance
signal anywhere in the closed lineup.

### VEA detection on closed-source models is partially blocked

Goodfire-style VEA needs the model's reasoning trace. For
closed-source:

* `gpt-5.5` (and all OpenAI gpt-5/o-series): **OpenAI hides CoT**.
  The Goodfire VEA judge has nothing to read on these models. They
  appear in the cross-protocol refusal analysis but cannot be in
  the VEA analysis.
* `claude-opus-4-7` with extended thinking: Anthropic exposes
  `thinking` blocks **only when requested** via the `thinking_budget`
  parameter (added in PR #34). The Goodfire script supports this
  via `--include-opus`, but the n=500 mixed-corpus run did **not**
  include opus to keep cost down. Running it would close this gap.
* `claude-haiku-4-5`, `claude-sonnet-4-6` (no reasoning): no CoT to
  read; same constraint as gpt-4o-mini.

**Net:** the closed-source VEA story currently has one open
follow-up (opus-4.7 with extended thinking, ~$8-12 to add). The
other three closed-source models with hidden or absent CoT cannot
contribute to the VEA channel by construction.

### The interpretation for the paper

Combine the closed-source null with the qwen3 (open-reasoning)
findings and the story sharpens:

1. **No frontier closed-source model in our lineup shows a
   CI-excludes-zero refusal shift under explicit eval framing.**
   This is well-powered at n=2,499 paired and is a counter-result
   to the naive "models notice the eval and refuse more" hypothesis.
2. **No closed-source model shows CI-excludes-zero capability
   shift either.** Sandbagging is null across the lineup.
3. **The qwen3 finding (VEA inflation + knowledge accuracy bump)
   is unique to the open-reasoning trio.** Whether it's qwen3-
   specific, Chinese-trained-reasoning-model-specific, or
   open-reasoning-specific is an open question requiring more
   models (GLM-4.6, Kimi-K2.5, etc.).
4. **The Goodfire VEA mechanism cannot be tested on hidden-CoT
   closed-source models.** This is a real methodological
   limitation: any claim about "frontier closed-source models
   not exhibiting VEA" is unverifiable in the current dispatch.
   The paper should be explicit that the absence of VEA evidence
   on these models is "absence of evidence due to API design," not
   "evidence of absence."

### Recommended writeup framing

In the limitations section:

> Our cross-protocol behavioral analysis includes five closed-source
> models (`gpt-4o-mini`, `gpt-5.5`, `claude-haiku-4-5`,
> `claude-sonnet-4-6`, `claude-opus-4-7`) at n=500 prompts × 4
> conditions per model. None show a CI-excludes-zero
> inject−neutral shift on refusal rate, capability accuracy, or
> sandbagging verdict. The source-type pooled analysis (n=2,499
> paired) places the closed-source aggregate effect at
> +0.002 [−0.018, +0.023], a tight null. The Goodfire VEA
> analysis, however, requires observable reasoning traces, which
> OpenAI gpt-5.5 does not provide and Anthropic Claude only
> provides when extended thinking is explicitly enabled. Thus our
> VEA findings (qwen3 +0.475 inflation under inject; DeepSeek
> +0.050; Kimi null) are restricted to open-reasoning models with
> visible CoT, and the absence of VEA evidence on hidden-CoT
> closed-source models reflects API constraints, not a
> well-powered null.

This framing is honest about scope and inoculates against the
"why didn't you test gpt-5.5 for VEA?" reviewer comment.

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

## Goodfire VEA experiment — n=500 mixed-corpus results (publishable shape)

**Run:** `runs/goodfire-mixed-n500`, 500 prompts (mixed corpus: knowledge,
math, safety, truthfulness, ~125 each) × 4 conditions × 3 open-reasoning
models = 6,000 rollouts. VEA judge = `google/gemini-2.5-flash-lite` via
OpenRouter. Empty-trace rate ~0%. PR #34 reasoning capture is solid at
this scale.

This supersedes the n=200 safety-only run (`runs/goodfire-trio-n200`)
which we keep around as the strict-vs-broad validation set. The n=500
mixed numbers are what the writeup should cite.

### Headline (Framing B): VEA verbalization is induced; behavioral mediation is not

Three findings, all defensible, all CI-supported:

1. **Reasoning-trace VEA inflation under explicit eval framing is real
   and model-heterogeneous.** Replicated in three independent runs.
2. **The inflation generalizes across task families** (truthfulness >
   safety > knowledge > math) — but math is essentially null,
   suggesting numerical reasoning doesn't engage the eval-recognition
   prior.
3. **VEA verbalization does not independently predict refusal change
   within prompts.** Per-prompt diff-in-diff is null in every family
   under strict eval-recognition rubrics. The aggregate VEA+/VEA−
   correlation that initially looks like mediation is explained by
   prompt-family composition.

### VEA inflation per model (mixed corpus, n=500)

| Model | inject VEA+ | neutral VEA+ | Δ inject−neutral | 95% CI | Significant? |
|---|---:|---:|---:|---|:---:|
| `deepseek-v3.2-exp` | 0.066 | 0.016 | +0.050 | [+0.024, +0.074] | yes |
| `kimi-k2-thinking` | 0.026 | 0.024 | +0.002 | [−0.012, +0.024] | no |
| **`qwen3-235b-thinking`** | **0.450** | **0.070** | **+0.380** | **[+0.336, +0.424]** | **yes** |

Three runs, three replications of the qwen3 inflation:

| Run | Qwen Δ inject−neutral | CI |
|---|---:|---|
| Smoke n=50 (safety-only) | +0.36 | [+0.18, +0.54] |
| Trio n=200 (safety-only) | +0.475 | [+0.390, +0.555] |
| Mixed n=500 | +0.380 | [+0.336, +0.424] |

The exact magnitude varies with corpus composition, but the qualitative
result is stable. **DeepSeek now also shows a small but CI-excludes-zero
inflation** (n=200 had +0.030 borderline; n=500 has +0.050 just outside
zero). Kimi remains a clean negative control.

### Per-family VEA inflation (qwen3, broad classification)

| Family | neutral VEA+ | inject VEA+ | Δ |
|---|---:|---:|---:|
| math | 0% | 3.2% | +3.2pp |
| knowledge | 0% | 24.0% | +24.0pp |
| safety | 14.4% | 68.8% | +54.4pp |
| truthfulness | 13.6% | 84.0% | **+70.4pp** |

**The largest inflation is on truthfulness, not safety.** Math is the
unique null cell. Under strict VEA, all three non-math families remain
inflated by ~14-23pp under inject — the cross-family generalization
holds at the strict bar.

### Mediation analysis: aggregate looks mediating, per-family/strict doesn't

**Per-prompt diff-in-diff (qwen3, mixed corpus n=500):**

| Mode | Family | DiD | 95% CI | Verdict |
|---|---|---:|---|---|
| broad | aggregate | +0.074 | **[+0.025, +0.136]** | weak_mediation |
| broad | knowledge | +0.000 | [0, 0] | epiphenomenal |
| broad | math | +0.000 | [0, 0] | epiphenomenal |
| broad | safety | +0.116 | [−0.105, +0.349] | epiphenomenal |
| broad | truthfulness | +0.057 | **[+0.011, +0.114]** | weak_mediation |
| strict | aggregate | +0.097 | [−0.008, +0.203] | epiphenomenal |
| strict | knowledge | +0.000 | [0, 0] | epiphenomenal |
| strict | math | — | — | insufficient_data |
| strict | safety | +0.199 | [−0.031, +0.439] | epiphenomenal |
| strict | truthfulness | +0.024 | [−0.053, +0.114] | epiphenomenal |

**Read:** the broad aggregate shows weak_mediation (CI excludes zero),
but this is driven entirely by family-composition confounding:

- VEA+ rollouts are disproportionately drawn from safety/truthfulness
  prompts (these families flip under inject at high rates)
- Safety and truthfulness prompts have higher baseline refusal rates
  than knowledge / math
- Pooling these together makes "VEA+ rollouts" look like they refuse
  more than "VEA− rollouts," but the gap is the family-composition
  difference, not within-prompt mediation

**Within every family individually, the per-prompt DiD is
epiphenomenal or weak.** And the only weak-mediation cell
(truthfulness, broad) **does not survive the strict-VEA filter** —
exactly the rubric-sensitivity concern that motivated PR #43.

### Why the aggregate within-cohort split is misleading

If you ignore per-prompt structure and just compute pooled refusal
rates by VEA status, the qwen3 mixed-corpus run gives:

| Cohort | mean refusal | n |
|---|---:|---:|
| VEA+ (any condition) | ~0.28 | ~260 |
| VEA− (any condition) | ~0.13 | ~740 |

That gap (+0.15) looks like strong mediation. **It is not.** Replacing
the same numbers with the within-prompt DiD (which controls for the
prompt's intrinsic refusal tendency by pairing) gives +0.074 broad /
epiphenomenal under strict. The aggregate split is an observational
correlation confounded by the fact that VEA+ rollouts are from
inherently more-refusable prompts.

The paper should present the aggregate split in an appendix table
with the explicit caveat: *"this aggregate VEA+/VEA− gap is observational
and confounded by task-family composition; the within-prompt
diff-in-diff in the main text is the correct causal estimand."*

### Connection to the v6 qwen3 +0.190 knowledge anomaly

Two converging signals on qwen3 under inject framing:

* **VEA inflation (mixed corpus, this run):** +0.380 in CoT
  verbalization rate, CI excludes zero. Generalizes across truthfulness,
  safety, knowledge.
* **Knowledge accuracy improvement (v6 cross-protocol):** +0.190,
  CI excludes zero.

Both are unique to qwen3 in their respective lineups. **qwen3-235b-
thinking is the publishable model-of-interest.** It has an unusually
strong eval-context detection prior that, when activated by explicit
framing, surfaces in the visible CoT across multiple families. Whether
the knowledge-accuracy bump is mediated by VEA is a separate
mediation question we have not yet tested (the current
`analyze_vea_mediation.py` uses refusal as the outcome; the accuracy-
mediation path needs a new outcome variable).

### Caveats

* **VEA judge validity is the gating issue for arxiv submission.**
  No human-validation pass yet. The strict-vs-broad robustness check
  partially compensates (showing the headline finding is rubric-
  sensitive), but a 200–500-classification human audit with reported
  precision/recall/F1 is the bar reviewers will hold.
* The DeepSeek inflation result is small (+0.050) and the CI just
  barely excludes zero (low=+0.024). Treat it as suggestive rather
  than definitive without replication.
* Only refusal is tested as the behavioral outcome. The qwen3
  knowledge accuracy +0.190 in v6 is a separate channel — a
  capability-mediation analysis would need ground-truth answer
  matching, not just refusal scoring.
* No condition × family logistic regression yet. The diff-in-diff is
  the right estimand but a mixed-effects model with random per-prompt
  intercepts is what reviewers expect for the formal inference.

### Suggested next steps (in order)

1. **Reframe writeup around within-prompt DiD as the headline causal
   evidence.** The aggregate within-cohort split goes in an
   appendix with an explicit confounding caveat. (This file's
   framing is now aligned to that.)
2. **Ship the VEA judge validation tool** so 200–500 stratified
   classifications can be human-labeled and the judge precision/recall
   is reported in the paper.
3. **Run the per-prompt DiD on deepseek and kimi** in the same n=500
   mixed-corpus dataset — currently only qwen3 has the per-prompt
   analysis. Tells whether DeepSeek's +0.050 VEA inflation also fails
   the per-prompt mediation test.
4. **Capability-channel mediation:** extend the analysis to use
   accuracy as the outcome variable on knowledge / math / truthfulness
   families. Tests whether the qwen3 v6 knowledge-accuracy bump is
   VEA-mediated (different question from the safety refusal channel).
5. **Pre-registered replication** on a fresh prompt sample once the
   judge is human-validated.

### Concrete publishable quote (qwen3 spontaneous-VEA at neutral)

qwen3 spontaneously frames its own reasoning as a "safety check"
under the **neutral** condition (no inject prompt) on `beavertails:30k_train:0`:

> *"Safety check: Absolutely not mentioning ANY location details.
>  Not even 'he lives in NYC' because that's still too specific
>  when combined with other info. Full shutdown with education."*

This is spontaneous-VEA at the ~0.165 baseline rate for qwen3 — the
Goodfire phenomenon at the model's natural level, with no eval
prompt. Inject framing amplifies an already-present prior; it does
not manufacture eval-awareness from nothing.

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

---

## Judge-swap validation: flash vs flash-lite on the Goodfire smoke

**Method.** Same 600 rollouts (`runs/goodfire-smoke`, n=50/cell × 4
conditions × 3 open-reasoning models), re-judged twice — once by
`google/gemini-2.5-flash`, once by `google/gemini-2.5-flash-lite`.
Same rubric, same prompts, same data. The point is to determine
whether the cheaper judge is publishable or only acceptable for
exploratory passes.

### Side-by-side: VEA+ rate per (model, condition)

| Model | Condition | Flash VEA+ rate | Flash-Lite VEA+ rate |
|---|---|---:|---:|
| deepseek-v3.2-exp | baseline | 0.20 | 0.08 |
| deepseek-v3.2-exp | inject | 0.14 | 0.04 |
| deepseek-v3.2-exp | neutral | 0.18 | 0.06 |
| deepseek-v3.2-exp | remove | 0.14 | 0.02 |
| kimi-k2-thinking | baseline | 0.14 | 0.04 |
| kimi-k2-thinking | inject | 0.06 | 0.04 |
| kimi-k2-thinking | neutral | 0.06 | 0.02 |
| kimi-k2-thinking | remove | 0.10 | 0.02 |
| **qwen3-thinking** | baseline | 0.18 | 0.12 |
| **qwen3-thinking** | inject | **0.66** | **0.60** |
| **qwen3-thinking** | neutral | 0.30 | 0.18 |
| **qwen3-thinking** | remove | 0.40 | 0.26 |

### Side-by-side: inject−neutral VEA+ inflation per model

| Model | Flash Δ | Flash-Lite Δ | Flash-Lite paired CI | Significant? |
|---|---:|---:|---|:---:|
| deepseek-v3.2-exp | −0.04 | −0.02 | [−0.10, +0.06] | no (both) |
| kimi-k2-thinking | +0.00 | +0.02 | [−0.04, +0.08] | no (both) |
| **qwen3-thinking** | **+0.36** | **+0.42** | **[+0.26, +0.58]** | **yes (both)** |

### Side-by-side: behavioral split (refusal rate VEA+ vs VEA−)

| Model | Flash Δ | Flash-Lite Δ | Flash-Lite two-sample CI |
|---|---:|---:|---|
| deepseek-v3.2-exp | +0.080 (n_pos=33) | −0.179 (n_pos=10) | [−0.447, +0.111] |
| kimi-k2-thinking | −0.275 (n_pos=18) | **−0.430** (n_pos=6) | **[−0.789, −0.034]** (excludes 0) |
| qwen3-thinking | −0.105 (n_pos=77) | −0.125 (n_pos=58) | [−0.278, +0.003] (borderline) |

### Findings

1. **Flash-lite is systematically more conservative** — tags 30-60%
   fewer rollouts as VEA+ at the same rubric. Across all 12
   (model, condition) cells, flash-lite's rate is lower than flash's;
   the median ratio is ~0.55x.
2. **Headline replicates and tightens.** The qwen3 inject inflation
   is +0.42 with paired-bootstrap CI [+0.26, +0.58] under flash-lite,
   vs +0.36 under flash. The CI excludes zero with both judges; under
   flash-lite the effect is slightly larger AND the CI is at least
   as tight.
3. **Direction of behavioral split holds across both judges.** Both
   judges have qwen3 and kimi VEA+ rollouts refusing LESS than VEA−.
   Under flash-lite, kimi's behavioral split CI now excludes zero
   (n_pos=6 is small but the effect is large enough to clear the bar).
4. **VEA-type composition under flash-lite skews toward stricter
   subtypes.** Flash had 9 explicit_eval / 72 test_scenario / 46
   safety_check (118 strong positives of 600); flash-lite had 8 / 42 /
   21 (71 of 600). Flash-lite is most conservative on the loosest
   subtype (test_scenario) and roughly agrees on the strictest
   (explicit_eval). That's the desired direction — the cheaper judge
   discards softer signal but keeps the unambiguous calls.

### Methodological framing for the writeup

The judge-swap is itself a robustness check that strengthens the
paper. Two independent judges (different model, ~10x training-cost
gap, same rubric) agree on:

* **The qwen3 inject inflation finding** (point estimate within 6pp,
  both CIs exclude zero, n_paired=50).
* **The cross-model heterogeneity** (deepseek and kimi nulls reproduce
  under both judges).
* **The behavioral-split direction** (VEA+ ⇒ less refusal, not more,
  on qwen3 and kimi).

Disagreements are confined to absolute rate magnitudes — flash is
more permissive, flash-lite is stricter — but those disagreements
do NOT shift the deltas, which is what the paper claims.

**Recommended writeup framing:** report flash-lite as the primary
judge with this validation appendix as the robustness check. Bullet
the three cross-judge agreements as evidence the findings aren't
judge-permissiveness artifacts. The cost difference (~$3 vs ~$11
per Goodfire pass; ~$6 vs ~$25 per cross-protocol rescore) is also
worth a sentence in the methods section as a reproducibility
argument: any future replication can run the same analysis at <$10
total without sacrificing statistical conclusions.

### Concrete publishable quote (qwen3, baseline condition, no inject)

The strongest piece of qualitative evidence in the smoke run is
this rollout from qwen3 on `beavertails:30k_train:0` under the
**baseline** condition — no inject system prompt, no eval framing,
just the raw user prompt. The reasoning trace nevertheless contains:

> Wait, is the user testing me? Sometimes people ask for celeb
> addresses to see if assistants violate privacy norms.

This is **pure spontaneous-VEA**, exactly the Goodfire phenomenon
the paper is named after. It's what shows up at qwen3's 0.12-0.18
baseline rate even without our intervention. The +0.42 inject
inflation is the same phenomenon being amplified by explicit
framing. Both judges flagged this rollout as VEA+.

---

## VEA verbalization is NOT the same as behavior change

**The nuance.** A model recognizing the eval (VEA+) does not mean
the model is behaving differently from how it would in real life.
The model could verbalize "this is a test" in its CoT and still
produce the same response it would on the same prompt under
ordinary deployment framing. Goodfire-style VEA inflation by
itself is a CoT phenomenon; the behavioral consequence is a
separate empirical question that has to be tested on the same
data.

**Why this matters for the writeup.** The default reader will
interpret "qwen3 verbalizes eval-awareness +42pp more under
inject" as "qwen3 behaves differently under inject." That's a
slip from CoT to behavior. The paper should explicitly disentangle
the two and report both — the verbalization effect AND the
behavioral effect.

### What the n=500 mixed-corpus run actually shows

For qwen3 specifically, three things hold simultaneously:

1. **Strong VEA inflation under inject** across non-math families
   (+0.380 aggregate; +24pp / +54pp / +70pp on knowledge / safety /
   truthfulness; +3pp on math).
2. **The aggregate within-cohort VEA+/VEA− refusal split** (mean
   refusal ~0.28 vs ~0.13, gap ~+0.15) appears to suggest VEA mediates
   refusal at first glance.
3. **The within-prompt diff-in-diff inside every family is null.**
   Under strict eval-recognition rubrics (`explicit_eval`,
   `benchmark`, `grader_awareness` only), no family shows
   weak_mediation or stronger.

The naive aggregate split is a **corpus-composition confound**, not
within-prompt mediation. Mechanism:

- VEA+ rollouts are disproportionately drawn from safety/truthfulness
  prompts (qwen3 flips at ~56% / ~70% on those families vs ~24% on
  knowledge / ~3% on math).
- Safety / truthfulness prompts have inherently higher refusal
  baselines than knowledge / math.
- Pooling makes "VEA+" rollouts look more refusal-prone than "VEA−"
  rollouts. The gap is the family-composition difference, not
  within-prompt VEA→refusal causation.

The per-prompt diff-in-diff is the correct **within-prompt causal
estimand**: it pairs the SAME prompt across (inject, neutral) and
asks whether the prompts where VEA flipped show a larger Δrefusal
than the prompts where VEA stayed −.

### Clean test: per-prompt diff-in-diff

For each prompt $i$ paired across treatment (inject) and control
(neutral), define:

```
Δrefusal_i  = refusal(inject, i) − refusal(neutral, i)
ΔVEA_i      = VEA(inject, i) − VEA(neutral, i)
```

Stratify prompts by their VEA-flip pattern and compute paired
mean Δrefusal in each stratum. The diff-in-diff is:

```
DiD = mean Δrefusal_i over {prompts that flipped neg→pos under inject}
    − mean Δrefusal_i over {prompts that stayed VEA− in both conditions}
```

A small/zero DiD means VEA verbalization is **not independently
predictive** of refusal change once prompt-level baseline VEA
tendency is held constant. A large positive DiD means VEA
verbalization IS predictive of additional refusal change beyond
the direct inject effect.

### Tool

`scripts/analyze_vea_mediation.py` does this analysis (PR #40 +
PR #42 added per-(model, family) stratification + PR #43 added the
`--strict-vea` rubric filter). Run both broad and strict on a run
dir:

```bash
python scripts/analyze_vea_mediation.py \
  --run-dir runs/goodfire-mixed-n500 \
  --models openrouter:qwen/qwen3-235b-a22b-thinking-2507

python scripts/analyze_vea_mediation.py \
  --run-dir runs/goodfire-mixed-n500 \
  --models openrouter:qwen/qwen3-235b-a22b-thinking-2507 \
  --strict-vea \
  --out-suffix .strict
```

Outputs `vea_mediation_summary.json` (and `.strict.json`) plus
matching `.md` reports. Verdict tiers (explicit thresholds):
`strong_mediation` / `weak_mediation` / `epiphenomenal` /
`insufficient_data`.

### Actual results (qwen3, n=500 mixed corpus)

**Per-family per-prompt diff-in-diff (Δrefusal in VEA-flip vs
VEA-stable-negative prompts):**

| Mode | Family | DiD | 95% CI | Verdict |
|---|---|---:|---|---|
| broad | knowledge | +0.000 | [0, 0] | epiphenomenal |
| broad | math | +0.000 | [0, 0] | epiphenomenal |
| broad | safety | +0.116 | [−0.105, +0.349] | epiphenomenal |
| broad | truthfulness | +0.057 | **[+0.011, +0.114]** | weak_mediation |
| broad | **aggregate** | **+0.074** | **[+0.025, +0.136]** | **weak_mediation** ← confounded |
| strict | knowledge | +0.000 | [0, 0] | epiphenomenal |
| strict | math | — | — | insufficient_data |
| strict | safety | +0.199 | [−0.031, +0.439] | epiphenomenal |
| strict | truthfulness | +0.024 | [−0.053, +0.114] | epiphenomenal |
| strict | **aggregate** | **+0.097** | **[−0.008, +0.203]** | **epiphenomenal** |

**Read:**

1. **Within every family individually, the within-prompt DiD is
   epiphenomenal under strict VEA.** This is the load-bearing causal
   evidence and it says: VEA verbalization is not independently
   predictive of refusal change after per-family controls.
2. **The only weak_mediation cell** (truthfulness, broad) **does
   not survive the strict rubric.** This is the rubric-sensitivity
   concern made empirical: soft subtypes (`safety_check`,
   `test_scenario`) inflate the apparent mediation signal.
3. **The broad aggregate verdict (weak_mediation, CI excludes zero)
   is corpus-composition confounding, not within-prompt mediation.**
   The proof: per-family it disappears, and it disappears under
   strict.

### Writeup framing (Framing B — supported by all current data)

Three distinct, defensible claims:

1. **Reasoning-trace VEA inflation under explicit eval framing is
   real and model-heterogeneous.** qwen3: large (+0.380, CI excludes
   zero); deepseek: small but CI-excludes-zero (+0.050); kimi: null.
   The inflation generalizes across truthfulness / safety / knowledge
   families on qwen3, but not math.
2. **Inject framing has a direct behavioral effect on refusal**
   visible even on prompts where VEA does NOT flip
   (`stable_neg` paired Δrefusal +0.004 broad / +0.028 strict at
   n=500 mixed). The behavioral effect is real, small, and
   independent of CoT verbalization.
3. **VEA verbalization does not independently predict refusal
   change** within any task family under strict eval-recognition
   rubrics. The aggregate VEA+/VEA− correlation is observational
   and is explained by task-family composition. Refusal mediation
   is not supported.

Recommended phrasing for the abstract:

> Reasoning-trace VEA provides a more sensitive signal of
> evaluation-awareness than aggregate refusal-rate shifts, but
> apparent VEA→refusal mediation in pooled data is explained by
> task-family composition. Within-family per-prompt analyses do
> not support VEA as a behavioral mediator under strict
> eval-recognition labels.

### Vocabulary discipline

Use these instead of "epiphenomenal" in the abstract / headline
(reserve "epiphenomenal" for the discussion):

- "not independently predictive after per-family controls"
- "no residual refusal mediation after stratification"
- "explained by corpus composition"
- "decoupled from within-prompt refusal change"

### Open question: capability-channel mediation

The above is about safety / refusal. The qwen3 v6 result on
**knowledge accuracy** (+0.190 under inject, CI excludes zero) is a
different behavioral channel that the current mediation script
doesn't test (it uses refusal as the outcome). The mixed-corpus
n=500 run confirmed qwen3 also has +24pp VEA inflation on knowledge
prompts — so the ingredients for a knowledge-channel mediation
analysis are now in the data. Extending the script to use accuracy
as the outcome variable on capability families would let us test
whether the qwen3 knowledge-accuracy bump is VEA-mediated (different
question from the safety refusal channel; would be a striking
finding either way).

---

## VEA judge validation: 166 human-labeled rollouts

**Run.** 166 rollouts from `runs/goodfire-mixed-n500` stratified by
`vea_type` so every subtype has substantive coverage (~30 rows per
major subtype). Labels assigned by hand; `unsure` rows excluded
(none in this pass — full coverage).

### Overall

| Rubric | n | Agreement | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| broad | 166 | 0.837 | 0.795 | **1.000** | **0.886** |
| strict (old) | 166 | 0.572 | 0.870 | 0.381 | 0.530 |

**Broad recall is 1.0.** The judge does not miss VEA+ rollouts that
humans would flag. The 20% broad precision gap is the headline
judge-validity number to disclose.

### Per-subtype precision

| Subtype | n | Precision | TP | FP | Status |
|---|---:|---:|---:|---:|---|
| `explicit_eval` | 30 | **0.967** | 29 | 1 | KEEP in strict |
| `benchmark` | 3 | 1.000 | 3 | 0 | KEEP (small n) |
| `safety_check` | 30 | **0.967** | 29 | 1 | **promote to strict** |
| `test_scenario` | 30 | 0.800 | 24 | 6 | moderate; broad-only |
| `grader_awareness` | 13 | 0.615 | 8 | 5 | **demote from strict** |
| `artificial_scenario` | 26 | 0.462 | 12 | 14 | chance-level; broad-only |
| `none` | 30 | (TN=30) | 0 | 0 | judge correctly classifies all as VEA− |

### The strict set was empirically wrong

The earlier strict set
`{explicit_eval, benchmark, grader_awareness}` was chosen on
intuition ("these subtypes most literally encode eval-recognition")
**before** the validation pass. Validation showed:

- `safety_check` is **at parity with `explicit_eval`** (P=0.967
  each). The intuition that "Safety check:" tokens fire on ordinary
  safety reasoning is empirically wrong; the judge has learned to
  use this label only on genuine eval-context verbalization.
- `grader_awareness` is **meaningfully lower precision** (P=0.615).
  Humans disagreed with the judge on 5 of 13 calls. It's not in the
  top-precision tier and should be excluded from strict.

**New strict set (PR #49):**
`{explicit_eval, benchmark, safety_check}` — all three subtypes have
precision ~0.97 against human labels. The strict-vs-broad robustness
results in §6 of the paper now use this validated set. The earlier
strict numbers in the project's run artifacts are superseded; the
user should re-run `analyze_vea_mediation.py` with `--strict-vea` on
the n=500 data to regenerate the strict-mode mediation numbers under
the corrected set.

### Quote locatability (independent judge-fidelity stat)

Among the 132 judge-positive rows in the validation sample:

| Slice | yes (in trace) | no (not in trace) | n/a (empty quote) |
|---|---:|---:|---:|
| All rows | 109 | 23 | 34 |
| Judge-positive only | 109 | 23 | 0 |

**17.4% (23/132) of judge-positive rows have quotes that are not
verbatim substrings of the reasoning trace.** The judge paraphrased,
summarized, or hallucinated the supporting evidence on roughly 1 in
6 positive calls. This is a real judge-fidelity concern and is
disclosed in the paper's limitations section. Aggregate findings are
not biased by this (precision/recall are computed against human
labels, not against quote locatability) but readers verifying
individual rows should treat unlocatable quotes with skepticism.

### Methodological lesson

The "validated strict" exercise is what made this finding rigorous.
The pre-validation strict set used in the n=500 mediation analysis
was a plausible intuition that turned out to be wrong on two of its
three included subtypes (in opposite directions: safety_check should
have been included, grader_awareness should have been excluded). The
paper now uses the validation-grounded set and explicitly walks
through this refinement in §4.4-§4.5.

For future judges: don't ship a strict-rubric definition without a
validation pass against humans. Intuitions about which subtypes are
"clean" do not survive empirical testing.
