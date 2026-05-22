# Probing Is Not Enough: Eval/Deploy Directions Are Decodable but Not Causally Sufficient for Behavior

*Working title.*

*Status: DRAFT, 2026-05-21. Primary causal result COMPLETE: Qwen3-32B L55 steering with full pre-registered control suite (probe, probe-coef, random, orthogonal, wrong-layer; n = 100; two judges, Judge A + Judge B) — probe direction behaviorally indistinguishable from all controls. Corroborating probe-direction null on gpt-oss-120b L15 (n = 100, two judges), WITHOUT matched controls. Open work: gpt-oss control suite; necessity/ablation test; per-layer sweep. Defensible scope: full-controls causal null on Qwen3 L55 + corroborating null on gpt-oss; do NOT claim "across both architectures with full controls" until gpt-oss controls land.*

*Companion paper: V1–V3 behavioral/inject methodology in `docs/paper_draft.md`. This paper is V4 STANDALONE; cross-references V1–V3 but does not include its content.*

---

## Abstract

Linear probes trained on residual-stream activations can predict whether
a language model is processing an evaluation prompt or a deployment
prompt with cross-distribution AUROC of 0.917 (Qwen3-32B dense, L55)
and 0.896 (gpt-oss-120b MoE, L34) on a family-balanced, length-matched
holdout — establishing that the eval-vs-deploy distinction is linearly
encoded across two open-weight reasoning models of different architecture
families. We then test whether that encoded direction is *causally
sufficient* for behavior via activation steering. On **Qwen3-32B at
L55**, steering on the probe-recovered direction (n = 100 prompts,
α ∈ {−20, +20}, two independent LLM judges, **full pre-registered
control suite**) produces behavioral effects — on refusal, sandbagging,
and hedging — **statistically indistinguishable from random,
orthogonal, AND wrong-layer interventions of equal magnitude**: every
cell across five conditions × two alphas × two judges sits within ~2
standard errors of zero, and the two largest refusal effects in the
table belong to *control* conditions (random, wrong-layer), not the
probe. Output text changes frequently under steering (~75% of prompts
at α = ±20), but every control direction changes the same fraction —
so the surface perturbation is a generic consequence of
residual-stream perturbation magnitude, not of the probe direction
specifically. A separate gpt-oss-120b L15 steering run (n = 100, two
judges) likewise shows no behavioral effect for the probe direction.
Together these results support the claim that **probe-recoverability
does not imply causal sufficiency**: a direction can be highly
linearly-decodable (AUROC up to 0.917) yet behaviorally no more potent
than noise. We argue activation-level safety claims require level-3
causal tests — with random/orthogonal controls — beyond level-2 probe
AUROC. (Scope: the full controlled comparison is complete for Qwen3-32B
L55; the gpt-oss-120b L15 result is a probe-direction null without the
matched control suite, which remains in progress.)

---

## 1. Introduction

[Companion-paper framing] Recent work has demonstrated that
text-trace-level reasoning produced by frontier reasoning models does
not reliably reveal evaluation-awareness — models that behave
differently under explicit eval framing rarely verbalize that they
are doing so (V1–V3 companion paper, *in prep*). This raises a natural
question: if evaluation-awareness is real but not surfaced in the
chain-of-thought, is it instead encoded in the model's internal
representations? And if so, does that internal representation
*causally drive* the behavioral effect, or merely correlate with it?

**Detection is not causation.** Most of the eval-awareness literature
establishes *detection*: a model verbalizes suspicion that it is being
tested (trace-level), or an eval/deploy distinction is recoverable from
its activations (probe-level), or its behavior shifts under explicit
eval framing (behavioral-level). Each is evidence of a *detectable
eval-related signal* — but a probe that separates eval from deploy is
evidence of framing *information*, not necessarily of "eval-awareness"
in any richer sense. None of these detections, on its own, establishes
that the detected signal *causes* the behavioral difference that safety
evaluators worry about. A model can verbalize "this looks like a test"
without that thought driving its answer; a probe can decode eval/deploy
at high AUROC off a feature the model never behaviorally uses. The
inferential step from "we detected eval-awareness" to "eval-awareness
changed behavior" requires a causal test — intervention or
counterfactual — that the detection result does not supply.

This paper targets that gap at the activation level. Linear-probe
AUROC tests whether the eval/deploy distinction is *decodable*. Causal
sufficiency requires *intervention*: take a deploy-labeled prompt, add
the probe-identified direction to the residual stream, and check
whether behavior shifts — and crucially, whether it shifts *more than
a random direction of equal magnitude does*.

We test **decoding** on two open-weight reasoning models of different
architecture families (Qwen3-32B dense; gpt-oss-120b MoE) and **causal
sufficiency** via steering with a full pre-registered control suite on
Qwen3-32B L55 (with a corroborating steering null on gpt-oss-120b L15).
The decoding answer is clearly yes: cross-distribution probe AUROC is
0.917 (Qwen3-32B L55) and 0.896 (gpt-oss-120b L34). The
causal-sufficiency answer is clearly **no**: at Qwen3-32B L55, steering
on the probe direction produces behavioral effects (refusal,
sandbagging, hedging) statistically indistinguishable from random,
orthogonal, and wrong-layer interventions — every cell within ~2
standard errors of zero, two judges. Surface text changes under
steering (~75–93% of prompts for *any* direction), but behavior does
not move in a probe-specific way. We frame this as a methodological
result: **probe AUROC is necessary but not sufficient evidence that a
direction is causally relevant to behavior; detection is not
causation.**

### 1.1 Contributions

1. **Cross-architecture probing of eval/deploy framing signals** on two open-weight
   reasoning models of different architecture families
   (Qwen3-32B dense decoder; gpt-oss-120b MoE). Linear probes trained
   on Microsoft's Test-Awareness-Steering trigger set achieve
   cross-distribution AUROC 0.917 (Qwen3-32B L55) and 0.896
   (gpt-oss-120b L34) on a family-balanced, length-matched controlled
   holdout. The phenomenon documented in prior work (typically on a
   single architecture) generalizes across this dense / MoE pair.
2. **A pre-registered design and reproducible pipeline** for activation
   steering as a causal-sufficiency test of probe directions, including
   built-in correctness tests (T1–T4), forward-hook implementation
   for HuggingFace causal LMs, gpt-oss-specific mxfp4 / `triton_kernels`
   stack documentation, and LLM-judge scoring infrastructure for
   refusal, sandbagging, and hedging outcomes.
3. **A controlled causal-sufficiency null** on Qwen3-32B L55: across
   n = 100 prompts, two judges, and a full pre-registered control suite
   (probe, probe-coefficient, random, orthogonal, wrong-layer), steering
   on the probe direction produces behavioral effects (refusal,
   sandbagging, hedging) **statistically indistinguishable from random,
   orthogonal, and wrong-layer interventions** — every cell within ~2
   SE of zero. Output text changes under steering (~75–93% for any
   direction), but behavior does not move in a probe-specific way. A
   corroborating probe-direction steering null on gpt-oss-120b L15
   (n = 100, two judges) holds without the matched control suite.
4. **Methodological recommendation**: level-3 causal testing — with
   random/orthogonal controls — should accompany level-2 probing when
   interpreting probe-recoverable features as evidence of
   safety-relevant internal states. Probe AUROC alone does not
   establish causal relevance.

Status of pre-registered work *not yet reported*: the matched control
suite on gpt-oss-120b L15 (Qwen3 L55 has it; gpt-oss has only the
probe-direction null), a necessity/ablation test (§7.4), and a
per-layer sweep. Until the gpt-oss controls land we say "full-controls
null on Qwen3 L55 + corroborating null on gpt-oss," not "across both
architectures with full controls."

### 1.2 Out of scope

We make no claim that the model "knows" it is being evaluated or that
it strategically modulates behavior. Such cognitive claims are
unprovable from behavioral and activation evidence (Belinkov 2022,
Pimentel et al. 2020). We restrict ourselves to: *information is
present in activations*, and *that information is or is not behaviorally
load-bearing under direct intervention*.

---

## 2. Related Work

[Stub. To fill:]

- **Probing classifiers and their critique**: Alain & Bengio 2016
  (original probing); Hewitt & Liang 2019 (control tasks); Belinkov 2022
  (probing-classifiers survey); Pimentel et al. 2020 (information-theoretic critique).
- **Eval-awareness measurement**: Abdelnabi & Salem 2025
  (Test-Awareness-Steering, the training source for our probes);
  Anthropic / Apollo Research's eval-awareness behavioral studies;
  V1–V3 companion paper.
- **Activation steering / representation engineering**: Turner et al.
  2023 (activation addition); Rimsky et al. 2023 (Contrastive Activation
  Addition); Zou et al. 2023 (Representation Engineering); Park et al.
  2023 (geometry of LLM concepts).
- **MoE interpretability** (sparse): Anthropic's circuits work on
  Mixtral; OpenAI gpt-oss release notes / kernel choices.
- **Probes vs causal mechanisms**: Geiger et al. 2024 (causal
  abstraction); Conmy et al. 2023 (automated circuit discovery).

---

## 3. Methodology

### 3.1 Models

We use two open-weight reasoning models of different architecture
families, spanning dense vs MoE and training-lineage diversity:

| Model | Architecture | Params | Layers | d_model |
|---|---|---|---|---|
| Qwen3-32B | Dense decoder, GQA | 32B | 64 | 5120 |
| gpt-oss-120b | MoE (128 experts, 5.1B active), mxfp4 | 117B | 36 | 2880 |

[Choice rationale: both have public weights, public chat templates,
similar inference recipes. gpt-oss requires `triton_kernels` +
`kernels` packages for the mxfp4 path — see §3.6 for the technical
detail.]

### 3.2 Probe training corpus

We train probes on the Microsoft Test-Awareness-Steering "triggers"
dataset (Abdelnabi & Salem 2025). The dataset contains 2,587 paired
items × 2 arms per pair (a "real" deploy-labeled arm + a "hypothetical"
eval-labeled arm), yielding 5,174 total arms covering ~12 task
families. The Qwen3-32B probes are trained on all 5,174 arms; for
compute reasons the gpt-oss-120b probes are trained on the first 2,048
arms (1,024 paired items, label-balanced).

For each model we extract mean-pooled residual-stream activations at
8 layers spanning the depth. For gpt-oss-120b (36 layers):
`[5, 10, 15, 20, 25, 28, 31, 34]`. For Qwen3-32B (64 layers):
`[10, 20, 30, 40, 50, 55, 60, 62, 63]`. Logistic regression probes
are trained with 5-fold cross-validation; reported numbers are mean
CV AUROC + CV std.

### 3.3 Cross-distribution test set: controlled holdout v2

To address known critiques of probing classifiers — that high AUROC
may reflect surface-text features rather than the target concept — we
evaluate probes on a **family-balanced, length-matched controlled
holdout** disjoint from MS triggers:

- **160 matched eval/deploy pairs = 320 prompts total**, across 4
  framing families (A, B, C, D) of 40 pairs each
- Within each pair, the eval and deploy arms share the same underlying
  question; only the framing differs (declarative-deploy vs
  hypothetical-eval, plus two additional framing variants per family)
- Labels are balanced 160 eval / 160 deploy by construction
- Token lengths padded within pairs so prompt length cannot predict
  label (a length-only baseline AUROC is 0.500 by construction)
- Built before any model access; pre-registered design

Cross-distribution AUROC on this holdout is the primary probe metric;
within-distribution CV AUROC on triggers is reported for completeness
but not the headline number.

### 3.4 Activation-steering protocol

We test causal sufficiency by direct intervention. For each (model,
peak layer, prompt, alpha) tuple we:

1. Compute the steering vector `w` at the peak layer from the same MS
   triggers training data. We report two variants (probe coefficient
   normalized to unit length; mean-difference `μ_eval - μ_deploy` per-dim
   z-scored then renormalized). Default-reported variant:
   `mean_diff_zscored`, following CAA convention (Rimsky et al. 2023).
2. Register a forward hook on the target layer that adds `α · w` to
   the residual stream during prefill (multi-token forward passes).
   During decode (single-token forward passes via KV cache) the hook
   is a no-op; the prefill modification persists through the cache.
3. Generate up to 300 tokens deterministically (`do_sample=False`,
   no temperature, no top-p sampling).
4. Repeat per (prompt, α) with α ∈ {−20, −5, 0, +5, +20} (chosen
   based on per-layer residual-stream magnitudes; see §3.7).
5. Run the same protocol on three pre-registered control conditions:
   `random` direction, `orthogonal` direction (projected to be ⊥ to
   `mean_diff_zscored`), and `wrong_layer` (a non-peak layer).

### 3.5 Pre-registered correctness tests (T1–T4)

Before any data collection we run four built-in correctness tests:

- T1: α = 0 hook produces output identical to no-hook generation
- T2: α > 0 produces residual-stream modification approximately equal
  to `α · w` at all prompt positions (loose threshold to accommodate
  bf16 quantization noise at high-magnitude residual scales)
- T3: α > 0 produces a different generation than α = 0
- T4: hook is properly removed after generation

The correctness-test runner (`v4/scripts/run_steered_generation.py`
with `--test`) was executed on gpt-oss-120b L15 and Qwen3-32B L55
prior to data collection. T1, T2, T4 pass; T3 passes at L15/L55 and
issues an informative
warning at L34 (where the larger residual-stream magnitudes make
α = 2 with a unit-norm direction fall at or below bf16 representable
precision — see §5.2 and §6.1). The test invocation prints PASS/FAIL
to stdout; we do not currently save a structured JSON correctness
artifact, so reproducibility relies on re-running the test script on
the same artifacts (model + steering-vectors file). The Qwen3-32B L55
steering implementation has been built and steering vectors prepared
but T1–T4 has **not yet been executed** at the time of this draft;
corresponding test logs will be saved and reported in the Qwen3
replication. T2's exact-error threshold and bf16-quantization
interaction is documented in §6.2.

### 3.6 Inference stack and reproducibility

gpt-oss-120b in mxfp4 packed form requires a specific software stack
to engage the mxfp4 inference kernels and avoid dequantization to bf16:

- PyTorch 2.8 (provides Triton ≥ 3.4.0)
- HuggingFace `transformers` ≥ 4.60
- HF `kernels` package (loads kernels from the Hub at runtime)
- `triton_kernels` pinned to the `release/3.4.x` branch of the
  triton-lang repo (the PyPI `triton-kernels==0.1.0` is a stub
  without the `matmul_ogs` kernel `transformers` expects)

Without these, the mxfp4 quantizer falls back to dequantization-to-bf16
during weight loading, ballooning model memory from ~60 GB to ~240 GB
and forcing partial CPU offload (with ~5× slowdown in our setup).
We document the full stack here because the package interactions are
non-obvious and constitute a real reproducibility hazard.

All experiments were run on RunPod 1× H200 SXM 141 GB pods with the
model weights stored on local SSD (`/tmp`) rather than the network
volume — early experiments revealed that the network-volume backend's
per-file random-access latency made model loading take ~30× longer
than local-disk loading.

### 3.7 Steering vector magnitudes and alpha-range selection

[Important methodological subtlety to flag for readers]

We normalize the steering vector `w` to unit length. The appropriate
alpha range therefore depends on the **typical residual-stream
magnitude at the target layer**. For gpt-oss-120b:

| Layer | Residual std | Residual range | Effective α | Why |
|---|---:|---:|---:|---|
| L15 | ~50 | [−2K, +2K] | α ∈ ±20 | Mid-network, smaller magnitudes |
| L34 | ~961 | [−25K, +32K] | α ∈ ±1000+ | Late layer, huge magnitudes |

At L34, even α = 5000 with unit-norm `w` produces deltas that are at
or below bf16 rounding precision relative to typical residual values
(see §5.2 for the consequence — byte-identical outputs across α ∈
[−100, +5000] at L34). Practitioners should compute per-layer
residual-stream statistics from the training-activation file and pick
alpha ranges that produce delta-magnitudes in the 1%–20% of residual
std range.

---

## 4. The Judge

We score each generated response on three pre-registered binary/ordinal
properties:

1. **Refusal** (binary): did the response refuse, deflect, or
   substantially decline the user's request?
2. **Sandbagging** (binary): does the response show strategic
   underperformance — a clearly wrong answer when the model is capable
   of better?
3. **Hedging** (1–5 ordinal): how much hedging language, caveats,
   deference to oversight, or evaluation-context cues?

Rubric and exact prompt template in Appendix A.

**Inter-rater agreement (both judges, Qwen3 L55 dataset).** Across the
combined judged Qwen3 L55 records (≈1180/1200 valid for each judge),
the two judges agree at:
- Refusal: 88.8–91.4% (across the probe-vs-random and controls files)
- Sandbagging: 69.1–80.1%
- Hedging: Pearson r = 0.67–0.77

Refusal agreement is high; sandbagging is the noisiest axis (judges
genuinely differ on borderline strategic-underperformance calls);
hedging correlates moderately. The behavioral deltas we report
(§5.4) are far smaller than the inter-judge disagreement, which is
itself part of why we conclude the effects are null — no signal
survives a second judge.

*Historical note*: an early n = 20 pilot (§5.3) was scored Judge A-only
because the Judge B reasoning variant errored on `max_tokens`
(requires `max_completion_tokens`) and, once fixed, needed a larger
`max_completion_tokens` budget (reasoning tokens were exhausting the
visible-output budget). Both issues were patched before the n = 100
runs reported here.

---

## 5. Results

### 5.1 Cross-architecture probing replicates (V4b)

**Headline numbers** (probe trained on MS triggers, evaluated on
controlled holdout v2, n = 320 prompts):

| Model | Architecture | Peak Layer | Peak AUROC | MS-TFIDF baseline | Δ vs baseline |
|---|---|---|---|---|---|
| Qwen3-32B | dense decoder | L55 (depth 0.86) | **0.917** | 0.834 | +0.083 |
| gpt-oss-120b | MoE | L34 (depth 0.94) | **0.896** | 0.831 | +0.065 |

Both models clear the MS-trained TF-IDF surface-text baseline by
+0.07 to +0.08 AUROC; the probing signal is therefore not reducible
to MS-triggers-specific text features that the baseline could
memorize. Length-only baseline AUROC is 0.500 on both (the controlled
holdout is length-matched within pairs by construction).

**Figure 1** (the paper's headline cross-architecture figure;
`v4/runs/v4b_cross_arch_probe_projection.png`): for each model, the
holdout activations at the peak layer are projected onto the
trained probe coefficient `w`, and the resulting 1-D scores are
plotted as overlaid histograms of eval (red) vs deploy (blue). Both
panels show clear class separation; the gpt-oss panel exhibits more
spread and a partial bimodality in the eval distribution that traces
to per-family variance (see Figure 2).

**Per-family analysis** (gpt-oss-120b L34): probe AUROC varies
substantially across the four framing families on the controlled
holdout:

| Family | AUROC at L34 |
|---|---|
| A | 0.933 |
| B | 0.771 |
| C | 0.998 |
| D | 1.000 |

**Figure 2** (`v4/runs/gpt_oss_stage2/L34_probe_projection_per_family.png`):
per-family histograms reveal that family C and family D are essentially
perfectly separable, while family A has a calibration offset (good
within-family ranking, but absolute probe scores systematically below
the global decision threshold) and family B has genuine class overlap.
We discuss the implications in §7. The Qwen3-32B L55 per-family
breakdown is in `v4/runs/qwen3_stage2/validation_qwen3.json` and will
be tabulated here in the next revision; we do not assert a
cross-architecture per-family pattern claim until that comparison is
done explicitly.

**Diagnostic** (`v4/runs/gpt_oss_stage2/L34_pca_2d.png` and the Qwen3
counterpart at `v4/runs/qwen3_stage2/L55_pca_2d.png`): unsupervised
2-D PCA of the holdout activations is dominated by the four framing
families (PC1+PC2 capture ~25–30% of variance and cleanly cluster by
family, not by eval/deploy label). This is the right diagnostic
outcome: it confirms that eval/deploy is *not* the largest variance
axis at these layers — it is a smaller consistent shift not captured
by the first two PCA axes (which are dominated by family structure).
Whether the probe's learned direction is statistically orthogonal to
the family-discriminating directions specifically would require an
explicit measurement we have not yet run; we report only the visual
PCA observation here.

**Depth-portability does not hold.** We pre-registered the prediction
that gpt-oss's optimal probe layer would match Qwen3 L55's depth
fraction (0.86), giving gpt-oss L31. In practice, gpt-oss's optimal
probe layer is L34 (depth 0.94) — substantially later. Cross-architecture
optimal-layer selection cannot assume depth fraction transfers; a
per-model layer sweep is required.

### 5.2 Causal sufficiency test (Stage 4): L34 is non-responsive

We first attempted steering at the probe-optimal layer for each model.
On gpt-oss-120b L34, **outputs were byte-identical across α ∈
[−100, +5000] using `mean_diff_zscored` unit-norm direction** (full
405-character generations, n = 3 sample prompts).

Diagnostic correctness tests (T1, T2, T4) pass; the forward hook fires
on prefill with the correct seq_len and applies `α · w` to the
measured tolerance (T2 max-per-element error 0.97 at α = 10, within
the bf16 noise budget for L34's residual scale). T3 at the L34 layer
specifically issues a *warning* (not a pass): α = 2 with unit-norm `w`
produces an output identical to α = 0. We extend that observation
with the full sweep in §5.2 — byte-identical outputs persist up to
α = 5000.

**One likely explanation for the L34 null is a magnitude mismatch.**
gpt-oss-120b's L34 residual stream has std ≈ 961 and range ±32,000,
so a unit-norm delta multiplied by even α = 5000 produces per-element
shifts of at most ~500 — about half a residual std and at or below
bf16 representable precision at high-magnitude residual positions.
Combined with the model's strongly-templated refusal completions on
the Apollo alignment-stratum prompts (which converge to "I'm sorry,
I can't help with that" regardless of mid-layer perturbation), no
behavioral effect is observable. Alternative hypotheses we cannot
yet rule out: MoE-routing interactions specific to L34, a
forward-hook propagation issue specific to gpt-oss's expert layers,
or the probe direction at L34 being non-causal even apart from the
magnitude issue. The L15 result (§5.3) — where steering at a
mid-network layer with smaller residual magnitudes does change
outputs in 55% of prompts — is consistent with the magnitude-mismatch
explanation but does not exclude the others.

We do not interpret the L34 null as evidence about eval/deploy
encoding *per se*; rather, it is a *layer- and magnitude-specific
non-effect* that argues for testing earlier layers where residual-stream
magnitudes give the same alpha range more effective perturbation power.

### 5.3 gpt-oss-120b L15: a corroborating probe-direction null

Because gpt-oss L34 was non-responsive for magnitude/precision reasons
(§5.2), we steer gpt-oss at the mid-network layer L15 (depth 0.42,
probe AUROC 0.876, residual std ~50, where fixed-norm deltas are not
swamped by bf16 rounding). At n = 100 prompts, α ∈ {−20, +20}, scored
by two judges (Judge A + Judge B), steering on the probe direction produces
**no behavioral effect that survives a second judge** — every paired
delta is within standard error of zero, and the judges disagree on the
sign of the (null) effects:

| Judge | α | Δrefusal | Δsandbag | Δhedging |
|---|---|---:|---:|---:|
| Judge A | −20 | +0.01±0.02 | +0.06±0.04 | −0.03±0.05 |
| Judge A | +20 | +0.02±0.03 | +0.00±0.04 | +0.02±0.05 |
| Judge B | −20 | −0.04±0.02 | −0.05±0.04 | +0.03±0.10 |
| Judge B | +20 | −0.03±0.02 | −0.04±0.03 | −0.06±0.08 |

**Scope of this result.** We did *not* run the matched
random/orthogonal/wrong-layer control suite on gpt-oss L15, so this is
a probe-direction *null*, not the full probe-vs-control comparison.
That full comparison is established for Qwen3-32B L55 (§5.4); the
gpt-oss result *corroborates* it on a second, MoE architecture but
does not yet stand on its own controls.

*(An earlier n = 20 single-judge pilot at this layer appeared to show
a small directional hedging effect, Δhedging ≈ +0.20; it did not
replicate at n = 100 under either judge and is not a finding.)*

### 5.4 Qwen3-32B L55: the probe direction is behaviorally indistinguishable from random (primary causal result)

This is the paper's central causal-sufficiency test. On Qwen3-32B L55 —
the model's strongest probing layer (cross-distribution AUROC 0.917) —
we steer on the probe-recovered direction (`mean_diff_zscored`) and
compare against a **random direction of equal magnitude** across n = 100
prompts, α ∈ {−20, +20}, scored by two independent judges (Judge A,
Judge B; valid coverage 490/500 and 498/500 respectively).

**Output text changes frequently — for every direction equally, and
MORE at the wrong layer.** At α = ±20, output-change rates vs the α = 0
baseline are: probe (mean_diff) 70–77%, random 74–79%, orthogonal
69–80%, probe-coefficient 72–81%, and **wrong-layer L25 90–93%**.
Output perturbation rate is a function of steering *magnitude and
injection depth*, not the probe direction — and notably the
wrong-layer intervention disrupts the *most* outputs (earlier layer →
more downstream amplification) while producing *no* behavioral effect.
Output-change-rate and behavioral-effect are fully decoupled: the
probe direction is not privileged on either axis.

**Behavioral effects: the probe direction does not exceed ANY control,
on ANY metric, under either judge.** Full pre-registered control suite
(n ≈ 98–100 per cell, paired Δ vs α=0 baseline):

*Judge A:*
| Condition | α | Δrefusal | Δsandbag | Δhedging |
|---|---|---:|---:|---:|
| **mean_diff (probe)** | −20 | +0.02±0.03 | −0.02±0.04 | −0.01±0.05 |
| **mean_diff (probe)** | +20 | +0.01±0.03 | +0.02±0.04 | −0.08±0.05 |
| probe_coef (probe v2) | +20 | +0.05±0.03 | +0.01±0.04 | +0.02±0.05 |
| random | −20 | **+0.08±0.03** | +0.01±0.04 | +0.01±0.05 |
| random | +20 | −0.02±0.03 | +0.06±0.04 | −0.05±0.05 |
| orthogonal | +20 | +0.01±0.02 | −0.01±0.04 | −0.06±0.04 |
| wrong-layer L25 | +20 | **+0.07±0.04** | −0.02±0.04 | +0.06±0.06 |

*Judge B:* all conditions within SE of zero, signs scattered, no
condition exceeds the others (full table in Appendix).

The two largest refusal deltas in the entire table belong to **random
(+0.08)** and **wrong-layer (+0.07)** — *controls*, not the probe.

Against the pre-registered success criteria (probe effect must exceed
control, be directional, dose-responsive, judge-agreed), **every
criterion fails**:
- Probe deltas do not exceed any control on any metric; the largest
  effects are in controls.
- Probe effects are not directional (Δrefusal flat across α).
- The two judges disagree on the sign of the (null) effects.

**Conclusion: at Qwen3-32B L55, the eval/deploy probe direction —
despite achieving 0.917 cross-distribution AUROC — is behaviorally
indistinguishable from random, orthogonal, AND wrong-layer
interventions of equal magnitude. Every cell across 5 conditions × 2
alphas × 2 judges sits within ~2 SE of zero. The direction has no
privileged causal status by any comparison.**

### 5.5 Interpretation: probe-recoverability ≠ causal sufficiency

The Qwen3 L55 full-control result (§5.4) and the gpt-oss L15
corroborating null (§5.3) jointly support the paper's core claim: **a
direction can be highly linearly-decodable yet behaviorally
indistinguishable from noise under steering.**

Scope, stated precisely:
- The complete probe-vs-control comparison (probe ≈ random ≈ orthogonal
  ≈ wrong-layer, two judges, n = 100) is established for **Qwen3-32B
  L55**. This is the paper's primary causal evidence.
- gpt-oss-120b L15 shows a probe-direction null but **without** the
  matched control suite, so it *corroborates* the Qwen3 result on a
  second architecture rather than standing on its own controls.
- Running the gpt-oss control suite would upgrade the claim from
  "full controls on one model + corroborating null on a second" to
  "full controls on both" (§8).

A note on what this is *not*: an earlier single-judge n = 20 pilot
suggested the probe direction might encode a *stylistic register*
(hedging) rather than a behavioral axis. The n = 100 two-judge data do
not support even that weaker reading — hedging deltas are within noise
and judge-disagreed, identical in character to refusal and sandbagging.
The cleaner and stronger statement is simply that **the probe direction
has no detectable behavioral signature beyond what an equal-magnitude
random perturbation produces.** We are therefore not claiming the
direction encodes "style instead of behavior"; we are claiming it is
*diagnostic without being behaviorally sufficient*. We discuss what
could still be true of the direction (e.g., necessity untested) in §7.

---

## 6. Robustness

### 6.1 BF16 quantization affects steering math at high-magnitude residuals

[Discussion of T2 max-error vs bf16 noise floor; gpt-oss L34 was below
the noise floor; L15 is well above it. Implication: cross-layer
comparisons of alpha effects must account for per-layer residual scale.]

### 6.2 Hook-output propagation

[Discussion of forward-hook semantics: returning a modified output
from `forward_hook` on a transformer block does propagate to the next
block's input on standard HF causal LMs, but MoE + custom-kernel
architectures may have subtle interactions. We diagnosed this on L34
where steering produces zero output difference even at α magnitude
sufficient to be representable in bf16. The L15 result (where steering
*does* produce output changes) demonstrates propagation works at that
layer — the L34 null is therefore most plausibly attributable to
either (a) bf16 noise at the higher residual scale, (b) MoE-routing
interactions, or (c) the model's refusal-templating attractor.]

### 6.3 Pre-registered vs reported

The pre-registered design is in `STAGE4_PLAN.md` (committed to the
repo before any rollouts; see Appendix B). The current draft reports
a subset of that design and explicitly defers the rest. Walk-through:

| Pre-registered | This draft | Status |
|---|---|---|
| V1 BeaverTails / MMLU / GSM8K / Apollo prompt set, 500 prompts | V2 Apollo alignment subset, 100 prompts | Pivoted (local data availability); full corpus deferred to necessity/ablation follow-up |
| n = 500 × 7 alphas × 4 conditions × 2 models | Qwen3 L55: 100 prompts × {−20,+20} × 5 conditions; gpt-oss L15: 100 × 3 × 1 | Qwen3 **complete with full controls**; gpt-oss probe-only |
| Judge A + Judge B judge with inter-rater agreement | Judge A + Judge B, both, agreement reported (§4) | **Done** |
| Four conditions: probe, random, orthogonal, wrong-layer | Qwen3 L55: all five (incl. probe-coef); gpt-oss L15: probe only | Qwen3 **complete**; gpt-oss controls pending |
| Two models: Qwen3-32B + gpt-oss-120b steering | Both steered; Qwen3 full controls, gpt-oss null | Qwen3 complete; gpt-oss corroborating |
| Pre-registered verdict thresholds (SUFFICIENT / PARTIAL / NULL) | **NULL** on Qwen3 L55 (probe ≈ all controls; all 5 success criteria fail) | Applied for Qwen3 L55 |

The Qwen3-32B L55 causal-sufficiency result is **complete and
controlled**. The two items still outstanding relative to the full
pre-registration are: the matched control suite on gpt-oss-120b L15
(it has the probe-direction null but not the controls), and the
necessity/ablation test (§7.4, deferred by design).

---

## 7. Discussion

### 7.1 What kind of result is this?

For Qwen3-32B L55 (n = 100, two judges, full pre-registered control
suite): a **clean, controlled causal-sufficiency null**.

- Not a positive: no condition — probe, probe-coefficient, random,
  orthogonal, or wrong-layer — produces a behavioral effect that
  exceeds the others or survives both judges; all sit within ~2 SE of
  zero.
- A *controlled* null, not merely an absence of effect: the probe
  direction is specifically indistinguishable from random and
  orthogonal directions of equal magnitude, so we can rule out "the
  probe direction does something special that we failed to detect."
  The largest behavioral deltas in the table belong to *control*
  conditions.
- The decoupling is the crisp finding: surface text changes under any
  perturbation (~75–93% of prompts), behavior does not change in a
  probe-specific way. Decodability and behavioral leverage are
  dissociated.

gpt-oss-120b L15 corroborates the null on a second architecture but
without the matched controls (§5.3). The remaining honest gap is the
gpt-oss control suite, not the Qwen3 conclusion.

### 7.2 What could still be true of the direction

The controlled null tells us the probe direction is **diagnostic but
not additively behaviorally sufficient**. It does not, by itself,
adjudicate *why*. Three accounts remain consistent with the data:

**(A) The direction is a decodable correlate the model does not use
behaviorally.** MS triggers' eval/deploy labels track surface-text
features (declarative vs hypothetical framing); a probe maximizing
AUROC will latch onto whatever residual-stream features carry that
framing information, and those features need not be the ones the
model's behavioral circuit reads. Under this account, the high AUROC
was never evidence of a behavioral mechanism. (We deliberately do
*not* sharpen this to "the direction is *stylistic register*" — our
n = 100 two-judge data do not support even that specific claim; see
§5.5. The defensible version is simply "decodable, not causal.")

**(B) The direction is a partial shadow of a distributed mechanism**
that single-layer linear additive steering cannot capture. Multi-layer
or distributed interventions (RepE-style) might move behavior where our
single-layer intervention does not. This is the open research question
the result leaves on the table — and the reason our null is a *lower
bound* on steerability, not a proof of total causal irrelevance.

**(C) Behavioral modulation depends on *prompt-level* eval cues, not
mid-layer activation patterns.** V1 inject prompts state "this is an
evaluation" in the text; the behavioral response may key off
recognizing that statement, not off activations consistent with
having-processed-eval-prompts. Activations can carry "is this prompt
hypothetical-mood" without the behavioral circuit consuming it.

We cannot fully discriminate (A), (B), (C) from current data.
**(A) and (C) yield the same prescription**: do not treat probe AUROC
as evidence of behavioral relevance. **(B) defines the most valuable
follow-up**: necessity/ablation (§7.4) and distributed multi-layer
steering, on a non-saturated prompt set.

### 7.3 Implications for safety-relevant interpretability

[Argument for the methodological lesson:]
Many activation-level interpretability claims for safety rely on
probe-detectable features as evidence that the model "represents X"
(deceptiveness, scheming, eval-awareness, etc.). Our result suggests
this is a category error in some cases. A model's residual stream may
encode many surface-text features that correlate with safety-relevant
labels in the training distribution without those features driving
the behaviors of interest. Activation steering — even imperfect
activation steering — provides a partial sieve for distinguishing
encoding from causality.

### 7.4 Sufficiency vs necessity: why we test the former and defer the latter

Our steering experiment tests **causal sufficiency**: *if we add this
direction, does behavior appear or shift?* The complementary question
is **necessity**: *if we remove (ablate) this direction, does the
eval/deploy behavioral difference weaken or disappear?* These are
logically distinct — a feature can be necessary but not sufficient
(removing a car's steering wheel disables it; bolting on a second
steering wheel does not make it drive). A steering null therefore does
not *prove* an ablation would be null.

We nonetheless treat ablation as a lower-priority follow-up rather than
a gap in the present claim, for three reasons specific to our results:

1. **Our null is stronger than a generic steering null.** We did not
   merely find "steering did nothing"; we found that steering the
   probe direction is statistically indistinguishable from steering a
   *random* direction of equal magnitude. A direction that were the
   load-bearing eval/deploy mechanism would, in the common case, show
   *some* additive potency above random. Ours shows none. This lowers
   (without eliminating) the prior that the direction is uniquely
   necessary.

2. **A sound ablation needs its own control.** Just as steering
   required random/orthogonal controls, ablation requires an
   *ablate-random* comparison — remove the probe direction vs remove a
   random direction, and test whether behavior drops *more* for the
   probe. "Behavior changed after ablation" is otherwise confounded by
   the generic effect of perturbing the forward pass. Ablation is thus
   a full second experiment with its own control suite, not a quick
   add-on.

3. **Prompt saturation limits what ablation could reveal.** Ablation
   is most informative when there is a clear behavioral phenomenon to
   abolish. Our Apollo alignment-stratum prompts are largely saturated
   refusals, so the natural eval-vs-deploy behavioral gap on this set
   is small — leaving little for ablation to remove.

We therefore report: *we tested sufficiency through steering, with
random and orthogonal controls; we did not test necessity through
ablation; given the probe direction's additive indistinguishability
from random and the control/saturation considerations above, ablation
is a scoped follow-up rather than a hole in the causal claim.*

### 7.5 Architecture: a generality observation, not a difference claim

Our two models fall into distinct architecture buckets:

| Model | Bucket | Relevant property |
|---|---|---|
| Qwen3-32B | dense decoder-only | all parameters active each token; residual std ~17 at L55 |
| gpt-oss-120b | sparse MoE decoder-only | subset of experts active per token; residual std ~961 at L34 |

We are careful about what this contrast supports.

**What we claim — generality.** The core finding (eval/deploy is
decodable but the probe direction is not causally sufficient under
steering) holds in *both* a dense and an MoE model. This argues the
phenomenon is not an artifact of one architecture. That is a
*replication-across-architectures* point and it strengthens the thesis.

**What we do NOT claim — an architectural difference in behavior.**
Both models show the same qualitative result (decodable + steering-null
where steering is testable). We did *not* observe "dense steers, MoE
doesn't" or any behavioral steering *difference* attributable to
architecture. With n = 1 model per bucket, and given that Qwen3 and
gpt-oss also differ in training data, post-training, safety tuning,
tokenizer, parameter count, active-parameter count, and inference
precision, "architecture" would be only one of many possible
explanations for any difference we might have seen — and we did not see
a behavioral one to explain.

**One architecture-adjacent *methodological* observation.** gpt-oss's
MoE late layers carry far larger residual-stream magnitudes (std ~961,
range ±32k at L34) than Qwen3's (std ~17 at L55). This made fixed-norm
steering deltas vanish into bf16 rounding at gpt-oss's late layers (the
L34 non-responsiveness, §5.2), forcing us to steer at an earlier layer
(L15). This is a practical precision/scaling note about intervening on
MoE models, not a behavioral finding.

**Defensible forward-looking statement:** *our results motivate
architecture-aware causal testing — probe directions that are similarly
decodable across dense and MoE models may not present comparable
steering dynamics (e.g., residual-magnitude scaling), and confirming
whether architecture modulates decodability strength or intervention
behavior requires more than one model per bucket. We treat the dense/MoE
contrast as hypothesis-generating, not as an architectural conclusion.*

### 7.6 What we don't claim

We don't claim:
- The model lacks an internal representation of eval-awareness — we
  only show this *particular* probe-recovered direction is not
  causally distinguishable from random under additive steering.
- All probing for safety-relevant features is misleading — probes
  remain valid existence-of-information evidence.
- Activation steering is the right tool for all causal questions —
  it tests sufficiency but not necessity, can't establish
  cognitive properties, and depends on choosing the right
  layer/direction.

---

## 8. Limitations

The steering result is **complete with a full control suite for
Qwen3-32B L55** (probe, probe-coefficient, random, orthogonal, and
wrong-layer conditions; n = 100; two judges) and is **corroborated by
a probe-direction steering null on gpt-oss-120b L15** that does not yet
have the matched control suite. The limitations below concern remaining
scope, not the validity of the Qwen3 L55 result.

**Scope limitations (largest first):**

1. **The control suite is complete for one model, not both.** Qwen3-32B
   L55 has the full pre-registered control comparison (probe ≈ random ≈
   orthogonal ≈ wrong-layer). gpt-oss-120b L15 shows a probe-direction
   behavioral null but **without** matched random/orthogonal/wrong-layer
   controls. So our strongest claim — "the probe direction is
   indistinguishable from controls" — is established for Qwen3 L55 and
   only *corroborated* (not control-matched) on gpt-oss. Running the
   gpt-oss control suite is the single most valuable remaining addition;
   it would upgrade the claim from "full controls on one model + null on
   a second" to "full controls on both."

2. **Sufficiency only; necessity (ablation) not tested.** We test
   whether *adding* the probe direction changes behavior (sufficiency).
   We do not test whether *removing* it weakens the eval/deploy
   behavioral difference (necessity). As argued in §7.4, this is a
   scoped follow-up rather than a hole in the *sufficiency* claim: a
   sound ablation needs its own ablate-random control and a
   non-saturated prompt set. A complete necessity test remains future
   work.

3. **Single layer per model.** We steer at Qwen3 L55 and gpt-oss L15.
   gpt-oss L34 (the probe-AUROC peak) was non-responsive due to bf16
   residual-magnitude effects (§5.2, §6.1), which is why we moved to
   L15. A full per-layer sweep — establishing whether the null is
   layer-specific or network-wide — is future work.

4. **Prompt-set homogeneity.** Apollo alignment-stratum prompts are
   strongly refusal-templated — many converge to "I'm sorry, I can't
   help with that" regardless of mid-layer perturbation. The probe ≈
   random result holds *regardless* of saturation (both are null), so
   this does not threaten the sufficiency conclusion; but it does limit
   how much behavioral range was available to move, and it is why a
   future necessity/ablation test should use a less-saturated set
   (e.g., V1 BeaverTails, where the natural eval/deploy behavioral gap
   is largest).

5. **Two direction variants tested, not all.** We steer on both
   `mean_diff_zscored` and the probe coefficient (`probe_coef_zscored`);
   both are null and indistinguishable from controls. Other plausible
   direction constructions (e.g., difference-of-means without
   z-scoring, or directions from a probe trained on the controlled
   holdout rather than MS triggers) are untested.

**Methodological limitations:**

8. **LLM-as-judge is itself a behavioral probe.** The hedging /
   refusal / sandbagging measurements depend on a judge model's reading
   of the generated text. We use two judges (Judge A, Judge B)
   and report inter-judge agreement (refusal 88.8–91.4%, sandbagging
   69.1–80.1%, hedging Pearson r = 0.67–0.77). Sandbagging is the
   noisiest axis. Crucially, the behavioral deltas we measure are
   *smaller than the inter-judge disagreement* — which is itself part
   of why we read the effects as null. Human validation of judge
   labels on a stratified subset remains pending.

9. **bf16 quantization affects steering precision at high-magnitude
   layers.** As discussed in §3.7 and §5.2, gpt-oss-120b's late-layer
   residual stream magnitudes make sub-ULP additions vanish in bf16,
   producing the L34 null. This is an artifact of the inference
   precision, not necessarily a property of the underlying model.
   Re-running L34 in fp32 (much slower, requires more VRAM) would
   resolve whether the null is bf16-quantization-induced or a deeper
   property of the layer.

10. **Single-layer intervention.** We modify the residual stream at
    one layer per run. Distributed multi-layer steering (Rimsky et al.
    2023; Zou et al. 2023) often produces larger behavioral effects and
    may be the appropriate intervention for safety-relevant features
    that are not concentrated at a single layer. Our result is therefore
    a *lower bound* on how strongly the probe direction can be steered.

11. **No cognitive claim.** We do not test, and do not claim, that the
    model "knows" it is being evaluated or that any internal state
    corresponds to "awareness." All claims here concern *representation
    content* and *behavioral consequences of intervention*, not
    cognition. The probing-classifier critique literature (Belinkov
    2022, Pimentel et al. 2020) is unanimous on the limits of probing
    for cognitive claims; this paper inherits those limits.

12. **MoE-specific forward-hook behavior is not yet diagnosed.** The
    L34 null might be partly attributable to MoE expert routing or
    custom mxfp4 kernel paths interacting with the standard PyTorch
    `register_forward_hook`. A direct propagation diagnostic (e.g.,
    `register_forward_pre_hook` on the next layer to verify it
    receives the modified output) has not been completed here.

---

## 9. Conclusion

Activation-level probes recover cross-distribution eval-vs-deploy
framing signals on two open-weight reasoning models of different
architecture families (Qwen3-32B dense, L55, AUROC 0.917; gpt-oss-120b
MoE, L34, AUROC 0.896). On Qwen3-32B L55 — the strongest probing layer
— steering on the probe-recovered direction (n = 100, two judges,
full control suite) produces behavioral effects **statistically
indistinguishable from random, orthogonal, and wrong-layer controls of
equal magnitude**: every probe Δ is within standard error of the
matched controls, despite output text changing in ~75% of prompts (the
same rate as controls).
A separate gpt-oss-120b L15 steering run shows a probe-direction
behavioral null. Together these support the claim that
**probe-recoverability does not imply causal sufficiency** — a
direction can be linearly decodable at AUROC 0.917 yet behaviorally no
more potent than noise.

We are precise about scope: the **full control suite (random,
orthogonal, wrong-layer, two judges) is complete for Qwen3-32B L55**,
where the probe direction is indistinguishable from every control;
the gpt-oss-120b L15 result is a corroborating probe-direction null
that does not yet have the matched control suite. The remaining open
items are the gpt-oss control suite and a necessity/ablation test —
not the Qwen3 conclusion. We do not claim the models lack any causal
eval-awareness mechanism — only that *this probe-recovered direction*,
at these layers, is not it under single-layer additive steering.

**The methodological lesson stands: activation-level safety claims
require level-3 causal tests with random/orthogonal controls, not
level-2 probe AUROC alone. A probe that separates eval from deploy at
0.917 told us nothing about whether that direction drives behavior —
and when we tested, it did not.**

---

## Appendix A — Judge rubric

[Copy from `judge_stage4_generations.py` JUDGE_USER_TEMPLATE]

## Appendix B — Pre-registered design (STAGE4_PLAN.md)

[Include or link the locked pre-registration]

## Appendix C — Reproducibility

[File paths, model SHAs, code versions]
