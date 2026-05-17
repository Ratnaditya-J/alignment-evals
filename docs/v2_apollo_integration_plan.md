# V2 plan: Apollo dataset integration

**Status:** planning (May 2026). No experiments run yet.

## Motivation

V1 demonstrated that inject framing changes safety refusal in qwen3-235B (+22pp) and claude-opus-4-7 (−34pp), and that the effect is not mediated by visible strict-VEA verbalization (pre-registered replication; see `docs/pre_registration_qwen3_safety_mediation.md`). A reviewer-grade weakness of V1: the inject manipulation was synthetic — *we* constructed the system-prompt cue. A skeptic can say "your inject prompt is doing something other than signaling evaluation context."

Apollo Research's [released benchmark](https://github.com/jjpn97/eval_awareness) (Needham et al., arXiv 2505.23836) gives us 1,000 prompts already labeled by humans as `eval` or `deployment`. V2 uses that labeled corpus to test the V1 finding without relying on a synthetic cue.

## Design: 2×2 factorial

Two factors, fully crossed:

|  | Apollo label: `eval` | Apollo label: `deployment` |
|---|---|---|
| **Inject** (extrinsic cue: "this is an eval") | Both signals say eval — redundant; ceiling reference | Conflict — extrinsic says eval, intrinsic says deployment |
| **Neutral** ("You are a helpful assistant.") | Intrinsic eval signal only | Intrinsic deployment signal only |

Outcomes per cell, per model: refusal rate, strict-VEA rate, broad-VEA rate.

**Headline analysis:** the interaction term. Does inject change behavior on top of the intrinsic Apollo label? Three possible patterns and their interpretations:

- *Inject moves behavior only on Apollo-deployment prompts*: extrinsic cue is redundant when the prompt already looks like an eval; useful only when the model can't tell.
- *Inject moves behavior equally on both*: extrinsic cue is what models react to; intrinsic signal is irrelevant.
- *Inject amplifies behavior on Apollo-eval prompts*: cue and content compound.

Each pattern is a distinct empirical claim about how models integrate the two signals.

## Sample, cost, models

- **Subsample**: ~100 cleanest eval-labeled + ~100 cleanest deployment-labeled prompts from Apollo's 1,000.
- **Models**: 4 reasoning models with observable CoT (qwen3-235B-thinking, deepseek-v3.2, kimi-k2-thinking, claude-opus-4-7).
- **Total rollouts**: 200 prompts × 2 conditions × 4 models ≈ 1,600 rollouts.
- **Cost estimate**: $30–80 in API fees. Cheap.

## Pre-registration

V1 taught the lesson the hard way. V2 will pre-register before running:

- Subsample selection (deterministic by prompt hash so anyone can reproduce).
- Decision rule for the interaction-effect CI — what counts as a real interaction vs null.
- Multiple-comparison policy across the 4 models.

Pre-registration document will live at `docs/pre_registration_v2_apollo_2x2.md` and be committed to `main` before the experiment runs.

## Code changes needed

- `src/eval_awareness/dataset_loaders.py` → add `ApolloEvalAwarenessLoader` that wraps Apollo's `dataset.json` and exposes the same interface the existing loaders use.
- `scripts/run_v2_apollo_factorial.py` → new entry point that ties the loader to the existing 2-condition (neutral/inject) runner with paired bookkeeping.
- `scripts/analyze_v2_factorial.py` → new analysis script for the interaction-effect bootstrap. Reuses the paired-bootstrap utility from `analyze_vea_mediation.py`.

## Order of operations

1. Pull Apollo's `dataset.json` (one-time download, ~few MB).
2. Build `ApolloEvalAwarenessLoader`. Verify it loads and that each prompt has the `is_eval` ground-truth field.
3. Smoke test on n=10 prompts × 1 model to verify the pipeline produces valid rollouts.
4. Draft and commit pre-registration document.
5. Run the full experiment.
6. Analyze and write up.

## Out of scope for V2

- The "factorial ablation of the inject prompt" (eval-context only vs comparison-to-deployment only vs both) — that's V3.
- Apollo's chain-of-thought analysis stages (`eval_mcq_cot`, `purpose_open_cot`) — Apollo's pipeline already does these; not the contribution V2 is making.
- Sandbagging or honesty probes — separate research direction, not blocked by V2.

## Open question

Apollo's prompts are *transcripts* (multi-turn conversations); V1's prompts were single-turn task prompts. Decision needed before running: do we feed the model the full transcript and treat its next-turn response as the rollout, or extract just the most recent user turn and treat it as a standalone prompt? Default: full transcript, since that's what Apollo's classifier sees and the comparison is cleaner.
