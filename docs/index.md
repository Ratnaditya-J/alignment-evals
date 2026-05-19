---
title: The Prompt Is the Tell, Not the Reasoning Trace
---

# The Prompt Is the Tell, Not the Reasoning Trace

Across **32,170 rollouts**, eval-related prompt cues predicted refusal shifts more reliably than verbalized eval-awareness in model traces.

**Short version:** the prompt was a better clue than the reasoning trace.

[Read the Medium post](https://medium.com/@ratnaditya/the-prompt-is-the-tell-not-the-reasoning-trace-eval-awareness-241287e9ac70) ·
[Paper PDF](paper.pdf) ·
[Code and data](https://github.com/Ratnaditya-J/alignment-evals) ·
[Rollout release](https://github.com/Ratnaditya-J/alignment-evals/releases/tag/v1.0-rollouts) ·

## Key Results

### 1. Intrinsic eval cues predicted refusal across models

Apollo Research's eval-awareness corpus labels prompts as either evaluation prompts or deployment-style prompts. Across six frontier and frontier-adjacent models from four labs, eval-labeled prompts produced **+7.5 to +13.0 percentage point** higher refusal rates than deployment-labeled prompts. Five of six models survived Holm-Bonferroni correction.

![V2 intrinsic-label main effect](figures/fig6_v2_intrinsic_main_effect.png)

### 2. The explicit eval-framing effect was distributed across clauses

The V1 inject sentence combined three cues: "this is an evaluation," "reviewed by safety researchers," and "compared against deployment behavior." In V3, no singleton clause survived correction. On Opus and qwen3, the sum of singleton effects matched the full-inject effect within the pre-registered additivity residual confidence interval.

This is **additive-compatible**, not proof of exact additivity.

![V3 clause attribution](figures/fig8_v3_clause_attribution.png)

### 3. The same Opus/qwen3 direction asymmetry appeared across three samples

On safety/alignment-relevant prompts, Opus became less likely to refuse under explicit eval framing, while qwen3 became more likely to refuse. The sign pattern appeared in V1, V2, and V3, with smaller magnitudes on the broader Apollo corpus.

Lines connect the same model across samples; they are not a time trend.

![V1 V2 V3 triangulation](figures/fig7_triangulation.png)

## What This Does Not Claim

- It does **not** prove sandbagging.
- It does **not** prove that models internally know they are being evaluated.
- It does **not** identify a randomized causal mechanism for the V2 intrinsic-label result.
- It does **not** say reasoning traces are useless.
- It does **not** apply to activation-level interpretability monitors such as Anthropic's Natural Language Autoencoders.

The narrower claim is: **text-level reasoning-trace mentions were not a reliable warning light for evaluation-induced refusal shifts in these experiments.**

## Artifacts

- [Medium post](https://medium.com/@ratnaditya/the-prompt-is-the-tell-not-the-reasoning-trace-eval-awareness-241287e9ac70)
- [Paper PDF](paper.pdf)
- [Technical draft](paper_draft.md)
- [Rollout data and summary JSONs](https://github.com/Ratnaditya-J/alignment-evals/releases/tag/v1.0-rollouts)
- [GitHub repository](https://github.com/Ratnaditya-J/alignment-evals)
