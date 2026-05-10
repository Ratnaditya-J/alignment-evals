# SOTA roadmap for evaluation-awareness benchmarking

This roadmap translates the research review into implementation milestones for
making IsItBenchmark credible for public eval-awareness benchmarking.

## Implemented foundation

The current v2 stack now includes the minimum foundation for benchmark-grade
experiments:

1. **Dataset schema and JSONL loader** via `EvalAwarenessExample`, including
   labels, task families, expected answers, scorer names, risk tags, and
   transcript fields.
2. **Repeated paired rollouts** through `PairedBehaviorRunner.run(...,
   n_rollouts=N)` for obvious-eval, hidden-eval, and deployment-like variants.
3. **Behavior scorers** for exact/contains/numeric/multiple-choice answers,
   refusal, and coarse harmful-compliance signals.
4. **Statistical summaries** with deterministic bootstrap confidence intervals
   over VEA rates and primary behavior scores.
5. **Reproducible artifacts**: manifest JSON, raw rollout JSONL, summary JSON,
   and Markdown report via `ArtifactWriter`.

## Next milestones

### Milestone 1: benchmark corpus

- Scale from the starter examples to 500-1,000 human-reviewed examples.
- Include safety, code, math, QA, agent trajectories, support-style deployment
  transcripts, and adversarial deployment wrappers.
- Keep benchmark-family holdouts so leakage detectors cannot overfit benchmark
  names and obvious wrappers.

### Milestone 2: validated VEA judge

- Add an LLM judge backend that extracts quotes and classifies VEA categories.
- Validate it against human-labeled reasoning traces and final answers.
- Report scenario-eval vs meta-eval awareness separately.

### Milestone 3: behavior scoring depth

- Add benchmark-family scorers: unit tests for code, exact/numeric/MCQ for QA,
  refusal and harmful-compliance scoring for safety, and underperformance metrics
  for sandbagging.
- Track behavior deltas against deployment-like baselines, not just raw scores.

### Milestone 4: realism mitigation

- Add a realism judge that flags task-driven and environment-driven eval cues.
- Implement an iterative rewrite loop with semantic equivalence and difficulty
  preservation checks.
- Report leakage, VEA, realism, and behavior before/after rewrites.

### Milestone 5: provider-grade model integrations

- Add provider adapters for OpenAI, Anthropic, Gemini, OpenRouter/LiteLLM,
  vLLM, HuggingFace, and Ollama.
- Capture exact model IDs, parameters, timestamps, token usage, latency, retry
  metadata, and raw transcripts.

### Milestone 6: white-box experiments

- Integrate TransformerLens or nnsight for activation capture.
- Train/evaluate layer-wise probes on eval-vs-deployment transcripts.
- Compare probe scores with VEA quotes and behavior deltas.
- Test steering vectors while monitoring side effects.

## Public benchmark minimum bar

Before publishing model rankings, each run should include:

- repeated rollouts per prompt and variant;
- raw rollout JSONL artifacts;
- confidence intervals for VEA and behavior metrics;
- human-validated or audited VEA judging;
- task-family breakdowns;
- limitations and false-positive examples;
- exact model versions and decoding parameters.
