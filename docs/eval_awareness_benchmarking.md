# Evaluation-awareness benchmarking implementation

This document describes the runnable Phase 1-5 stack for measuring whether a
prompt, transcript, or model rollout leaks that a model is being evaluated.

## Phase 1: validated leakage scanner

`LeakageValidator` runs `EvaluationAwarenessDetector` over labeled
`LabeledTranscript` examples and reports accuracy, precision, recall, F1, and
per-example false-positive/false-negative data. The repository includes a tiny
`starter_validation_set()` for CI and demos; public claims should use a larger
human-reviewed corpus.

## Phase 2: verbalized eval-awareness monitor

`HeuristicVerbalizedAwarenessJudge` scans model outputs or reasoning traces for
quotes like “this appears to be an evaluation,” benchmark identification,
synthetic-scenario recognition, grading awareness, and oversight awareness. It
returns quote spans, labels, confidence values, and an aggregate VEA score.

## Phase 3: paired behavior-shift runner

`PairedBehaviorRunner` builds obvious-eval, hidden-eval, and deployment-like
variants for each `BenchmarkTask`. It runs those variants against pluggable
`ModelClient` implementations and reports model-level VEA rates. This is the
core benchmark harness for comparing closed-source APIs, OpenAI-compatible
endpoints, local vLLM servers, and open-source HuggingFace models.

Included clients:

- `ScriptedModelClient` for deterministic offline tests and examples.
- `HTTPChatModelClient` for OpenAI-compatible `/chat/completions` endpoints,
  hosted gateways, or local vLLM/LiteLLM servers.
- `HuggingFaceLocalModelClient` for local `transformers` text-generation models.

## Phase 4: leakage-reducing rewrite workflow

`EvalLeakageRewriter` removes or neutralizes obvious benchmark/eval leakage,
filters risky metadata keys, reruns the detector, and reports score reduction
plus a coarse preserved-terms ratio. This is a deterministic baseline; it can be
replaced by an LLM rewrite engine while preserving the result schema.

## Phase 5: white-box probes and steering hooks

`LinearActivationProbe` scores caller-supplied activation vectors with calibrated
weights, and `SteeringVector` applies a simple activation-space shift for
caller-managed inference loops. These utilities do not capture activations
themselves; they provide stable interfaces for open-weight integrations.

## Reproducible artifacts

Use `--n-rollouts` to estimate pass-rate style behavior over repeated samples, `--dataset` to load a JSONL `EvalAwarenessExample` corpus, and `--output-dir` to write a manifest, raw rollout JSONL, summary JSON, and Markdown report.

## Local showcase

Run the offline deterministic benchmark:

```bash
python scripts/run_eval_awareness_benchmark.py --n-rollouts 3 --output-dir runs/local-demo
```

Run against an OpenAI-compatible endpoint:

```bash
export EVAL_AWARENESS_BASE_URL="https://your-provider.example/v1"
export EVAL_AWARENESS_API_KEY="..."
python scripts/run_eval_awareness_benchmark.py --dataset data/eval_awareness.jsonl --n-rollouts 10 --http-model your-model-name --http-name provider-model --output-dir runs/provider-model
```

The output contains validation metrics, paired rollout records, VEA quote
signals, rewrite deltas, confidence intervals, raw JSONL, CSV, HTML, and a
Markdown benchmark report. For a public benchmark post, run multiple seeds or
rollouts per variant and publish confidence intervals alongside raw JSON.

## Additional infrastructure

- `judges.py` provides Apollo-style LLM and ensemble VEA quote extraction.
- `providers.py` provides OpenAI-compatible, OpenAI, Anthropic, Gemini,
  OpenRouter, LiteLLM, vLLM, and Ollama clients.
- `realism.py` provides task/environment realism checks and iterative rewrites.
- `adversarial.py` provides deterministic prompt-environment mutation search.
- `whitebox.py` includes dependency-free difference-probe training and steering
  vector helpers for tiny/local model tests.
