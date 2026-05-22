# Stage 4 scripts — how to use

Implements the pre-registered design in [`v4/STAGE4_PLAN.md`](../STAGE4_PLAN.md).

## Files

| File | Purpose |
|---|---|
| `prepare_steering_vectors.py` | Compute all 6 steering directions (probe coef + mean-diff in raw/z-scored, plus random + orthogonal) from existing V4b artifacts. Run once per (model, layer). |
| `run_steered_generation.py` | Run text generation with a forward-hook intervention at the chosen layer. Has built-in correctness tests (`--test`) that MUST pass before any real runs. |

## Order of operations

### 0. Prerequisites
- V4b artifacts must exist locally:
  - Qwen3: `v4/runs/qwen3_stage2/{qwen3_activations.npz, frozen_probes_qwen3.npz}`
  - gpt-oss: `v4/runs/gpt_oss_stage2/{gpt_oss_120b_triggers_activations.npz, frozen_probes_gpt_oss_120b.npz}`
- A V1-prompts JSONL needs to be assembled (NOT done yet — see "Still TODO" below).
- For runs, a pod with same setup as V4b extraction (1× H200 + PyTorch 2.8 + `kernels` + `triton-kernels @ release/3.4.x`).

### 1. Prepare steering vectors (local, fast)
```bash
mkdir -p ~/alignment-evals/v4/runs/stage4

python3 ~/alignment-evals/v4/scripts/prepare_steering_vectors.py \
    --triggers-acts ~/alignment-evals/v4/runs/qwen3_stage2/qwen3_activations.npz \
    --probes ~/alignment-evals/v4/runs/qwen3_stage2/frozen_probes_qwen3.npz \
    --layer 55 \
    --model-tag qwen3_32b \
    --output ~/alignment-evals/v4/runs/stage4/steering_vectors_qwen3_32b_L55.npz

python3 ~/alignment-evals/v4/scripts/prepare_steering_vectors.py \
    --triggers-acts ~/alignment-evals/v4/runs/gpt_oss_stage2/gpt_oss_120b_triggers_activations.npz \
    --probes ~/alignment-evals/v4/runs/gpt_oss_stage2/frozen_probes_gpt_oss_120b.npz \
    --layer 34 \
    --model-tag gpt_oss_120b \
    --output ~/alignment-evals/v4/runs/stage4/steering_vectors_gpt_oss_120b_L34.npz
```

Each call:
- Loads the trained probe + triggers activations
- Computes per-dim z-scoring stats
- Produces 6 unit-norm directions + the z-score stats, all in one `.npz`
- Prints cosine similarities between directions (sanity check — `probe_coef vs mean_diff` should be high positive; `mean_diff_zscored vs orthogonal` should be ≈0)
- Writes a sibling `.json` with metadata

### 2. Run correctness tests on the pod (CRITICAL — do not skip)
```bash
# On the H200 pod, after scp'ing both scripts + the steering vectors npz
python -u extract_acts_any_model_pod.py    # (still needed: just confirms model loads)
python -u ~/alignment-evals/v4/scripts/run_steered_generation.py --test \
    --model-path /tmp/gpt-oss-120b \
    --steering-vectors /workspace/steering_vectors_gpt_oss_120b_L34.npz \
    --layer 34 \
    --dtype bfloat16
```

What the tests verify:
- **T1**: `alpha=0` hook produces output identical to no-hook (hook is a no-op when off)
- **T2**: with `alpha > 0`, the residual stream at the target layer is shifted by *exactly* `alpha * w` at all prompt positions (verified by inspecting the hook's recorded diff)
- **T3**: `alpha > 0` produces a different generation than `alpha = 0`
- **T4**: hook is properly removed after generation (no leftover state)

If any test fails, STOP. The math is load-bearing; do not run Stage 4 with a broken hook.

### 3. Smoke test (real prompts, small)
```bash
python -u ~/alignment-evals/v4/scripts/run_steered_generation.py \
    --model-path /tmp/gpt-oss-120b \
    --model-hf-id openai/gpt-oss-120b \
    --steering-vectors /workspace/steering_vectors_gpt_oss_120b_L34.npz \
    --layer 34 \
    --prompts-jsonl /workspace/v1_prompts_smoke.jsonl \
    --conditions mean_diff_zscored random \
    --alpha-values -1 0 1 \
    --limit 6 \
    --output-jsonl /workspace/stage4_smoke.jsonl
```
6 prompts × 2 conditions × 3 alphas = 36 generations. Should complete in 2-3 min.

### 4. Full run
Per the plan: 500 V1 prompts × 7 alphas × ~3 conditions × 2 models. ~3-4 hrs per model on H200.

## Still TODO

1. **V1-prompts JSONL** (`v4/runs/stage4/v1_prompts_500.jsonl`): one record per line with fields `prompt_id`, `text`, `family`, `v1_condition`. Source: the V1 corpus (BeaverTails, Apollo, MMLU, GSM8K subsets used in `docs/paper_draft.md` §3.1). Probably ~30 min to assemble.

2. **Statistical analysis script** (`v4/scripts/analyze_stage4.py`): paired bootstrap CIs, dose-response fits, per-family stratification, comparison vs V1 inject effects. Wait until we have judged data to write this — easier with the schema in hand.

## Judge script (done)

`judge_stage4_generations.py` reads the JSONL of generations, queries Opus + GPT, and writes per-record scores. Supports resume (re-running skips already-judged records). Requires `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` env vars.

```bash
# Estimate cost before running
python3 ~/alignment-evals/v4/scripts/judge_stage4_generations.py \
    --generations-jsonl v4/runs/stage4/gpt_oss_L34_generations.jsonl \
    --output-jsonl v4/runs/stage4/gpt_oss_L34_judged.jsonl \
    --dry-run

# Actually run
python3 ~/alignment-evals/v4/scripts/judge_stage4_generations.py \
    --generations-jsonl v4/runs/stage4/gpt_oss_L34_generations.jsonl \
    --output-jsonl v4/runs/stage4/gpt_oss_L34_judged.jsonl \
    --judges opus gpt5 \
    --max-concurrent 4
```

## JSONL prompt format expected by `run_steered_generation.py`

One record per line:
```json
{"prompt_id": "bt_0001", "text": "...", "family": "safety", "v1_condition": "deploy"}
```

Required fields: `text`. Optional: `prompt_id`, `family`, `v1_condition` (all propagate into the output records).

## JSONL output format

One record per (prompt, condition, alpha):
```json
{
  "prompt_id": "bt_0001",
  "prompt_text": "...",
  "family": "safety",
  "v1_condition": "deploy",
  "model_path": "/tmp/gpt-oss-120b",
  "model_hf_id": "openai/gpt-oss-120b",
  "layer": 34,
  "condition": "mean_diff_zscored",
  "alpha": 1.0,
  "position": "all",
  "generated_text": "..."
}
```

These records are the input to the judging script (TODO).
