# Stage 3 Phase 0: Pre-flight Smoke Test

**Goal:** verify Qwen3-32B integration works end-to-end on a 10-document dataset before committing to the $2k-6k full run.

**Cost:** ~$5 total (~$3 GPU + ~$1 API + ~$1 misc)
**Time:** ~1-2 hours of pod time, half a day of attention

**Gate to Phase 1:** all checks below pass.

---

## Step 1: Spin up a single H100 pod

Same RunPod template as before (H100 80GB, runpod-torch-v240, 150 GB container). Just 1 GPU — this is a smoke test, not a full run.

## Step 2: Clone kitft repo and install patches

```bash
cd /workspace
git clone https://github.com/kitft/natural_language_autoencoders.git
cd natural_language_autoencoders

# Read setup.md for current install steps; expect:
pip install -e .

# Apply Miles + SGLang patches per docs/setup.md
# (look for setup commands like 'pip install miles' + 'bash patches/apply_sglang_patches.sh')
cat docs/setup.md | head -60
```

**Paste the head of setup.md back to me** so I can give exact install commands matching the current repo state.

## Step 3: Set env vars

```bash
export ANTHROPIC_API_KEY=hf_YOUR_REAL_KEY  # use Haiku 4.5 for cheapness
export HF_TOKEN=hf_YOUR_REAL_KEY
export BASE_MODEL=Qwen/Qwen3-32B
export LOSS_MASK_TYPE=qwen3
export NLA_EMBED_DUMP_DIR=/dev/shm/nla
mkdir -p /dev/shm/nla
```

## Step 4: Add Qwen3-32B preset

Edit `nla/datagen/model_presets.py`. Add this entry to the `MODELS` dict (location: alongside `qwen2_5_7b`, `gemma3_27b`, etc.):

```python
"qwen3_32b": {
    "base_model": "Qwen/Qwen3-32B",
    "num_layers": 64,
    "d_model": 5120,
    "head_dim": 128,
    "accepts_system_role": True,
    "extractor_kwargs": {
        "batch_size": 4,           # conservative for 32B
        "max_length": 4096,
        "device_map": "auto",
        "torch_dtype": "float16",
    },
    "turn_marker": "<|im_start|>",  # Qwen3 chat marker, verify with tokenizer
},
```

**Note:** the exact schema of `MODELS` dict may differ from this sketch — read the file first and match the existing entries' structure. If `qwen2_5_7b` is an entry, copy its schema and adapt the values above.

## Step 5: Create smoke-test datagen yaml

Create `configs/datagen/qwen3_32b_10docs_test.yaml`:

```yaml
# Smoke test: 10 docs, all Phase 0 plumbing
model: qwen3_32b
base_model: Qwen/Qwen3-32B
layer_index: 55

corpus: ultrafineweb-en
num_docs: 10
positions_per_doc: 5
seed: 42

split:
  av_sft: 0.4
  ar_sft: 0.4
  rl: 0.2

stage2:
  provider: anthropic
  model: claude-haiku-4-5-20251001  # cheap for smoke
  max_tokens: 300
  temperature: 1.0
  concurrency: 5  # don't hammer API on smoke

output_dir: /workspace/datagen_smoke/qwen3_32b_10docs
```

Copy `configs/datagen/gemma3_27b_*.yaml` as a template if any of the above field names are wrong; the schema is whatever the repo uses.

## Step 6: Run the pipeline

```bash
cd /workspace/natural_language_autoencoders
python -m nla.datagen.run_pipeline --config configs/datagen/qwen3_32b_10docs_test.yaml 2>&1 | tee preflight.log
```

Expected: completes in 10-30 minutes. Watch for crashes.

## Step 7: Verify outputs

```bash
# Stage 0 outputs
ls -la /workspace/datagen_smoke/qwen3_32b_10docs/

# Activations file should be present
find /workspace/datagen_smoke -name "*.parquet" -o -name "*.h5" -o -name "*.npz"

# Stage 2: check a few Claude explanations
find /workspace/datagen_smoke -name "*stage2*" -o -name "*explain*" | head -3
# Then look at one with python to see what came back
```

For each output type, paste the file listing back to me and 1-2 sample records (without committing them to git — these are just for inspection).

## Step 8: Pre-flight gate checks

Manual checks before declaring Phase 0 passed:

| Check | Pass if... | Fail action |
|---|---|---|
| Stage 0 completes | Parquet/h5 with ~50 activations (10 docs × 5 positions) | Debug stage 0; usually preset issue |
| Stage 2 produces explanations | At least ~80% of activations got non-empty `<analysis>` | Check API key, retry if rate-limited |
| Explanations are coherent | Sample 5 explanations: do they describe the input text plausibly? | If garbage: prompt issue, Claude having a bad day, or preset feeding wrong text |
| No CJK garbage anywhere | All explanations are English | Investigate injection_token auto-pick |
| Final parquet has correct schema | rows have (text, activation_vector, explanation) triples | Schema mismatch — usually a preset field |
| Memory footprint reasonable | Qwen3-32B forward pass at batch_size=4 fits in 80GB | OOM: reduce batch_size to 2 or 1 |

## Step 9: Save preflight artifacts and terminate pod

```bash
# From laptop, scp the log + a sample of outputs (no plaintext SAD-style concerns; this is our own data)
scp -i ~/.ssh/runpod-key -P <PORT> root@<POD_IP>:/workspace/natural_language_autoencoders/preflight.log ~/alignment-evals/v4/runs/stage3/
scp -i ~/.ssh/runpod-key -P <PORT> root@<POD_IP>:/workspace/datagen_smoke/qwen3_32b_10docs/*.parquet ~/alignment-evals/v4/runs/stage3/preflight_sample/
```

(Make sure `~/alignment-evals/v4/runs/stage3/` exists locally first.)

Then **terminate the pod** to stop the $3/hr clock.

## What success looks like

After Phase 0 completes successfully:
- We've verified Qwen3-32B integration with the kitft pipeline
- We have realistic timing data for stage 0 on Qwen3-32B
- We've confirmed Claude (Haiku 4.5) produces reasonable explanations for this corpus
- We're ready to commit to Phase 1 (full setup) and Phase 2 (full data gen)

Total Phase 0 spend should be under $10. If it goes over $25, something's wrong — pause and debug.

## What to paste back to me after Phase 0

1. Output of `cat docs/setup.md | head -60` from the cloned repo
2. Output of `head -100 nla/datagen/model_presets.py` so I can verify the preset schema
3. After the pipeline runs: `preflight.log` (full) + 3 sample stage-2 explanations + file listing of `/workspace/datagen_smoke/qwen3_32b_10docs/`
4. Any errors or weirdness you noticed

I'll give the next concrete steps (Phase 1 setup) based on what we see.
