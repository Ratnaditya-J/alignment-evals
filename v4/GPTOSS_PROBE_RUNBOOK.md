# gpt-oss-120b Activation/Probe Validation — Runbook

**Goal:** replicate Qwen3-32B L55 finding on OpenAI's gpt-oss-120b. Establish cross-architecture cross-lab evidence for activation-level eval-awareness.

**Cost:** ~$30 total ($15-25 GPU + ~$10 OpenRouter for any behavior re-runs)
**Time:** 2-3 days end-to-end (mostly waiting on pod time and model downloads)
**Decision gate:** does gpt-oss-120b have a probe layer that beats MS-trained TF-IDF baseline (~0.83) on the controlled holdout? If yes → strong two-model paper. If no → publishable cross-model contrast.

---

## Architecture facts (gpt-oss-120b)

- **117B total params, 5.1B active per token** (Mixture of Experts)
- **36 layers**, hidden_size = 2880
- **128 experts per MoE layer, 4 active per token**
- Apache 2.0 licensed (`openai/gpt-oss-120b` on HuggingFace)
- Ships in **mxfp4 quantization natively** — designed to fit on 1× H100 80GB
- For probe work the MoE detail doesn't matter — residual stream is shared across tokens regardless of which experts routed; `output_hidden_states` extracts the same shape as a dense model

## Layer sweep plan

Qwen3-32B (64 layers) peaked at L55 = relative depth **0.86**. We test gpt-oss-120b at 8 layers covering similar depth range plus the depth-matched candidate:

| Layer | Depth | Note |
|---|---|---|
| 5 | 0.14 | Early baseline |
| 10 | 0.28 | Matches Qwen3 L18 region |
| 15 | 0.42 | Mid-stream |
| 20 | 0.56 | Around Qwen3 L36 |
| 25 | 0.69 | Around Qwen3 L44 |
| 28 | 0.78 | |
| **31** | **0.86** | **Depth-matched to Qwen3 L55** |
| 34 | 0.94 | Late, near output |

If the depth-0.86 hypothesis is right, L31 is the peak. If it's a Qwen-specific artifact, we won't see it at any layer.

---

## Phase 1: Pod setup (~10 min, ~$0.50)

**Spin up RunPod:**
- 1× H100 80GB SXM (Community Cloud, ~$2-3/hr)
- Template: `runpod-torch-v240` (PyTorch 2.4 + CUDA preinstalled)
- Container disk: **200 GB** (gpt-oss-120b weights are ~60-70 GB in mxfp4 + Python env + activations)
- SSH key + Jupyter as you've done before

**From your laptop, scp scripts and data:**

```bash
scp -i ~/.ssh/runpod-key -P <PORT> \
    ~/alignment-evals/v4/scripts/extract_acts_any_model_pod.py \
    ~/alignment-evals/v4/prompts/controlled_holdout_v2.yaml \
    root@<POD_IP>:/workspace/

scp -i ~/.ssh/runpod-key -P <PORT> \
    /tmp/Test_Awareness_Steering/data/triggers/triggers.json \
    root@<POD_IP>:/workspace/triggers.json
```

(If you don't have triggers.json local anymore, on the pod: `curl -sL https://raw.githubusercontent.com/microsoft/Test_Awareness_Steering/main/data/triggers/triggers.json -o /workspace/triggers.json`)

## Phase 2: Pod environment setup (~10 min)

SSH into the pod. Then:

```bash
cd /workspace

# Latest transformers needed for gpt-oss support (added in 4.55+)
pip install -U "transformers>=4.55" accelerate bitsandbytes pyyaml

# Verify versions
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# HF token for download (gpt-oss is Apache 2.0 but auth helps with rate limits)
export HF_TOKEN=hf_YOUR_TOKEN
huggingface-cli login --token $HF_TOKEN
```

If transformers install conflicts with existing packages, force the upgrade:
```bash
pip install --force-reinstall --no-deps "transformers>=4.55"
```

## Phase 3: Download gpt-oss-120b (~5-10 min, ~60-70 GB)

```bash
mkdir -p /workspace/models
hf download openai/gpt-oss-120b --local-dir /workspace/models/gpt-oss-120b

# Verify size
du -sh /workspace/models/gpt-oss-120b
```

Expected size: ~60-70 GB (mxfp4 format).

## Phase 4: Smoke test (~5 min, ~$0.20)

Before committing to a full extraction, verify the pipeline works on a tiny subset. Use `--limit 6` to process only first 3 pairs (6 prompts) from the controlled holdout.

```bash
cd /workspace
python extract_acts_any_model_pod.py \
    --model-path /workspace/models/gpt-oss-120b \
    --model-tag gpt_oss_120b_smoke \
    --input-yaml /workspace/controlled_holdout_v2.yaml \
    --layers 10 20 31 \
    --limit 6 \
    --output-dir /workspace/v4_outputs 2>&1 | tee /workspace/gpt_oss_smoke.log
```

**Smoke test passes if:**
- Model loads without OOM
- Prints `n_layers=36, d_model=2880, Model class: GptOss...ForCausalLM`
- Extracts 6 activations across 3 layers without crash
- Output npz exists at `/workspace/v4_outputs/gpt_oss_120b_smoke_controlled_activations.npz`

**Common failure modes:**

| Error | Fix |
|---|---|
| `OOM during model load` | Add `--load-in-4bit` flag (uses bitsandbytes int4 alternative quantization) |
| `KeyError: 'gpt_oss'` or unknown architecture | transformers too old; `pip install -U "transformers>=4.55"` |
| `output_hidden_states not supported` | gpt-oss may need a specific transformers version that exposes hidden states for MoE; try `pip install -U "transformers>=4.56"` |
| `Module 'flash_attn' missing` | `pip install flash-attn --no-build-isolation` OR add `attn_implementation="eager"` to the script |
| `apply_chat_template fails` | The script has a fallback to raw tokenization; should auto-recover |

**Paste the smoke test output to me before proceeding to Phase 5.** Especially the line that says `n_layers=X, d_model=Y, Model class: Z` — verifies our architecture assumptions.

## Phase 5: Full extraction (Microsoft triggers → probe training data) (~15-25 min, ~$8-12)

```bash
cd /workspace
python extract_acts_any_model_pod.py \
    --model-path /workspace/models/gpt-oss-120b \
    --model-tag gpt_oss_120b \
    --input-triggers /workspace/triggers.json \
    --layers 5 10 15 20 25 28 31 34 \
    --output-dir /workspace/v4_outputs 2>&1 | tee /workspace/gpt_oss_triggers_extract.log
```

Expected: ~680 prompts (340 Microsoft pairs × 2 arms) × 8 layers, ~15-25 min on H100.

Output: `/workspace/v4_outputs/gpt_oss_120b_triggers_activations.npz` (~80-100 MB).

## Phase 6: Full extraction (controlled holdout v2 → eval data) (~5-10 min, ~$3-5)

```bash
cd /workspace
python extract_acts_any_model_pod.py \
    --model-path /workspace/models/gpt-oss-120b \
    --model-tag gpt_oss_120b \
    --input-yaml /workspace/controlled_holdout_v2.yaml \
    --layers 5 10 15 20 25 28 31 34 \
    --output-dir /workspace/v4_outputs 2>&1 | tee /workspace/gpt_oss_controlled_extract.log
```

Expected: 320 prompts × 8 layers, ~5-10 min.

Output: `/workspace/v4_outputs/gpt_oss_120b_controlled_activations.npz` (~30-40 MB).

## Phase 7: scp results back to laptop

```bash
scp -i ~/.ssh/runpod-key -P <PORT> \
    "root@<POD_IP>:/workspace/v4_outputs/gpt_oss_120b_*.npz" \
    "root@<POD_IP>:/workspace/v4_outputs/gpt_oss_120b_*.json" \
    "root@<POD_IP>:/workspace/gpt_oss_*.log" \
    ~/alignment-evals/v4/runs/qwen3_stage2/
```

## Phase 8: Terminate pod

Once files confirmed local, terminate pod to stop billing. ~$15-25 total spend.

## Phase 9: Local probe training (~10 sec, $0)

```bash
cd ~/alignment-evals
./.venv/bin/python v4/scripts/train_probes_any_model.py \
    --acts v4/runs/qwen3_stage2/gpt_oss_120b_triggers_activations.npz \
    --model-tag gpt_oss_120b \
    --output-dir v4/runs/qwen3_stage2
```

Output: `v4/runs/qwen3_stage2/frozen_probes_gpt_oss_120b.npz` + metadata json.

The CV AUROC per layer tells us whether gpt-oss-120b has the Microsoft entity-fakeness signal AT ALL. Expect CV >0.9 at some layer if so (Qwen3 had 0.996 at L55).

## Phase 10: Local scoring (~10 sec, $0) — THE PUNCHLINE

```bash
./.venv/bin/python v4/scripts/score_holdout_any_model.py \
    --probes v4/runs/qwen3_stage2/frozen_probes_gpt_oss_120b.npz \
    --probes-meta v4/runs/qwen3_stage2/frozen_probes_gpt_oss_120b_meta.json \
    --holdout-acts v4/runs/qwen3_stage2/gpt_oss_120b_controlled_activations.npz \
    --holdout-yaml v4/prompts/controlled_holdout_v2.yaml \
    --triggers /tmp/Test_Awareness_Steering/triggers.json \
    --model-tag gpt_oss_120b \
    --output v4/runs/qwen3_stage2/validation_gpt_oss_120b.json
```

The output table shows per-layer probe AUROC on the controlled holdout, vs MS-trained TF-IDF baseline (0.83 from earlier sanity check).

**Pre-registered verdict thresholds:**

| Peak probe AUROC | What it means |
|---|---|
| ≥ 0.95 | STRONG cross-architecture replication. Bulletproof paper. |
| 0.90-0.95 | Solid replication. Two-model story locked. |
| 0.85-0.90 | Above baseline. Solid mod evidence. |
| 0.83 ± 0.03 | At baseline. gpt-oss probe matches TF-IDF, no activation-specific signal. |
| < 0.80 | Below baseline. gpt-oss lacks the signal Qwen3 has. **Cross-model contrast story.** |

## Decision tree after Phase 10

- **gpt-oss-120b peak ≥ 0.85**: cross-lab replication confirmed. Paper headline: "Activation-level eval-awareness validated on Qwen3-32B and gpt-oss-120b, across families/architectures, beats text baselines." Then either commit to Stage 3 (custom NLA training) OR write up V4b as-is.

- **gpt-oss-120b peak 0.83 ± 0.03**: probe matches text baseline. Means our methodology works (Qwen3 result holds) but gpt-oss doesn't have an above-text-baseline activation signal at any tested layer. Still publishable: "Activation-level eval-awareness is model-specific; not all open-weight models exhibit it."

- **gpt-oss-120b peak < 0.80**: cross-model contrast finding. Qwen3 has the signal, gpt-oss doesn't. Also publishable, and arguably more interesting from an interpretability standpoint (why does training/architecture matter?). Pair with gpt-oss-20b extraction (negative control) for clean scale story.

---

## Optional: gpt-oss-20b in parallel (~$5, +10 min)

If you want the scale comparison in the same pod session, run extraction on gpt-oss-20b after the 120b runs finish. Same script, different model path:

```bash
hf download openai/gpt-oss-20b --local-dir /workspace/models/gpt-oss-20b

# 20b has 24 layers (vs 36 for 120b); scale layer indices accordingly
python extract_acts_any_model_pod.py \
    --model-path /workspace/models/gpt-oss-20b \
    --model-tag gpt_oss_20b \
    --input-triggers /workspace/triggers.json \
    --layers 4 8 12 16 18 20 22 \
    --output-dir /workspace/v4_outputs 2>&1 | tee /workspace/gpt_oss_20b_triggers_extract.log

python extract_acts_any_model_pod.py \
    --model-path /workspace/models/gpt-oss-20b \
    --model-tag gpt_oss_20b \
    --input-yaml /workspace/controlled_holdout_v2.yaml \
    --layers 4 8 12 16 18 20 22 \
    --output-dir /workspace/v4_outputs 2>&1 | tee /workspace/gpt_oss_20b_controlled_extract.log
```

20b at bf16 is ~40GB, fits easily on the same H100. No quantization needed.

Then scp back, train probes, score — same Phase 9-10 commands but `--model-tag gpt_oss_20b`.

---

## Checklist before kicking off

- [ ] Kimi K2 sweep finished (or running in parallel)
- [ ] RunPod credit balance ≥ $30
- [ ] HuggingFace account has access to openai/gpt-oss-120b (visit https://huggingface.co/openai/gpt-oss-120b — should be open-access)
- [ ] Local has Microsoft triggers.json at /tmp/Test_Awareness_Steering/triggers.json
- [ ] Local has controlled_holdout_v2.yaml at v4/prompts/
- [ ] SSH key working with RunPod
- [ ] `set -a; source v4/.env; set +a` in the shell you'll scp from

## Cumulative cost projection

| Phase | Cost |
|---|---|
| Pod time (gpt-oss-120b only): ~1.5 hr × $3/hr | $4.50 |
| Pod time (with gpt-oss-20b add-on): ~2 hr × $3/hr | $6 |
| OpenRouter (if any reruns needed) | ~$2-5 |
| **Total** | **~$10-15** |

(My earlier $30 estimate was conservative; this scopes tighter.)

## What I need from you between phases

- After Phase 4 (smoke): paste the smoke log so I can verify architecture
- After Phase 10: paste the per-layer scoring table so we can interpret jointly
