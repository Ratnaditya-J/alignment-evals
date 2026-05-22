# V4b Stage 3: NLA Training Plan — Qwen3-32B at L55

**Status:** planning complete, ready to execute Phase 0
**Budget:** $2,500 (lean, recommended) to $6,000 (production)
**Wall clock:** 1-2 weeks realistic, 3-4 days happy-path
**Source of truth:** `https://github.com/kitft/natural_language_autoencoders` (Apache 2.0)
**Goal:** train a Natural Language Autoencoder that verbalizes Qwen3-32B's L55 residual-stream activations, then use it to characterize what L55's eval-awareness direction "looks like" in plain English

---

## Executive summary

We've validated L55 captures cross-distribution eval-awareness signal on Qwen3-32B (controlled holdout v2: AUROC 0.917 at n=320, beats MS-trained TF-IDF baseline by +0.086 overall and +0.585 on Family B at ~9σ). Stage 3 trains an NLA so we can read what L55 is encoding in English.

The kitft repo has a complete training pipeline (data gen → AR SFT → AV SFT → RL) that's been used to produce 4 released NLAs (Qwen2.5-7B, Gemma-3-12B/27B, Llama-3.3-70B). Adapting to Qwen3-32B at L55 requires:
- ~5 small file changes (model preset, datagen yaml, env vars)
- Fresh stage-0 (GPU forward passes on Qwen3-32B) + stage-2 (Claude API explanations) — no cache reuse since Qwen3-32B has a different tokenizer than any released model
- ~$240 of GPU for stage 0, ~$300-3000 of API for stage 2, ~$2400 of GPU for RL

---

## Phased execution plan

Each phase has a gate. Don't enter Phase N+1 unless Phase N's gate passes.

### Phase 0: Pre-flight smoke test ($1-50, 1 day)

**Goal:** verify Qwen3-32B plumbing works end-to-end on a 10-doc dataset BEFORE committing to the full 100k-doc run.

**Why it matters:** kitft mentions specific failure modes (CJK output, NCCL deadlock, OOM on length drift, embed-vs-input-ids confusion). Better to catch them at $1 cost than at $200 into the real run.

**Tasks:**
1. Rent 1× H100-80GB (~$3/hr) — cheapest pod is fine
2. Clone kitft repo + apply Miles + SGLang patches per `docs/setup.md`
3. Create minimal `configs/datagen/qwen3_32b_10docs_test.yaml` (copy gemma3_27b_*.yaml, swap model preset)
4. Add `qwen3_32b` entry to `nla/datagen/model_presets.py`
5. Run stage 0 + stage 2 on 10 docs (`python -m nla.datagen.run_pipeline --config <yaml>`)
6. Inspect output: are activations being extracted, are Claude explanations sensible

**Gate to Phase 1:**
- Stage 0 completes without crash
- Stage 2 produces non-empty `<analysis>` blocks
- A few spot-checked explanations are coherent (i.e., Claude actually understood the text)
- No CJK garbage in any outputs
- Total cost ≤ $5 ($1 API + $3 GPU-hr)

**If it fails:** debug before spending real money. Common causes per kitft docs:
- Wrong injection scale for Qwen3 (auto-picker should handle this; verify it picked a single-token CJK char)
- Missing SGLang patches
- `--gradient-checkpointing` enabled during RL (causes NCCL deadlock)
- Forgetting to identity-init value_head in `prepare_critic_checkpoint.py`

### Phase 1: Full setup + Qwen3-32B integration ($30-50, 1-2 days)

**Goal:** complete the Qwen3-32B preset, write production configs, verify large-scale data extraction works.

**Tasks:**
1. Finalize `nla/datagen/model_presets.py` entry for `qwen3_32b`:
   - `num_layers=64`, `d_model=5120`, `head_dim=128`
   - `extractor_kwargs={batch_size: 4, max_length: 4096, device_map: "auto"}` (start conservative for 32B)
   - `turn_marker`: verify Qwen3 chat template tokens
   - `accepts_system_role=True`
2. Set `LOSS_MASK_TYPE=qwen3` in training env vars (Miles patch already has this handler)
3. Write production `configs/datagen/qwen3_32b_fineweb_100k.yaml`:
   - `model: qwen3_32b`, `base_model: Qwen/Qwen3-32B`
   - `layer_index: 55` (critical — our finding)
   - `corpus: ultrafineweb-en` (per kitft default)
   - `num_docs: 100000`, `positions_per_doc: 10`
   - `split: {av_sft: 0.25, ar_sft: 0.25, rl: 0.5}`
4. Write `configs/{actor_sft,critic_sft,rl}_qwen3_32b.sh`:
   - Copy existing configs, swap `BASE_MODEL=Qwen/Qwen3-32B`
   - Adjust per-stage GPU counts and batch sizes for 32B scale
   - Set `LOSS_MASK_TYPE=qwen3`
   - Set `NLA_EMBED_DUMP_DIR=/dev/shm/nla` (not /tmp — kitft warns about overlay-fs slowdown)
5. Decide on Stage 2 model: **Haiku 4.5 (~$300) or Sonnet 4.6 (~$3000)** — see "Open decisions" below
6. Test the production datagen yaml on a 1000-doc slice (~$3 of API + $30 GPU)

**Gate to Phase 2:**
- 1000-doc slice runs to completion
- Stage 0 throughput is reasonable (~10 docs/sec/GPU at batch_size=4, max_length=4096)
- Stage 2 API calls succeed at >95% rate
- Generated parquet files are well-formed (correct schema, no NaNs in activations)

### Phase 2: Full data generation ($240 GPU + $300-3000 API, 1-2 days)

**Goal:** generate the full ~1M-vector training dataset.

**Tasks:**
1. Rent 8× H100-80GB (cheapest provider — RunPod community, Vast.ai)
2. Run `nla.datagen.run_pipeline` on the production yaml
3. Monitor: stage 0 typically takes ~10 hours on 8× H100 for 100k docs at 32B scale
4. Stage 2: ~500k API calls, parallelism=100 → ~2-4 hours wall clock (rate-limited)
5. Stage 3: merge into train-ready parquets

**Cost breakdown:**
- Stage 0 GPU: ~$240 (8 GPUs × 10 hr × $3/hr)
- Stage 2 Claude API: $300 (Haiku) or $3000 (Sonnet)
- Stage 1+3 CPU: negligible

**Gate to Phase 3:**
- 250k AV-SFT rows + 250k AR-SFT rows + 500k RL rows produced
- Random spot-check of 20 rows shows coherent (text, activation, explanation) triples
- Activations have expected statistics (non-zero, varied, not degenerate)

### Phase 3: SFT training ($150-200, 6-12 hours wall clock)

**Goal:** supervised fine-tune AR (Activation Reconstructor) then AV (Activation Verbalizer).

**Tasks:**
1. `python -m nla.scripts.prepare_critic_checkpoint` (one-time prereq)
   - Builds AR by truncating Qwen3-32B to layers 0..55, replacing final LN with Identity, adding `d_model × d_model` head
   - **CRITICAL: identity-init the value_head** (TRAINING_NOTES.md:66-71). Default kaiming makes step-0 loss 17% worse.
2. `bash configs/critic_sft_qwen3_32b.sh` — AR SFT
   - Reference (Qwen2.5-7B): 2× H100, micro_bsz=64, global_bsz=256, 1000 steps, ~50 min wall, FVE 37.5% at end
   - For 32B: scale up — expect ~2-3× the time per step, possibly need 4-8× H100
3. `bash configs/actor_sft_qwen3_32b.sh` — AV SFT
   - Reference (Qwen2.5-7B): 2× H100, micro_bsz=16, global_bsz=256, 1000 steps, ~83 min wall
   - For 32B: similar scaling

**Gate to Phase 4:**
- AR SFT FVE > 0 (any value above baseline is fine for this stage)
- AV SFT loss ~0.2 lower than random baseline
- AV outputs are English, not CJK garbage (the most common failure mode)
- AV outputs look like coherent `<analysis>...</analysis>` blocks describing text features

### Phase 4: RL training ($1500-2400, 2-3 days)

**Goal:** jointly RL-train AV (via GRPO) + AR (supervised MSE) to maximize FVE.

**Tasks:**
1. Rent 16× H100-80GB setup: 8 actor + 4 critic + 4 SGLang rollout
2. `bash configs/rl_qwen3_32b.sh`
3. Key settings (from kitft):
   - `lr_actor=1e-6`, `lr_critic=5e-5`
   - `rollout_max_response_len=150` (length-drift guard)
   - `n_samples_per_prompt=4-8` (GRPO group)
   - `global_batch=512`
   - `save_interval=100`
   - `kl_coef=0.01`
4. Monitor `train/fve_nrm` (logged per step)
5. Expected behavior: rapid initial growth, plateau at ~150 steps for 7B; for 32B, may take 300-500 steps

**Gate to Phase 5:**
- FVE plateaus (don't waste $1500 on diminishing returns past step ~500)
- Final FVE ≥ 50% (working threshold); ≥ 65% (good)
- AV outputs remain English (not collapsed to CJK)
- No NCCL deadlocks or OOM crashes

**Stop criteria (pre-registered):**
- If FVE plateaus below 40%: stop, the training data may be inadequate; iterate or accept lower quality
- If FVE plateaus 40-60%: stop at plateau, save checkpoint
- If FVE keeps growing past step 500: extend to 1000-2000 steps as budget allows

### Phase 5: Evaluation + verbalization ($50, 1-2 days)

**Goal:** use the trained NLA to characterize L55's eval-awareness signal.

**Tasks:**
1. Load trained NLA (AV checkpoint from Phase 4)
2. Apply AV to L55 activations on:
   - Microsoft triggers (~340 pairs, training-distribution sanity check)
   - ed_xx (our 10 pairs)
   - controlled holdout v2 (160 pairs)
3. Collect AV outputs (text descriptions of each activation)
4. Compare AV outputs systematically:
   - Per-pair: does AV say different things on the eval-framed vs deploy-framed arm?
   - Cross-distribution: do the eval-framed AVs share common themes across Microsoft / ed_xx / controlled?
5. Use an LLM judge (Opus 4.7) to categorize AV outputs (per the rubric_v1.md style from V4a) — does the AV consistently mention "research"/"evaluation"/"awareness" on the eval-framed prompts?

**Success criteria:**
- AV outputs on eval-framed prompts mention eval/test/research themes at meaningfully higher rate than on deploy-framed prompts (e.g., 70%+ vs 20%+)
- The themes are consistent across stimulus distributions (Microsoft / ed_xx / controlled)
- Themes are interpretable enough that a human reader would say "yes, the model is representing 'this is an eval'"

If these all hit: **publishable Stage 3 result.** L55 captures a representation of eval-awareness that's both (a) statistically detectable via probes AND (b) human-interpretable via NLA verbalizations.

If verbalizations are garbage despite high FVE: still publishable as a negative result on NLA interpretability for this specific feature, with the L55 probe finding standing.

---

## Open decisions (resolve before Phase 1)

1. **Stage 2 API model: Haiku 4.5 ($300) vs Sonnet 4.6 ($3000).**
   - Haiku: 10× cheaper. Released Qwen2.5-7B NLA used Haiku 4.5 — kitft validated this works.
   - Sonnet: arguably higher-quality explanations → potentially higher final FVE. Other released NLAs used Sonnet.
   - **Recommendation: Haiku 4.5.** $2700 savings, established to work, can always re-run Stage 2 with Sonnet if Haiku-based NLA underperforms.

2. **GPU provider:** RunPod community ($1.99/hr H100), Vast.ai ($1.50-$2.50/hr), Lambda ($2.49-$3.29/hr).
   - **Recommendation: RunPod community for SFT (cheapest, OK reliability). Lambda for multi-day RL run (most reliable, you're paying for uptime).**

3. **RL step count: 500 (lean) vs 2000 (full).**
   - 7B saturates at ~150 steps. 32B may take more — but the kitft RL config defaults to 2000 and Anthropic's note says marginal gains past step 150.
   - **Recommendation: train to 500 steps, save checkpoint every 100 steps, eval offline, extend only if FVE is still growing meaningfully at step 500.**

4. **Qwen3 thinking mode for AV decoding.** Qwen3 supports `enable_thinking=False` for non-CoT outputs. AV training is on short `<analysis>` blocks, not chain-of-thought — likely want thinking disabled.
   - **Recommendation: disable thinking for AV decoding** (test both on Phase 0 smoke run).

5. **Skip RL stage?** SFT alone produces a working NLA (FVE ~37% per 7B reference). If budget is tight, ship SFT-only.
   - **Recommendation: NO. RL is where most of the FVE quality comes from. SFT-only NLA verbalizations are notably worse.**

---

## Risk register

From kitft's TRAINING_NOTES.md and code comments. Each has a documented mitigation.

| Risk | Severity | Mitigation |
|---|---|---|
| CJK garbage outputs | HIGH | Auto-picker for injection token; if breaks, retrain SFT with different injection_scale |
| NCCL deadlock during RL | HIGH | Never enable `--gradient-checkpointing` in RL config |
| OOM from response-length drift | HIGH | Set `--rollout-max-response-len 150` (already in shipped config) |
| `/tmp` overlay-fs slowdown | MEDIUM | Set `NLA_EMBED_DUMP_DIR=/dev/shm/nla` |
| GRPO importance-ratio diagnostic stuck | MEDIUM | `del payload["input_ids"]` when sending embeds (already patched) |
| Value-head bad init | MEDIUM | Identity-init in `prepare_critic_checkpoint.py` (already in script) |
| API rate limits Stage 2 | MEDIUM | concurrency=100 default; respect retry-after; cache by `detokenized_text_truncated` hash |
| 32B-specific failure not seen in 7B | UNKNOWN | Phase 0 smoke test is the primary mitigation |
| Final FVE < 50% | UNKNOWN | Will surface during Phase 4; pre-registered stop criteria |
| AV outputs uninterpretable despite high FVE | UNKNOWN | Acceptable negative finding for paper; L55 probe result still stands |
| 97 chunked-prefill SGLang patches broken | UNKNOWN | Run `patches/apply_sglang_patches.sh`; if fails, the SGLang version moved |
| Miles upstream commit drift | UNKNOWN | Check `nla/miles_patches/UPSTREAM_PIN`; rebase if needed |

---

## File changes required (Phase 1)

**New files:**
- `configs/datagen/qwen3_32b_fineweb_100k.yaml` (~30 lines)
- `configs/datagen/qwen3_32b_10docs_test.yaml` (Phase 0 smoke test, ~30 lines)
- `configs/actor_sft_qwen3_32b.sh` (~30 lines, copy of `actor_sft.sh`)
- `configs/critic_sft_qwen3_32b.sh` (~30 lines, copy of `critic_sft.sh`)
- `configs/rl_qwen3_32b.sh` (~50 lines, copy of `rl.sh`)

**Modified files:**
- `nla/datagen/model_presets.py`: add `"qwen3_32b"` entry in `MODELS` dict
- `nla/datagen/injection_token_cache.yaml`: will auto-populate on first run

**Env vars to set:**
- `BASE_MODEL=Qwen/Qwen3-32B`
- `LOSS_MASK_TYPE=qwen3`
- `NLA_EMBED_DUMP_DIR=/dev/shm/nla`
- `ANTHROPIC_API_KEY=...` (real key)
- `HF_TOKEN=...` (real key)

---

## Things to scp back to local before each pod termination

Phase 2 (data gen): training parquets are HUGE (~50-100GB). Don't scp — leave on a persistent network volume or accept that re-generating costs ~$240+API.

Phase 3 (SFT): AR and AV checkpoints (~60GB each for 32B). scp these — needed for Phase 4 RL.

Phase 4 (RL): final NLA checkpoints (~60GB AV + ~30GB AR). scp these — these ARE the trained NLA.

Phase 5: AV outputs as JSONL — small (~10MB), scp easily.

---

## Decision points along the way

After Phase 0: continue to Phase 1, or debug?
After Phase 1: $300-3000 commitment to Phase 2, or abort?
After Phase 2: $150-200 to Phase 3, or abort?
After Phase 3: $1500-2400 to Phase 4, or accept SFT-only?
After Phase 4: how much extra RL training (if FVE still growing)?
After Phase 5: results good enough for paper? Or pivot to negative-result framing?

---

## Cumulative budget projection (lean path with Haiku)

| Phase | Cost | Cumulative |
|---|---|---|
| 0 (preflight) | $5 | $5 |
| 1 (setup) | $30 | $35 |
| 2 (data gen, Haiku) | $540 | $575 |
| 3 (SFT) | $170 | $745 |
| 4 (RL, 500 steps) | $1200 | $1945 |
| 5 (eval) | $50 | $1995 |
| **Lean total** | | **~$2,000** |

| Phase | Cost | Cumulative |
|---|---|---|
| 0 (preflight) | $5 | $5 |
| 1 (setup) | $30 | $35 |
| 2 (data gen, Sonnet) | $3240 | $3275 |
| 3 (SFT) | $200 | $3475 |
| 4 (RL, 2000 steps) | $2400 | $5875 |
| 5 (eval) | $50 | $5925 |
| **Production total** | | **~$6,000** |

---

## My honest recommendation on the budget tier

**Go lean (Haiku + 500 RL steps + RunPod community GPUs).** Total ~$2,000.

Reasons:
- Released Qwen2.5-7B NLA used Haiku 4.5 → kitft validated this works
- RL FVE saturates by ~150 steps for 7B; 500 steps gives 32B comfortable headroom
- If lean version gives garbage AV outputs, we KNOW the methodology, can re-run Phase 2 with Sonnet for ~$2700 more
- $2000 to learn whether L55's verbalization is interpretable is good cost-benefit
- $6000 production run before knowing the lean version works = bad cost-benefit

If Phase 4 lean produces a working NLA (FVE > 50% and English AV outputs): we're done.
If lean Phase 4 underperforms: targeted re-runs with Sonnet are cheaper than committing to full production from the start.

---

## What's NOT in this plan (out of scope for Stage 3)

- Training NLAs on layers other than L55 (we already have probe evidence L55 is the right layer)
- Training NLAs on other models (Qwen3-32B is our chosen target)
- Steering experiments (V5 future work, per Hua/Marks/Nanda methodology)
- Comparison to Anthropic's Claude-internal NLA (proprietary)
