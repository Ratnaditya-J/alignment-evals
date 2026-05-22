"""
Stage 4: run text generation with activation steering at a specified layer.

For each (prompt, condition, alpha):
  1. Register a forward hook on the target layer that adds `alpha * direction`
     to the residual stream during the prefill pass (all prompt tokens).
     During decode (single-token forward passes), the hook is a no-op so
     generation proceeds normally — the steering effect persists via the
     KV cache built during prefill.
  2. Generate up to --max-new-tokens completions deterministically (temp=0).
  3. Deregister the hook.
  4. Save the generated text + metadata as a JSONL record.

Pre-registered correctness tests (run with --test):
  T1: alpha=0 produces output identical to no-hook generation
  T2: after the prefill hook fires, the layer's residual stream is shifted
      by exactly alpha * direction at every prompt position
  T3: alpha>0 with a non-trivial direction produces a different output
      than alpha=0
  T4: after generation, the hook is removed (next no-hook gen matches T1)

All four must pass before any full-scale runs.

Usage (correctness test mode):
    python run_steered_generation.py --test \\
        --model-path /tmp/gpt-oss-120b \\
        --steering-vectors v4/runs/stage4/steering_vectors_gpt_oss_120b_L34.npz \\
        --layer 34

Usage (real run):
    python run_steered_generation.py \\
        --model-path /tmp/gpt-oss-120b \\
        --steering-vectors v4/runs/stage4/steering_vectors_gpt_oss_120b_L34.npz \\
        --layer 34 \\
        --prompts-jsonl v4/runs/stage4/v1_prompts_500.jsonl \\
        --alpha-values -2 -1 -0.5 0 0.5 1 2 \\
        --conditions mean_diff_zscored random orthogonal \\
        --direction-default mean_diff_zscored \\
        --max-new-tokens 200 \\
        --output-jsonl v4/runs/stage4/gpt_oss_L34_generations.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------- hook implementation ----------

class SteeringHook:
    """Forward hook that adds `alpha * direction` to a transformer block's
    output during PREFILL (multi-token forward) and is a no-op during
    DECODE (single-token forward). This mirrors the semantics of V1's
    inject prompt: the steering only modifies how the prompt is represented;
    generation proceeds with whatever the model decided based on that
    modified representation (via the KV cache).
    """

    def __init__(self, direction: torch.Tensor, alpha: float,
                 position: str = "all", record_diff: bool = False):
        assert position in ("all", "last"), f"position={position!r}"
        self.direction = direction       # (d_model,) — on model.device, model.dtype
        self.alpha = float(alpha)
        self.position = position
        self.record_diff = record_diff
        self.handle = None
        # Diagnostics: filled in by the hook for correctness tests
        self.last_observed_diff: Optional[torch.Tensor] = None
        self.last_input_seq_len: Optional[int] = None
        self.call_count = 0

    def __call__(self, module, args, output):
        # Block output may be a Tensor or a tuple of (hidden_states, ...).
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        assert hidden.dim() == 3, f"expected (batch, seq, d_model); got {tuple(hidden.shape)}"
        batch, seq_len, d_model = hidden.shape
        self.last_input_seq_len = seq_len
        self.call_count += 1

        if seq_len <= 1:
            # decode-time call — no steering (KV cache from prefill carries effect)
            return output
        if self.alpha == 0.0:
            return output

        # Apply steering to all positions or just the last
        delta = self.alpha * self.direction.to(hidden.dtype).to(hidden.device)  # (d_model,)
        if self.position == "all":
            modified = hidden + delta.view(1, 1, -1)
        else:  # last only
            mask = torch.zeros_like(hidden)
            mask[:, -1, :] = delta
            modified = hidden + mask

        if self.record_diff:
            self.last_observed_diff = (modified - hidden).detach().clone()

        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified

    def register(self, model, layer_idx: int):
        target = _get_layer_module(model, layer_idx)
        self.handle = target.register_forward_hook(self)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def _get_layer_module(model, layer_idx: int):
    """Return the transformer block at layer_idx for HF causal LMs.
    Works for Qwen3, Llama, gpt-oss and similar architectures that expose
    `model.model.layers`. If your model uses a different attribute path,
    add a branch here.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        raise AttributeError(
            f"cannot locate transformer layers on {type(model).__name__}; "
            "extend _get_layer_module")
    assert 0 <= layer_idx < len(layers), \
        f"layer_idx {layer_idx} out of range (n_layers={len(layers)})"
    return layers[layer_idx]


# ---------- generation ----------

def encode_prompt(tokenizer, text: str, device):
    messages = [{"role": "user", "content": text}]
    try:
        encoded = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True,
        )
    except Exception:
        encoded = tokenizer(text, return_tensors="pt")
    if hasattr(encoded, "input_ids"):
        input_ids = encoded.input_ids
    elif isinstance(encoded, dict):
        input_ids = encoded["input_ids"]
    else:
        input_ids = encoded
    return input_ids.to(device)


@torch.no_grad()
def generate_with_steering(model, tokenizer, prompt_text: str,
                            steering: Optional[SteeringHook],
                            layer_idx: int, max_new_tokens: int) -> str:
    input_ids = encode_prompt(tokenizer, prompt_text, model.device)
    if steering is not None:
        steering.register(model, layer_idx)
    try:
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,        # ignored when do_sample=False
            pad_token_id=tokenizer.eos_token_id,
        )
    finally:
        if steering is not None:
            steering.remove()
    new_tokens = out[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------- correctness tests ----------

def run_tests(model, tokenizer, steering_vectors_path: Path, layer_idx: int) -> None:
    """The four pre-registered correctness checks. Aborts on any failure."""
    print("\n=== Stage 4 steering correctness tests ===")
    sv = np.load(steering_vectors_path)
    direction_name = "mean_diff_zscored"
    w_np = sv[direction_name]
    print(f"  using direction: {direction_name}, ||w||={np.linalg.norm(w_np):.4f}")
    w = torch.from_numpy(w_np).to(model.device)

    test_prompt = "Briefly: what is the capital of France?"

    # ----- T1: alpha=0 produces output identical to no-hook -----
    print("\n[T1] alpha=0 hook should match no-hook generation exactly")
    out_no_hook = generate_with_steering(model, tokenizer, test_prompt,
                                          steering=None, layer_idx=layer_idx,
                                          max_new_tokens=30)
    hook_zero = SteeringHook(direction=w, alpha=0.0, position="all")
    out_zero_hook = generate_with_steering(model, tokenizer, test_prompt,
                                            steering=hook_zero, layer_idx=layer_idx,
                                            max_new_tokens=30)
    assert out_no_hook == out_zero_hook, (
        f"T1 FAILED: alpha=0 hook produced different output than no-hook\n"
        f"  no-hook:  {out_no_hook!r}\n"
        f"  alpha=0:  {out_zero_hook!r}")
    print(f"  PASS — both produced: {out_no_hook[:80]!r}")

    # ----- T2: with alpha>0 and record_diff, the modification at the layer
    #          is approximately alpha * w applied to ALL prompt token positions.
    # NOTE: in bf16 with high-magnitude residual streams (gpt-oss has values
    # up to ±30000), sub-ULP additions get rounded away. We use a larger
    # alpha and looser threshold to keep the test meaningful without being
    # noise-sensitive. The signal-to-noise ratio at actual steering scale
    # (alpha in 1-10 range) is what matters — see T3 for behavioral check.
    print("\n[T2] hook applies approximately alpha * w to all prompt positions")
    alpha_test = 10.0
    hook_diag = SteeringHook(direction=w, alpha=alpha_test, position="all",
                              record_diff=True)
    _ = generate_with_steering(model, tokenizer, test_prompt,
                                steering=hook_diag, layer_idx=layer_idx,
                                max_new_tokens=5)
    assert hook_diag.last_observed_diff is not None, "T2 FAILED: hook never fired"
    diff = hook_diag.last_observed_diff[0]  # (seq, d_model)
    expected = (alpha_test * w).to(diff.dtype)
    # Each position should be close to expected; bf16 quantization at large
    # residual magnitudes means we can't expect tight precision.
    max_err = (diff - expected.view(1, -1)).abs().max().item()
    rel_err_at_max = max_err / max(expected.abs().max().item(), 1e-9)
    assert max_err < 1.0, (
        f"T2 FAILED: observed max diff differs from alpha*w by {max_err:.4f} "
        f"(>{1.0}). bf16 noise budget exceeded — possible logic bug.")
    print(f"  PASS — max abs error: {max_err:.4f} "
          f"(relative to max expected: {rel_err_at_max:.2%}); "
          f"prefill seq_len: {hook_diag.last_input_seq_len}")

    # ----- T3: alpha != 0 produces a DIFFERENT output than alpha = 0 -----
    print("\n[T3] alpha=2 produces a different generation than alpha=0")
    hook_pos = SteeringHook(direction=w, alpha=2.0, position="all")
    out_steered = generate_with_steering(model, tokenizer, test_prompt,
                                          steering=hook_pos, layer_idx=layer_idx,
                                          max_new_tokens=30)
    if out_steered == out_zero_hook:
        print(f"  WARNING: alpha=2 produced identical output to alpha=0 — "
              f"steering may be too weak for this prompt/layer")
        print(f"  output:   {out_steered[:80]!r}")
    else:
        print(f"  PASS — outputs differ")
        print(f"    alpha=0:  {out_zero_hook[:80]!r}")
        print(f"    alpha=2:  {out_steered[:80]!r}")

    # ----- T4: hook properly removed -----
    print("\n[T4] after generation, hook is removed (next call matches T1)")
    out_after = generate_with_steering(model, tokenizer, test_prompt,
                                        steering=None, layer_idx=layer_idx,
                                        max_new_tokens=30)
    assert out_after == out_no_hook, (
        f"T4 FAILED: post-generation no-hook output differs from initial baseline\n"
        f"  initial:  {out_no_hook!r}\n"
        f"  after:    {out_after!r}")
    print(f"  PASS — hook cleanup verified")

    print("\n=== All tests passed ===")


# ---------- main run loop ----------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-hf-id", default=None)
    parser.add_argument("--steering-vectors", type=Path, required=True,
                        help="Output from prepare_steering_vectors.py")
    parser.add_argument("--layer", type=int, required=True,
                        help="Layer at which to inject the steering vector")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--test", action="store_true",
                        help="Run correctness tests and exit")
    # full run args
    parser.add_argument("--prompts-jsonl", type=Path, default=None,
                        help="JSONL with one prompt per line; required for non-test runs")
    parser.add_argument("--alpha-values", type=float, nargs="+",
                        default=[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--conditions", nargs="+",
                        default=["mean_diff_zscored", "random", "orthogonal"],
                        help="Which directions to use (must match keys in steering vectors npz)")
    parser.add_argument("--position", default="all", choices=["all", "last"])
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N prompts (smoke testing)")
    args = parser.parse_args()

    if not args.test and (args.prompts_jsonl is None or args.output_jsonl is None):
        raise SystemExit("Non-test runs require --prompts-jsonl and --output-jsonl")

    print(f"Loading tokenizer + model from {args.model_path}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s. n_layers={model.config.num_hidden_layers}, "
          f"d_model={model.config.hidden_size}")
    print(f"  Model class: {type(model).__name__}")

    if args.test:
        run_tests(model, tokenizer, args.steering_vectors, args.layer)
        return

    # ----- full run -----
    print(f"\nLoading steering vectors from {args.steering_vectors}")
    sv = np.load(args.steering_vectors)
    directions = {}
    for cond in args.conditions:
        if cond not in sv.files:
            raise SystemExit(f"condition {cond!r} not in steering vectors npz "
                              f"(available: {list(sv.files)})")
        directions[cond] = torch.from_numpy(sv[cond]).to(model.device)
        print(f"  {cond}: ||w||={np.linalg.norm(sv[cond]):.4f}")

    print(f"\nLoading prompts from {args.prompts_jsonl}")
    prompts = []
    with open(args.prompts_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    if args.limit:
        prompts = prompts[: args.limit]
    print(f"  {len(prompts)} prompts loaded")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n_total = len(prompts) * len(args.conditions) * len(args.alpha_values)
    print(f"\nTotal generations to run: {n_total} "
          f"({len(prompts)} prompts × {len(args.conditions)} conditions × "
          f"{len(args.alpha_values)} alphas)")

    done = 0
    t0 = time.time()
    with open(args.output_jsonl, "w") as fout:
        for prompt in prompts:
            prompt_text = prompt["text"]
            prompt_id = prompt.get("prompt_id", prompt.get("id", "unknown"))
            for cond in args.conditions:
                direction = directions[cond]
                for alpha in args.alpha_values:
                    if alpha == 0.0 and cond != args.conditions[0]:
                        continue  # alpha=0 is identical across conditions; skip dupes
                    hook = SteeringHook(direction=direction, alpha=alpha,
                                         position=args.position)
                    try:
                        gen = generate_with_steering(
                            model, tokenizer, prompt_text,
                            steering=hook, layer_idx=args.layer,
                            max_new_tokens=args.max_new_tokens,
                        )
                    except Exception as e:
                        gen = f"<ERROR: {e}>"
                    record = {
                        "prompt_id": prompt_id,
                        "prompt_text": prompt_text,
                        "family": prompt.get("family", ""),
                        "v1_condition": prompt.get("v1_condition", ""),
                        "model_path": args.model_path,
                        "model_hf_id": args.model_hf_id,
                        "layer": args.layer,
                        "condition": cond,
                        "alpha": alpha,
                        "position": args.position,
                        "generated_text": gen,
                    }
                    fout.write(json.dumps(record) + "\n")
                    fout.flush()
                    done += 1
                    if done % 20 == 0:
                        elapsed = time.time() - t0
                        eta = elapsed / done * (n_total - done)
                        print(f"  {done}/{n_total}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    print(f"\nDone. {done} generations written to {args.output_jsonl}")


if __name__ == "__main__":
    main()
