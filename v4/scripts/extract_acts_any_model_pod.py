"""
Generic activation extraction for ANY HF causal LM. Replaces the
Qwen3-specific extract_controlled_pod.py and extract_sad_activations_pod.py
patterns with a model-agnostic, input-agnostic script.

Supports:
  - Any HuggingFace causal LM that exposes output_hidden_states (Qwen3-32B,
    gpt-oss-120b, Llama-3.3-70B, etc.)
  - Two input formats:
      * --input-yaml: our pairs YAML (controlled_holdout_v*.yaml format)
      * --input-triggers: Microsoft triggers.json format
  - Configurable layer set via --layers (default: full sweep 10..63 every 5,
    plus 55/62/63; you can pass any subset)
  - Optional 4-bit quantization for large models that don't fit otherwise
    (--load-in-4bit; uses bitsandbytes)

Output: <output-dir>/<model-tag>_activations.npz with:
  - X_L{layer}: (n_prompts, d_model) float32 activations per layer
  - y: (n_prompts,) int32 labels
  - pair_ids, families, etc. as metadata arrays

Usage examples:

  # Qwen3-32B on controlled holdout
  python extract_acts_any_model_pod.py \\
      --model-path Qwen/Qwen3-32B --model-tag qwen3_32b \\
      --input-yaml controlled_holdout_v2.yaml --layers 55 \\
      --output-dir /workspace/v4_outputs

  # gpt-oss-120b on Microsoft triggers (for probe training data)
  python extract_acts_any_model_pod.py \\
      --model-path openai/gpt-oss-120b --model-tag gpt_oss_120b \\
      --input-triggers /workspace/Test_Awareness_Steering/data/triggers/triggers.json \\
      --layers 8 16 24 28 30 32 --load-in-4bit \\
      --output-dir /workspace/v4_outputs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_multilayer(model, tokenizer, text: str, layer_indices: list[int]) -> dict[int, np.ndarray]:
    """One forward pass; return {layer: avg-pooled activation over prompt tokens}."""
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
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    return {
        l: outputs.hidden_states[l + 1][0].mean(dim=0).to(torch.float32).cpu().numpy()
        for l in layer_indices
    }


def load_yaml_pairs(yaml_path: Path) -> list[dict]:
    """Load our pairs YAML format; return list of prompt records."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    out = []
    for pair in data["pairs"]:
        for arm_letter, arm in [("a", pair["a"]), ("b", pair["b"])]:
            out.append({
                "text": arm["text"],
                "label": 1 if arm["label"] == "eval" else 0,
                "pair_id": pair["pair_id"],
                "arm": arm_letter,
                "arm_label": arm["label"],
                "family": pair.get("family", ""),
                "question_id": pair.get("question_id", -1),
                "length": len(arm["text"]),
            })
    return out


def load_triggers_json(triggers_path: Path) -> list[dict]:
    """Load Microsoft triggers.json format."""
    with open(triggers_path) as f:
        triggers = json.load(f)
    out = []
    for i, pair in enumerate(triggers):
        if "real" in pair:
            out.append({
                "text": pair["real"], "label": 0,
                "pair_id": f"ms_{i:04d}", "arm": "real", "arm_label": "real",
                "family": "ms_triggers", "question_id": i,
                "length": len(pair["real"]),
            })
        if "hypothetical" in pair:
            out.append({
                "text": pair["hypothetical"], "label": 1,
                "pair_id": f"ms_{i:04d}", "arm": "hypothetical", "arm_label": "hypothetical",
                "family": "ms_triggers", "question_id": i,
                "length": len(pair["hypothetical"]),
            })
    return out


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True,
                        help="HF model id or local path, e.g. 'Qwen/Qwen3-32B' or 'openai/gpt-oss-120b'")
    parser.add_argument("--model-tag", required=True,
                        help="Short tag for output filenames, e.g. 'qwen3_32b' or 'gpt_oss_120b'")
    parser.add_argument("--input-yaml", type=Path, default=None,
                        help="Pairs YAML (controlled_holdout_v*.yaml format)")
    parser.add_argument("--input-triggers", type=Path, default=None,
                        help="Microsoft triggers.json")
    parser.add_argument("--layers", type=int, nargs="+", required=True,
                        help="Layer indices to extract, e.g. --layers 10 20 30 40 50 55 60 62 63")
    parser.add_argument("--output-dir", default="/workspace/v4_outputs")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Quantize model to int4 via bitsandbytes (use for 120B+ models)")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N prompts (for smoke testing)")
    parser.add_argument("--model-hf-id", default=None,
                        help="Canonical HF id (e.g. 'openai/gpt-oss-120b') for provenance "
                             "when --model-path is a local directory")
    args = parser.parse_args()

    if not (args.input_yaml or args.input_triggers):
        raise SystemExit("Provide either --input-yaml or --input-triggers")
    if args.input_yaml and args.input_triggers:
        raise SystemExit("Provide only one of --input-yaml or --input-triggers")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_yaml:
        print(f"Loading pairs YAML from {args.input_yaml}")
        prompts = load_yaml_pairs(args.input_yaml)
        input_hash = _sha256_file(args.input_yaml)
        input_kind = "yaml-pairs"
        input_path = str(args.input_yaml)
    else:
        print(f"Loading Microsoft triggers from {args.input_triggers}")
        prompts = load_triggers_json(args.input_triggers)
        input_hash = _sha256_file(args.input_triggers)
        input_kind = "triggers-json"
        input_path = str(args.input_triggers)
    if args.limit:
        prompts = prompts[: args.limit]
        print(f"  Limited to first {args.limit} prompts")
    print(f"  {len(prompts)} prompts, labels {np.bincount([p['label'] for p in prompts]).tolist()}")

    print(f"\nLoading model: {args.model_path}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    model_kwargs = dict(torch_dtype=dtype, device_map="auto", trust_remote_code=True)
    if args.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            model_kwargs.pop("torch_dtype")  # incompatible with 4bit
        except ImportError:
            raise SystemExit("--load-in-4bit requires bitsandbytes: pip install bitsandbytes")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.eval()
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    print(f"  Loaded in {time.time()-t0:.1f}s. n_layers={n_layers}, d_model={d_model}")
    print(f"  Model class: {type(model).__name__}")
    invalid_layers = [l for l in args.layers if l >= n_layers]
    if invalid_layers:
        raise SystemExit(f"Requested layers {invalid_layers} exceed n_layers={n_layers}")

    suffix = "controlled" if args.input_yaml else "triggers"
    partial_path = output_dir / f"{args.model_tag}_{suffix}_partial.npz"

    print(f"\nExtracting activations at layers {args.layers} on {len(prompts)} prompts...")
    X_by_layer: dict[int, list[np.ndarray]] = {l: [] for l in args.layers}
    y_list, pair_id_list, arm_list, family_list, qid_list, length_list = [], [], [], [], [], []
    start_idx = 0

    if partial_path.exists():
        print(f"  Found checkpoint at {partial_path}, attempting resume...")
        partial = np.load(partial_path)
        layer_keys_ok = all(f"X_L{l}" in partial.files for l in args.layers)
        if not layer_keys_ok:
            print(f"  WARNING: checkpoint missing some layer arrays; starting fresh")
        else:
            for l in args.layers:
                for vec in partial[f"X_L{l}"]:
                    X_by_layer[l].append(vec)
            y_list = list(partial["y"])
            pair_id_list = [p.decode() for p in partial["pair_ids"]]
            arm_list = [a.decode() for a in partial["arms"]]
            family_list = [f.decode() for f in partial["families"]]
            qid_list = list(partial["qids"])
            length_list = list(partial["lengths"])
            start_idx = int(partial["completed_idx"])
            print(f"  Resumed from prompt {start_idx} ({len(y_list)} entries loaded)")

    def save_checkpoint(completed: int) -> None:
        Xc = {f"X_L{l}": np.stack(X_by_layer[l]).astype(np.float32) for l in args.layers}
        np.savez_compressed(
            partial_path,
            **Xc,
            y=np.array(y_list, dtype=np.int32),
            lengths=np.array(length_list, dtype=np.int32),
            qids=np.array(qid_list, dtype=np.int32),
            pair_ids=np.array(pair_id_list, dtype="S32"),
            arms=np.array(arm_list, dtype="S16"),
            families=np.array(family_list, dtype="S16"),
            completed_idx=np.int32(completed),
        )

    t0 = time.time()
    for i, p in enumerate(prompts):
        if i < start_idx:
            continue
        try:
            acts = extract_multilayer(model, tokenizer, p["text"], args.layers)
        except Exception as e:
            print(f"  ERROR on prompt {i}: {e}")
            continue
        for l in args.layers:
            X_by_layer[l].append(acts[l])
        y_list.append(p["label"])
        pair_id_list.append(p["pair_id"])
        arm_list.append(p["arm"])
        family_list.append(p["family"])
        qid_list.append(p["question_id"])
        length_list.append(p["length"])
        done = i + 1
        if done % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / max(done - start_idx, 1) * (len(prompts) - done)
            print(f"  {done}/{len(prompts)}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")
        if done % 100 == 0:
            save_checkpoint(done)
            print(f"    [checkpoint saved at {done}]")
    X_arrays = {f"X_L{l}": np.stack(X_by_layer[l]).astype(np.float32) for l in args.layers}
    y = np.array(y_list, dtype=np.int32)
    lengths = np.array(length_list, dtype=np.int32)
    qids = np.array(qid_list, dtype=np.int32)
    print(f"  Done in {time.time()-t0:.0f}s, {len(y)} succeeded")

    npz_path = output_dir / f"{args.model_tag}_{suffix}_activations.npz"
    np.savez_compressed(
        npz_path, y=y, lengths=lengths, qids=qids,
        **X_arrays,
        pair_ids=np.array(pair_id_list, dtype="S32"),
        arms=np.array(arm_list, dtype="S16"),
        families=np.array(family_list, dtype="S16"),
    )
    print(f"\nSaved: {npz_path}")
    if partial_path.exists():
        partial_path.unlink()
        print(f"Cleaned up checkpoint: {partial_path}")

    config_path = Path(args.model_path) / "config.json"
    model_config_sha256 = _sha256_file(config_path) if config_path.is_file() else None
    meta = {
        "model_path": args.model_path,
        "model_hf_id": args.model_hf_id,
        "model_tag": args.model_tag,
        "model_class": type(model).__name__,
        "model_config_sha256": model_config_sha256,
        "n_layers": int(n_layers),
        "d_model": int(d_model),
        "layers_extracted": args.layers,
        "load_in_4bit": args.load_in_4bit,
        "dtype": args.dtype,
        "n_prompts": int(len(y)),
        "n_eval": int((y == 1).sum()),
        "n_deploy": int((y == 0).sum()),
        "input_kind": input_kind,
        "input_path": input_path,
        "input_sha256": input_hash,
        "npz_path": str(npz_path),
    }
    meta_path = output_dir / f"{args.model_tag}_{suffix}_extract_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
