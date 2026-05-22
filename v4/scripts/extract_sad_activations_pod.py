"""
V4b Step D (RunPod): extract Qwen3-32B activations on deduplicated SAD prompts.

Runs ON the pod. Reads /workspace/sad_oversight_dedup.json (which you scp from
~/.cache/alignment-evals-local/sad_oversight_dedup.json on your laptop -- per
DATA_POLICY.md, SAD plaintext lives outside the repo). Loads Qwen3-32B,
extracts avg-pooled prompt activations at all 9 layers
(L10/20/30/40/50/55/60/62/63), saves a compressed npz.

Expected runtime: ~5-10 min for 400 prompts on H100 80GB (no training, just
forward passes). Cost: ~$0.30-0.50.

Outputs (on pod, /workspace/v4_outputs/):
  sad_activations.npz  -- arrays: X_L{l} for each layer, labels, lengths, ids
  sad_extract_meta.json -- metadata: prompt counts, model, timings, hashes

After running:
  scp the npz back to local for Step E (scoring with frozen probes).
  The npz contains activations derived from SAD prompts. Per DATA_POLICY.md,
  it inherits SAD's plaintext constraints -- keep local, gitignore.

Usage on pod (after Qwen3-32B is downloaded to /workspace/models/Qwen3-32B):
    cd /workspace
    python extract_sad_activations_pod.py 2>&1 | tee sad_extract.log
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LAYERS = [10, 20, 30, 40, 50, 55, 60, 62, 63]


def extract_multilayer(model, tokenizer, text: str, layer_indices: list[int]) -> dict[int, np.ndarray]:
    """One forward pass, return dict {layer: avg-pooled activation (float32)}."""
    messages = [{"role": "user", "content": text}]
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True,
    ).to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    return {
        l: outputs.hidden_states[l + 1][0].mean(dim=0).to(torch.float32).cpu().numpy()
        for l in layer_indices
    }


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/workspace/models/Qwen3-32B")
    parser.add_argument("--dedup-json", default="/workspace/sad_oversight_dedup.json")
    parser.add_argument("--output-dir", default="/workspace/v4_outputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load deduped SAD prompts
    print(f"Loading deduped SAD from {args.dedup_json}")
    with open(args.dedup_json) as f:
        dedup_data = json.load(f)
    kept = dedup_data["kept"]
    print(f"  {len(kept)} kept prompts")
    print(f"  by label: {sum(1 for k in kept if k['label']==1)} test, "
          f"{sum(1 for k in kept if k['label']==0)} deploy")

    # Load model
    print(f"\nLoading Qwen3-32B from {args.model_path}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    print(f"  Loaded in {time.time()-t0:.1f}s. n_layers={n_layers}, d_model={d_model}")
    assert max(LAYERS) < n_layers, f"Max layer {max(LAYERS)} >= n_layers {n_layers}"

    # Extract activations
    print(f"\nExtracting activations on {len(kept)} SAD prompts at layers {LAYERS}...")
    X_by_layer: dict[int, list[np.ndarray]] = {l: [] for l in LAYERS}
    y_list: list[int] = []
    id_list: list[str] = []
    source_list: list[str] = []
    length_list: list[int] = []
    t0 = time.time()
    for i, item in enumerate(kept):
        acts = extract_multilayer(model, tokenizer, item["body"], LAYERS)
        for l in LAYERS:
            X_by_layer[l].append(acts[l])
        y_list.append(item["label"])
        id_list.append(item["sad_id"])
        source_list.append(item["source"])
        length_list.append(item["length"])
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(kept) - i - 1)
            print(f"  {i+1}/{len(kept)}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")
    X_arrays = {f"X_L{l}": np.stack(X_by_layer[l]).astype(np.float32) for l in LAYERS}
    y = np.array(y_list, dtype=np.int32)
    lengths = np.array(length_list, dtype=np.int32)
    print(f"  Done in {time.time()-t0:.0f}s")

    # Save npz
    npz_path = output_dir / "sad_activations.npz"
    # Note: sad_ids are 75 chars (MCQuestion- prefix + 64-hex-sha). Use S128
    # to avoid truncation. (Earlier S64 silently truncated; scoring script
    # works around it by ordering against metadata.)
    np.savez_compressed(
        npz_path,
        y=y,
        lengths=lengths,
        **X_arrays,
        ids=np.array(id_list, dtype="S128"),
        sources=np.array(source_list, dtype="S32"),
    )
    print(f"\nSaved activations: {npz_path}")
    print(f"  size: ~{sum(X_arrays[k].nbytes for k in X_arrays)/(1024*1024):.0f} MB uncompressed")

    # Metadata
    meta = {
        "model_path": args.model_path,
        "n_layers": int(n_layers),
        "d_model": int(d_model),
        "layers_extracted": LAYERS,
        "n_prompts": int(len(kept)),
        "n_test": int((y == 1).sum()),
        "n_deploy": int((y == 0).sum()),
        "dedup_json_sha256": _sha256_file(Path(args.dedup_json)),
        "dedup_json_path": args.dedup_json,
        "npz_path": str(npz_path),
        "extraction_seconds": float(time.time() - t0),
    }
    meta_path = output_dir / "sad_extract_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {meta_path}")
    print(f"\nNext: scp {npz_path} + {meta_path} back to laptop for Step E (scoring).")


if __name__ == "__main__":
    main()
