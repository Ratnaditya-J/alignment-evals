"""
V4b Step D2 (RunPod): extract Qwen3-32B activations on controlled holdout.

Runs ON the pod. Reads /workspace/controlled_holdout_v1.yaml (scp'd from
v4/prompts/), loads Qwen3-32B, extracts avg-pooled prompt activations at
9 layers, saves a compressed npz.

This dataset is author-written (no third-party plaintext concerns); the
yaml is in the repo at v4/prompts/controlled_holdout_v1.yaml and the
output activations npz can also live in the repo (small ~30 MB, gitignore
as before).

Expected runtime: ~2-3 min for 80 prompts on H100 80GB. Cost: ~$0.15-0.25.

Usage on pod (after Qwen3-32B downloaded to /workspace/models/Qwen3-32B):
    cd /workspace
    python extract_controlled_pod.py 2>&1 | tee controlled_extract.log
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

LAYERS = [10, 20, 30, 40, 50, 55, 60, 62, 63]


def extract_multilayer(model, tokenizer, text: str, layer_indices: list[int]) -> dict[int, np.ndarray]:
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
    parser.add_argument("--holdout-yaml", default="/workspace/controlled_holdout_v1.yaml")
    parser.add_argument("--output-dir", default="/workspace/v4_outputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading controlled holdout from {args.holdout_yaml}")
    with open(args.holdout_yaml) as f:
        data = yaml.safe_load(f)
    pairs = data["pairs"]
    prompts = []  # list of (pair_id, arm, text, label, family, qid)
    for pair in pairs:
        for arm_letter, arm in [("a", pair["a"]), ("b", pair["b"])]:
            prompts.append({
                "pair_id": pair["pair_id"],
                "arm": arm_letter,
                "arm_label": arm["label"],
                "label": 1 if arm["label"] == "eval" else 0,
                "family": pair["family"],
                "question_id": pair["question_id"],
                "question_topic": pair["question_topic"],
                "text": arm["text"],
                "length": len(arm["text"]),
            })
    print(f"  {len(prompts)} prompts from {len(pairs)} pairs")
    print(f"  by label: {sum(1 for p in prompts if p['label']==1)} eval, "
          f"{sum(1 for p in prompts if p['label']==0)} deploy")

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

    print(f"\nExtracting activations on {len(prompts)} prompts at layers {LAYERS}...")
    X_by_layer: dict[int, list[np.ndarray]] = {l: [] for l in LAYERS}
    y_list: list[int] = []
    pair_id_list: list[str] = []
    arm_list: list[str] = []
    family_list: list[str] = []
    qid_list: list[int] = []
    length_list: list[int] = []
    t0 = time.time()
    for i, p in enumerate(prompts):
        acts = extract_multilayer(model, tokenizer, p["text"], LAYERS)
        for l in LAYERS:
            X_by_layer[l].append(acts[l])
        y_list.append(p["label"])
        pair_id_list.append(p["pair_id"])
        arm_list.append(p["arm"])
        family_list.append(p["family"])
        qid_list.append(p["question_id"])
        length_list.append(p["length"])
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(prompts)}  elapsed={elapsed:.0f}s")
    X_arrays = {f"X_L{l}": np.stack(X_by_layer[l]).astype(np.float32) for l in LAYERS}
    y = np.array(y_list, dtype=np.int32)
    lengths = np.array(length_list, dtype=np.int32)
    qids = np.array(qid_list, dtype=np.int32)
    print(f"  Done in {time.time()-t0:.0f}s")

    npz_path = output_dir / "controlled_activations.npz"
    np.savez_compressed(
        npz_path,
        y=y,
        lengths=lengths,
        qids=qids,
        **X_arrays,
        pair_ids=np.array(pair_id_list, dtype="S16"),
        arms=np.array(arm_list, dtype="S2"),
        families=np.array(family_list, dtype="S4"),
    )
    print(f"\nSaved activations: {npz_path}")

    meta = {
        "model_path": args.model_path,
        "n_layers": int(n_layers),
        "d_model": int(d_model),
        "layers_extracted": LAYERS,
        "n_prompts": int(len(prompts)),
        "n_pairs": int(len(pairs)),
        "n_eval": int((y == 1).sum()),
        "n_deploy": int((y == 0).sum()),
        "holdout_yaml_sha256": _sha256_file(Path(args.holdout_yaml)),
        "holdout_yaml_path": args.holdout_yaml,
        "npz_path": str(npz_path),
        "extraction_seconds": float(time.time() - t0),
    }
    meta_path = output_dir / "controlled_extract_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {meta_path}")
    print(f"\nNext: scp {npz_path} + {meta_path} back for Step E2 scoring.")


if __name__ == "__main__":
    main()
