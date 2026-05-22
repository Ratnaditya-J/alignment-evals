"""
Pre-compute all Stage 4 steering vectors from existing V4b artifacts.

Produces 5 normalized directions per (model, layer) for use by the
steering generation script:

  1. probe_coef         — trained LR probe weights, normalized to unit length
  2. probe_coef_zscored — same, but computed after per-dim z-scoring
  3. mean_diff          — μ(eval) − μ(deploy) over training activations, normalized
  4. mean_diff_zscored  — same, in z-scored activation space
  5. random             — N(0, I) sample, normalized; fixed seed for reproducibility
  6. orthogonal         — random direction orthogonalized against the probe direction

Also saves per-dim z-scoring statistics (mean, std) so the generation
script can apply them consistently.

Usage:
    python prepare_steering_vectors.py \\
        --triggers-acts v4/runs/qwen3_stage2/qwen3_activations.npz \\
        --probes v4/runs/qwen3_stage2/frozen_probes_qwen3.npz \\
        --layer 55 \\
        --model-tag qwen3_32b \\
        --output v4/runs/stage4/steering_vectors_qwen3_32b_L55.npz \\
        --seed 0

    python prepare_steering_vectors.py \\
        --triggers-acts v4/runs/gpt_oss_stage2/gpt_oss_120b_triggers_activations.npz \\
        --probes v4/runs/gpt_oss_stage2/frozen_probes_gpt_oss_120b.npz \\
        --layer 34 \\
        --model-tag gpt_oss_120b \\
        --output v4/runs/stage4/steering_vectors_gpt_oss_120b_L34.npz \\
        --seed 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize a vector. Returns zero if input is degenerate."""
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triggers-acts", type=Path, required=True,
                        help="Path to *_triggers_activations.npz (probe training data)")
    parser.add_argument("--probes", type=Path, required=True,
                        help="Path to frozen_probes_*.npz")
    parser.add_argument("--layer", type=int, required=True,
                        help="Layer index for which to compute steering vectors")
    parser.add_argument("--model-tag", required=True,
                        help="Short tag, e.g. 'qwen3_32b' or 'gpt_oss_120b'")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output .npz path")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for random direction reproducibility")
    args = parser.parse_args()

    print(f"Loading triggers activations from {args.triggers_acts}")
    acts = np.load(args.triggers_acts)
    layer_key = f"X_L{args.layer}"
    if layer_key in acts.files:
        # Generic-script schema: flat X_L<layer> + y
        X = acts[layer_key]
        y = acts["y"]
        print(f"  schema: flat (X_L{args.layer})")
    elif f"train_{layer_key}" in acts.files:
        # Older Qwen3 schema: train_X_L<layer>/test_X_L<layer> with separate y
        X = np.concatenate(
            [acts[f"train_X_L{args.layer}"], acts[f"test_X_L{args.layer}"]], axis=0)
        y = np.concatenate([acts["train_y"], acts["test_y"]], axis=0)
        print(f"  schema: split (train_X_L{args.layer} + test_X_L{args.layer} concatenated)")
    else:
        raise SystemExit(
            f"Cannot find layer {args.layer} in {args.triggers_acts}. "
            f"Available keys: {sorted(acts.files)[:20]}")
    assert X.shape[0] == y.shape[0], "label/activation count mismatch"
    n_train, d_model = X.shape
    n_eval, n_deploy = int((y == 1).sum()), int((y == 0).sum())
    print(f"  n_train={n_train}, d_model={d_model}, eval={n_eval}, deploy={n_deploy}")

    print(f"Loading probes from {args.probes}")
    probes = np.load(args.probes)
    w_probe_raw = probes[f"coef_L{args.layer}"].astype(np.float64)   # (d_model,)
    print(f"  probe coefficient L{args.layer}: shape={w_probe_raw.shape}, "
          f"||w||={np.linalg.norm(w_probe_raw):.4f}")

    # Per-dim z-scoring stats (computed on triggers training data)
    mu = X.mean(axis=0).astype(np.float64)
    sigma = X.std(axis=0).astype(np.float64)
    # Guard: dims with zero variance get sigma=1 (no rescaling)
    sigma_safe = np.where(sigma < 1e-12, 1.0, sigma)
    print(f"  z-score stats: mu_range=[{mu.min():.3f}, {mu.max():.3f}], "
          f"sigma_range=[{sigma.min():.4f}, {sigma.max():.4f}]")
    n_low_var = int((sigma < 1e-12).sum())
    if n_low_var > 0:
        print(f"  WARN: {n_low_var} dims with near-zero variance (treated as sigma=1)")

    # === 1. probe_coef (raw, normalized) ===
    probe_coef = normalize(w_probe_raw)

    # === 2. probe_coef_zscored ===
    # The probe was trained on raw X. To get its equivalent direction in
    # z-scored space (X' = (X - mu) / sigma), we use: w' = w * sigma
    # because (X' · w') = ((X - mu) / sigma) · (w * sigma) = (X - mu) · w
    # (constant offset doesn't change ranking; dropping it).
    probe_coef_zscored = normalize(w_probe_raw * sigma_safe)

    # === 3. mean_diff (raw, normalized) ===
    mu_eval = X[y == 1].mean(axis=0).astype(np.float64)
    mu_deploy = X[y == 0].mean(axis=0).astype(np.float64)
    mean_diff = normalize(mu_eval - mu_deploy)

    # === 4. mean_diff_zscored ===
    # Compute in z-scored space then convert back.
    X_z = (X - mu) / sigma_safe
    mu_eval_z = X_z[y == 1].mean(axis=0)
    mu_deploy_z = X_z[y == 0].mean(axis=0)
    mean_diff_z = normalize(mu_eval_z - mu_deploy_z)
    # Convert from z-scored space to raw activation space: w_raw = w_z / sigma
    # (then re-normalize).
    mean_diff_zscored = normalize(mean_diff_z / sigma_safe)

    # === 5. random (deterministic via seed) ===
    rng = np.random.default_rng(args.seed)
    random_dir = normalize(rng.standard_normal(d_model))

    # === 6. orthogonal — random direction with the probe component removed ===
    # Default reference for orthogonalization: mean_diff_zscored (our default
    # reporting direction). Could also do this against probe_coef; both are fine.
    ref = mean_diff_zscored
    rand2 = rng.standard_normal(d_model)
    orth_raw = rand2 - np.dot(rand2, ref) * ref
    orthogonal = normalize(orth_raw)

    # Sanity prints
    print("\nDirection cosine similarities (after normalization):")
    print(f"  probe_coef vs mean_diff           : {np.dot(probe_coef, mean_diff):+.4f}")
    print(f"  probe_coef vs mean_diff_zscored   : {np.dot(probe_coef, mean_diff_zscored):+.4f}")
    print(f"  probe_coef_zscored vs mean_diff_z : {np.dot(probe_coef_zscored, mean_diff_zscored):+.4f}")
    print(f"  mean_diff_zscored vs orthogonal   : {np.dot(mean_diff_zscored, orthogonal):+.4f} (should be ≈0)")
    print(f"  mean_diff_zscored vs random       : {np.dot(mean_diff_zscored, random_dir):+.4f} (small)")
    for name, vec in [
        ("probe_coef", probe_coef),
        ("probe_coef_zscored", probe_coef_zscored),
        ("mean_diff", mean_diff),
        ("mean_diff_zscored", mean_diff_zscored),
        ("random", random_dir),
        ("orthogonal", orthogonal),
    ]:
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-6, f"{name} not unit-norm"

    # Save everything
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        probe_coef=probe_coef.astype(np.float32),
        probe_coef_zscored=probe_coef_zscored.astype(np.float32),
        mean_diff=mean_diff.astype(np.float32),
        mean_diff_zscored=mean_diff_zscored.astype(np.float32),
        random=random_dir.astype(np.float32),
        orthogonal=orthogonal.astype(np.float32),
        zscore_mu=mu.astype(np.float32),
        zscore_sigma=sigma.astype(np.float32),
        layer=np.int32(args.layer),
    )
    print(f"\nSaved: {args.output}")

    meta = {
        "model_tag": args.model_tag,
        "layer": int(args.layer),
        "d_model": int(d_model),
        "n_train_prompts": int(n_train),
        "n_eval": int(n_eval),
        "n_deploy": int(n_deploy),
        "seed": int(args.seed),
        "directions": [
            "probe_coef", "probe_coef_zscored",
            "mean_diff", "mean_diff_zscored",
            "random", "orthogonal",
        ],
        "default_direction_for_reporting": "mean_diff_zscored",
        "triggers_acts_source": str(args.triggers_acts),
        "probes_source": str(args.probes),
        "n_low_variance_dims": int(n_low_var),
    }
    meta_path = args.output.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
