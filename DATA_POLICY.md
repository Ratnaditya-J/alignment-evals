# Data Policy

This document codifies hygiene rules for third-party datasets, model outputs,
and activations used in this project. It exists to prevent (a) evaluation
contamination, (b) reuse of code/data beyond its license, and (c) avoidable
reviewer objections at paper-submission time.

## SAD (Situational Awareness Dataset)

Source: <https://github.com/LRudL/sad> — CC-BY-4.0.

SAD is used by V4 (and may be used by other experiments) as part of the
training data for eval/deploy probes. The SAD maintainers ship the dataset
encrypted because **including SAD prompts in LLM pretraining corpora would
contaminate the benchmark** for every downstream user.

We are not pretraining anything, so probe training is in-scope. But the
maintainers also ask that question/answer text not be left in scrapeable
plaintext anywhere downstream — this means our repo, our paper appendix, and
any blog posts.

**Rules for this repo:**

- **No committed raw SAD prompts.** SAD content (questions, answers, options)
  must never be committed in plaintext to this repository.
- **No plaintext exports inside the repo.** Scripts that consume SAD must
  load it from a local checkout of `LRudL/sad` (typically via `unzip.sh`)
  and operate on it in-memory. Do not write decrypted SAD content to any
  file inside this repo, even in `runs/` or `data/` subdirectories,
  even with a `.gitignore` entry — gitignored files can still be staged
  accidentally and the directory is the wrong place for them on principle.
- **External cache directory for unavoidable plaintext artifacts.** When a
  pipeline step genuinely needs SAD plaintext written to disk (e.g., to
  scp to a GPU pod for activation extraction), write it to
  `~/.cache/alignment-evals-local/` instead. That path is outside the repo
  tree entirely. Scripts in this repo MAY read from there, but must NEVER
  write SAD plaintext inside the repo. The convention is enforced in
  `v4/scripts/load_and_dedup_sad.py` — see it for an example.
- **Inside the repo, only commit metadata.** Per-prompt records may
  contain: `sad_id`, `hash`, `label`, `source`, `length`. They must NOT
  contain: `body`, `text`, the prompt itself, model responses to SAD
  prompts, or any field that would let a reader reconstruct SAD content.
- **No SAD items in the paper appendix.** Cite, count, and hash. Do not
  reproduce items.
- **Loaders, hashes, counts, citations only.** Files we may commit include:
  loader code that reads SAD from an external path, the SHA-256 of each
  SAD file we depend on (for reproducibility), per-split counts, and the
  citation. The actual prompts stay out.
- **Disclose overlap in writing.** If any evaluation prompt is derived from
  SAD (e.g. for a held-out split), the paper must explicitly disclose this
  and confirm clean train/eval splits.

## Nguyen et al. evaluation-awareness-probing

Source: <https://github.com/Jordine/evaluation-awareness-probing> (deanonymized
from the ICML 2025 anonymous-review URL).

**No LICENSE file.** Treat this repo as **methodological reference only**. Do
not copy code or data wholesale into this repo. If we want to reuse any of
their constructed contrastive pairs or their loader scripts, email the authors
and confirm permission in writing first. Save the confirmation alongside any
reused artifact.

The pattern we are using ("Nguyen-style contrastive eval/deploy dataset
combining SAD `stages_oversight` with author-constructed pairs") is a
methodological approach, not a reuse of their specific data — we cite Nguyen
et al. as precedent and build our own equivalent.

## Apollo deception-detection

Source: <https://github.com/ApolloResearch/deception-detection>.

**No LICENSE file.** Pre-trained probe weights, datasets (sandbagging_v2,
ai_audit, roleplaying, insider_trading), and Llama-3.3-70B rollouts are all
shipped publicly, but without an explicit license. Before copying any of these
into this repo, email Apollo (Mikita Balesni / Marius Hobbhahn /
Nicholas Goldowsky-Dill) and confirm reuse terms. Save the confirmation
alongside any reused artifact.

Apollo are publicly working on related problems (see their metagaming post:
<https://www.apolloresearch.ai/blog/metagaming-matters-for-training-evaluation-and-oversight/>),
so coordination is also reasonable on its own terms — confirming reuse and
flagging research direction can be one email.

## Llama-3.3-70B activations

Activations extracted from Llama-3.3-70B at any layer are model-specific,
gigabyte-scale, and not useful to commit. They also encode the prompts they
were extracted on; if any of those prompts came from SAD, the activations
inherit the SAD plaintext constraint.

**Rules:**

- Do not commit raw activation files (`.pt`, `.npy`, `.safetensors`) to git.
  Use `runs/` for run outputs (already gitignored) or external storage.
- Per-prompt activation files derived from SAD prompts must be treated as
  SAD-derived artifacts: kept local, not exported, not referenced by
  filename-with-prompt-content.

## NLA outputs (AV explanations)

Per the agreed logging discipline, every AV call should log: model ID, source
ID, layer, token positions, prompt text, raw AV output, cosine/MSE,
timestamps, and any version headers from the hosted endpoint.

**Rules:**

- AV-output logs that include SAD prompts must follow the SAD plaintext
  constraint. Either redact the prompt field, or keep the logs out of the
  repo entirely (recommended).
- AV outputs themselves (not the prompts) can be committed — they are
  generated text and don't carry SAD's contamination concern.

## When in doubt

- Don't commit. Ask the relevant maintainers (SAD, Nguyen, Apollo) by email.
- Save permissions alongside reused artifacts so we can prove provenance at
  paper-submission time.
- The cost of an unnecessary email is much lower than the cost of having a
  reviewer flag a license or contamination concern.
