# alignment-evals

Empirical alignment-eval research and the shared infrastructure that supports it. Topics: evaluation awareness, sandbagging, scheming-style behavior, honesty evals — measured directly, with pre-registered designs and replication discipline.

**NOTE: Refer to V2 [below](https://github.com/Ratnaditya-J/alignment-evals/blob/main/README.md#v2) for the complete picutre on eval-awareness work.**

## What's in here

- **V1 Evaluation Awareness (complete).** A cross-protocol behavioral study of 8 frontier models + a Goodfire-style reasoning-trace analysis on the open-reasoning trio + a closed-source extension on `claude-opus-4-7` + a pre-registered disjoint-prompt replication on `qwen3-235B`.
  - Paper: [`docs/paper_draft.md`](docs/paper_draft.md)
  - Blog (long-form): [`docs/blog_draft.md`](docs/blog_draft.md)
  - Blog (~800-word version): [`docs/blog_draft_short.md`](docs/blog_draft_short.md)
  - Pre-registration document (filed before the replication ran; see git history for proof of timing): [`docs/pre_registration_qwen3_safety_mediation.md`](docs/pre_registration_qwen3_safety_mediation.md)
- **V2 (completed) ** Extending to Apollo Research's released 1,000-prompt evaluation-awareness benchmark for a labeled-ground-truth analysis.
- **Future Roadmap** for sandbagging / scheming-style probes and honesty evals: docs/v2_research_synthesis.md, docs/v2_architecture.md, docs/eval_awareness_sota_roadmap.md.

## V1 takeways

> Inject framing changes safety refusal in two reasoning models in opposite directions (qwen3 refuses +22pp more, opus refuses −34pp less). Neither effect shows up in the strict-VEA reasoning-trace signal; both manifest in the stratum where the model never wrote anything like "this is an eval." Explicit verbalized eval-awareness is correlated with the inject framing (replicated four times on qwen3) but is **not** the channel through which inject changes refusal behavior on a fresh, pre-registered prompt sample.

For details, the [paper](docs/paper_draft.md) is the canonical reference.

## Repro results

Two reproduction paths:

- **Judge-only** (~$10-$20, ~30 min run time). Re-run only the VEA judge against the released rollout bundle at [v1.0-rollouts](https://github.com/Ratnaditya-J/alignment-evals/releases/tag/v1.0-rollouts) (`alignment-evals-rollouts.tar.gz`; namespaced `v1/`, `v2/`, `v3/`; the `runs/` tree itself is not committed to this repo because of cumulative size); the trace JSON is saved per rollout so the model-under-test does not need to be re-invoked. Recommended for verifying §5.2-§5.4 numbers.
- **Full** (~$500 in API fees, a few hours). Re-run every model + judge call from scratch across the cross-protocol experiment (8 models × 4 conditions × n=2,000), the Goodfire open-reasoning trio (n=500), the opus-4.7 extension (n=300), the pre-registered qwen3 replication (n=200), and the judge passes over all of them.

Each summary JSON records the seeds, judge model identifiers, and prompt-template hashes used in its production.

## V2 

- **V2 (Completed):** integrate Apollo Research's [eval-awareness benchmark](https://github.com/jjpn97/eval_awareness) (1,000 labeled eval-vs-deployment transcripts) as a new corpus. Run the existing 4-condition arxiv protocol on top to study intrinsic eval signal × extrinsic eval cue interactions. Paper: [`docs/paper.pdf`](docs/paper.pdf)
- **Sandbagging probes (upcoming)** behavioral measurements designed to detect strategic under-performance during evaluation.
- **Honesty evals (upcoming):** measuring whether model self-reports about reasoning, beliefs, and capabilities track ground truth.
- **Scheming-style behavioral probes (upcoming)** following Apollo Research's lead.

See docs/v2_research_synthesis.md and docs/v2_architecture.md for the design details.

## Related work

- Apollo Research, [*Large Language Models Often Know When They Are Being Evaluated*](https://arxiv.org/abs/2505.23836) (Needham et al. 2025) — transcript-level classification of eval vs deployment.
- Laine et al., [*Me, Myself, and AI: The Situational Awareness Dataset (SAD)*](https://arxiv.org/abs/2407.04694) (NeurIPS 2024) — 7-category, 16-task benchmark of model self-knowledge.
- Apollo Research, [*Claude Sonnet 3.7 often knows when it's in alignment evaluations*](https://www.apolloresearch.ai/science/claude-sonnet-37-often-knows-when-its-in-alignment-evaluations/).
- Goodfire's spontaneous-VEA work and Anthropic's alignment-faking research are also relevant prior art.

## Background

I created eval-awareness work originally inside the `IsItBenchmark` repository (which focuses on benchmark contamination detection) and was spun off into this dedicated repository in May 2026 to:

- give the alignment-eval audience a clearly-named home for the work,
- make space for the planned sandbagging / honesty / scheming work,
- maintain the contamination-detection codebase separately.

Commit history from before the spinoff is preserved in this repository via `git filter-repo` original commit timestamps and authors are intact, which matters for the pre-registration document's filing timestamp. The pre-spinoff commits are also visible in [`Ratnaditya-J/IsItBenchmark`](https://github.com/Ratnaditya-J/IsItBenchmark).

## License

MIT. Feel free to use this for any future development or improvements on eval awareness work. 
