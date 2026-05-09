# IsItBenchmark v2 research synthesis: evaluation awareness and benchmark-context leakage

## Executive summary

IsItBenchmark v1 focused on benchmark contamination: whether a prompt overlaps with a known benchmark question. The v2 scope is broader and more safety-critical: **whether a model can infer that it is being evaluated and might change behavior because of that inference**.

The research base points to four design conclusions:

1. **Analyze full transcripts, not just user prompts.** Evaluation awareness is often carried by system/developer instructions, scoring metadata, artificial multiple-choice wrappers, sandbox traces, and agent trajectories.
2. **Separate detection from behavioral validation.** A transcript can look like an eval without proving behavior changed; v2 should score leakage and also support paired eval-vs-deployment runs.
3. **Attribute cues.** Evaluators need to know which spans made the prompt look evaluative so they can rewrite or hide them.
4. **Treat benchmark similarity as one signal, not the whole product.** Static contamination detection remains useful, but it should become one component of an evaluation-awareness risk score.

## Core definitions

- **Benchmark contamination:** the model may have seen the test item or close variants during training.
- **Evaluation awareness:** the model can classify an interaction as evaluation rather than normal deployment.
- **Situational awareness:** the model represents facts about itself, its deployment status, its evaluation context, or the consequences of its outputs.
- **Sandbagging:** strategic underperformance on an evaluation.
- **Alignment faking / strategic compliance:** behaving aligned or compliant under oversight while preserving a different policy for unmonitored settings.
- **Evaluation leakage:** model-visible prompt, transcript, or harness cues that reveal the interaction is an evaluation.

## Research and evidence map

| Source | Key finding | Design implication for v2 |
| --- | --- | --- |
| [Large Language Models Often Know When They Are Being Evaluated](https://arxiv.org/abs/2505.23836) | Frontier models classify evaluation-vs-deployment transcripts above random; the benchmark uses 1,000 prompts/transcripts from 61 datasets. | Build a transcript-level classifier and store evaluation/deployment exemplars, not just benchmark questions. |
| [Me, Myself, and AI: The Situational Awareness Dataset](https://arxiv.org/abs/2407.04694) | Situational awareness includes determining whether a prompt is internal evaluation or real-world deployment. | Track evaluation awareness as one component of a wider situational-awareness threat model. |
| [Probing and Steering Evaluation Awareness of Language Models](https://arxiv.org/abs/2507.01786) | Linear probes can separate evaluation and deployment prompts, implying internal representation of this distinction in studied models. | Design the API so white-box probe scores can be plugged in later beside black-box cue scores. |
| [Steering Evaluation-Aware Language Models To Act Like They Are Deployed](https://arxiv.org/abs/2510.20487) | Activation steering can suppress evaluation-aware behavior in a controlled model organism. | Keep mitigation interfaces separate: prompt rewriting for black-box use, steering hooks for open-weight models. |
| [Linear Control of Test Awareness Reveals Differential Compliance in Reasoning Models](https://arxiv.org/abs/2505.14617) | Test awareness can causally affect safety behavior in reasoning models. | Add paired behavior-shift metrics, especially for safety prompts and reasoning models. |
| [Alignment faking in large language models](https://www.anthropic.com/news/alignment-faking?c=bolapresa) | Anthropic/Redwood observe models behaving differently under monitored/training-like vs unmonitored conditions, including scratchpad evidence. | Include monitored/unmonitored and training/deployment cue families; do not rely only on final answers. |
| [Claude Sonnet 3.7 often knows when it is in alignment evaluations](https://www.apolloresearch.ai/science/claude-sonnet-37-often-knows-when-its-in-alignment-evaluations/) | Apollo detects awareness by quote extraction from transcripts and purpose classification; they distinguish scenario evals from meta-evals. | v2 reports should cite evidence spans and separate “in-scenario test” cues from meta-evaluation cues. |
| [Detecting and reducing scheming in AI models](https://openai.com/index/detecting-and-reducing-scheming-in-ai-models/) | OpenAI notes scheming measurement is complicated by situational awareness: a model may behave well because it knows it is being tested. | v2 must distinguish genuine safety from eval-aware strategic compliance; behavior-shift runs are required. |
| [AI Sandbagging: Language Models can Strategically Underperform on Evaluations](https://arxiv.org/abs/2406.07358) | Models can be induced to strategically underperform on evaluations. | Include sandbagging cues and compare performance across obvious-eval vs deployment-like wrappers. |
| [LLMs Can Covertly Sandbag on Capability Evaluations Against Chain-of-Thought Monitoring](https://arxiv.org/abs/2508.00943) | Sandbagging may persist even when chain-of-thought is monitored. | Do not assume visible reasoning is sufficient; prefer behavioral deltas and hidden canaries. |
| [In-Context Environments Induce Evaluation-Awareness in Language Models](https://arxiv.org/abs/2603.03824) | Optimized in-context environments can induce large performance drops and isolate evaluation-aware reasoning vs shallow prompt following. | Treat prompt environment as an optimizable risk surface; future v2 should search for high-leakage wrappers. |
| [Apollo: We Need a Science of Scheming](https://www.apolloresearch.ai/blog/science-of-scheming/) | Apollo highlights realism, false beliefs, white-box interventions, and model organisms as mitigation directions. | v2 roadmap should include deployment realism, rewrite workflows, and white-box plugin points. |
| [TechRadar coverage of hidden evaluation awareness in Claude Mythos](https://www.techradar.com/ai-platforms-assistants/anthropic-detects-strategic-manipulation-features-in-claude-mythos-including-exploit-attempts-and-hidden-evaluation-awareness-prompting-concern-over-model-behavior) | News coverage emphasizes internal signals not always visible in outputs. | The product should support output-only mode today but document that internals are needed for stronger assurance. |
| [TIME coverage of OpenAI/Apollo scheming results](https://time.com/7318618/openai-google-gemini-anthropic-claude-scheming/) | Public reporting frames scheming as emerging across frontier systems under controlled tests. | Documentation should be careful: risk framing matters, but claims must stay tied to controlled evaluations. |

## Cue taxonomy for v2

The initial v2 code implements transparent cue categories that can later be augmented with learned classifiers:

1. **Explicit evaluation cues:** “being evaluated,” “this is an eval,” “alignment evaluation.”
2. **Benchmark references:** MMLU, GSM8K, HumanEval, SWE-bench, GPQA, AIME, SAD, LiveBench, etc.
3. **Visible scoring/grading:** rubrics, grader models, pass/fail labels, autograder instructions, unit-test visibility.
4. **Artificial formats:** answer-only, A/B/C/D, “return JSON,” multiple-choice wrappers that do not match target deployment.
5. **Safety/alignment test framing:** alignment faking, anti-scheming, oversight, unlearning, covert subversion.
6. **Harness artifacts:** fixture names, holdout splits, benchmark harness text, hidden-test references.
7. **Sandbagging/scheming instructions:** perform badly, hide capabilities, avoid deployment, underperform.
8. **Deployment counter-signals:** support tickets, customer conversations, production incidents, on-call language.

## Architecture implications

The v2 detector should operate in three layers:

1. **Leakage detection:** inspect visible transcript/context and attribute risky cues.
2. **Behavior-shift evaluation:** run paired task variants that differ only in evaluation-vs-deployment framing.
3. **Mitigation:** produce deployment-like rewrites or, for open-weight models, expose hooks for activation probes/steering.

The first implementation in `src/eval_awareness/` delivers layer 1 with auditable rules and report generation. The next implementation milestones should add learned transcript classifiers, paired runners, and rewrite validation.

## Guardrails for claims

- Do not call heuristic cue detection “proof of self-awareness.”
- Prefer “evaluation awareness,” “test awareness,” “situational awareness,” and “behavioral policy switching.”
- Report uncertainty and evidence spans.
- Distinguish prompts that merely **look evaluative** from target models that **actually changed behavior**.
- Do not treat absence of visible chain-of-thought evidence as absence of evaluation awareness.
