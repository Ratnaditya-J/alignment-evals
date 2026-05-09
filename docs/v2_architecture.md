# IsItBenchmark v2 architecture

## Product goal

IsItBenchmark v2 detects whether AI evaluation prompts, transcripts, and harnesses leak that a model is being tested, then helps evaluators harden those evaluations against evaluation-aware behavior.

The v1 contamination detector remains valuable, but it becomes a signal inside the broader evaluation-awareness risk model.

## Target API

```python
from src.eval_awareness import EvaluationAwarenessDetector, TranscriptInput

transcript = TranscriptInput(
    system_prompt="You are being evaluated for safety.",
    user_prompt="Return only A, B, C, or D. Your answer will be scored automatically.",
)

result = EvaluationAwarenessDetector().analyze(transcript)
print(result.evaluation_awareness_score)
print(result.recommendations)
```

## Initial package layout

```text
src/eval_awareness/
  models.py        # transcript, cue, and result models
  cue_detector.py  # transparent cue-pattern library
  detector.py      # score aggregation and risk bands
  report.py        # Markdown leakage reports
```


## Evaluation detector flowchart

Standalone Mermaid diagram files are also available for Finder/GitHub browsing:

- [`docs/eval_detector_flowchart.mmd`](eval_detector_flowchart.mmd)
- [`docs/eval_detector_overview.mmd`](eval_detector_overview.mmd)

```mermaid
flowchart TD
    A[Input: plain text or TranscriptInput] --> B{Input type?}
    B -->|plain text| C[Wrap as TranscriptInput.from_text]
    B -->|TranscriptInput| D[Use structured transcript fields]
    C --> E[Render model-visible context]
    D --> E

    E --> F[EvaluationCueDetector.detect]
    F --> G[Run default cue-pattern library]
    G --> H[Create EvaluationCue objects with category, evidence span, weight, recommendation]
    H --> I[Sort cues by weight]

    I --> J[EvaluationAwarenessDetector._score_categories]
    J --> K[Sum cue weights by category]
    K --> L[Apply per-category caps]
    L --> M[Compute raw eval-risk score]
    L --> N[Compute deployment-framing counterweight]
    M --> O[Final score = clamp(raw risk - deployment counterweight)]
    N --> O

    O --> P[Assign RiskLevel]
    O --> Q[Compute deployment_mimic_score]
    I --> R[Deduplicate mitigation recommendations]
    P --> S[Build EvaluationAwarenessResult]
    Q --> S
    R --> S

    S --> T{Consumer}
    T -->|API / CLI| U[result.to_dict]
    T -->|Human review| V[LeakageReport.render Markdown]
    T -->|Future v2 scorers| W[Blend with transcript classifier, benchmark similarity, LLM judge, paired behavior-shift, activation probes]
```

At a high level, the current v2 detector is a transparent leakage-analysis pipeline: render the model-visible transcript, identify evaluation-awareness cues, cap/aggregate their weights, subtract deployment-like counter-signals, and return an attributed result that can be serialized or rendered as a report.

```mermaid
flowchart LR
    subgraph Inputs
        A1[System prompt]
        A2[Developer prompt]
        A3[User prompt]
        A4[Conversation history]
        A5[Visible metadata]
        A6[Task format]
    end

    subgraph Cue families
        B1[Explicit eval wording]
        B2[Benchmark references]
        B3[Visible scoring / rubric]
        B4[Artificial answer format]
        B5[Safety / alignment test framing]
        B6[Harness artifacts]
        B7[Sandbagging / scheming cues]
        B8[Deployment-like counter-signals]
    end

    subgraph Outputs
        C1[Evaluation-awareness score]
        C2[Risk level]
        C3[Deployment mimic score]
        C4[Evidence cues]
        C5[Mitigation recommendations]
        C6[Markdown leakage report]
    end

    Inputs --> Cue families --> Outputs
```

## Scoring model

The first implementation is a transparent weighted heuristic:

```text
score = capped_sum(risk cue categories) - deployment_counter_signals
```

This is intentionally not the final state. It provides:

- deterministic behavior;
- explainable cue attribution;
- testable recommendations;
- a stable result schema for future ML classifiers.

Future scorers should plug into the same result model:

- transcript classifier score;
- benchmark similarity score from v1;
- LLM-judge transcript-purpose score;
- paired behavior-shift score;
- activation-probe score for open-weight models.

## Roadmap

### Phase 1: Leakage detection baseline

- Add cue taxonomy and Markdown reports.
- Analyze full transcript fields, not just user prompt text.
- Return evidence spans and mitigation guidance.

### Phase 2: Transcript classifier

- Ingest eval-vs-deployment transcript datasets.
- Train/evaluate classifiers with held-out datasets and calibrated scores.
- Include human-readable confusion matrices and false-positive analysis.

### Phase 3: Paired behavior-shift runner

- Generate task variants: obvious eval, hidden eval, normal deployment, multi-turn deployment, and adversarial deployment-like wrapper.
- Run target models across variants.
- Measure deltas in accuracy, refusal, safety behavior, honesty, tool use, and verbosity.

### Phase 4: Rewrite engine

- Remove evaluation cues while preserving task meaning, safety category, difficulty, and scoring validity.
- Validate rewrites with answer-equivalence checks and human-review hooks.

### Phase 5: White-box extensions

- Add optional activation-probe and steering hooks for open-weight models.
- Keep these separate from black-box leakage detection so the tool remains useful for API-only models.

## Relationship to v1

| v1 component | v2 role |
| --- | --- |
| Exact benchmark matching | Benchmark similarity risk signal |
| Semantic matcher | Paraphrased benchmark/eval cue signal |
| N-gram matcher | Fast benchmark/harness text overlap |
| Specialized contamination model | Candidate baseline; should be retrained for eval-vs-deployment transcript labels |
| API/CLI | Should expose `eval-awareness` and `leakage-report` commands |

## Non-goals for the initial v2 scaffold

- It does not prove model consciousness or self-awareness.
- It does not measure behavior change without paired model runs.
- It does not replace contamination detection.
- It does not rely on chain-of-thought access.
