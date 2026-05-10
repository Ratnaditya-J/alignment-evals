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
- [`docs/eval_awareness_architecture.mmd`](eval_awareness_architecture.mmd)

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
    %% Polished evaluation-awareness architecture overview for IsItBenchmark v2

    subgraph SourceSignals["Model-visible source signals"]
        direction TB
        S1["System & developer prompts"]
        S2["User task + answer format"]
        S3["Conversation history"]
        S4["Visible metadata / harness traces"]
        S5["Benchmark or rubric context"]
    end

    subgraph Normalization["Transcript normalization"]
        direction TB
        N1["TranscriptInput / plain text wrapper"]
        N2["Render model-visible context"]
        N3["Preserve evidence spans"]
    end

    subgraph Detection["Cue detection layer"]
        direction TB
        D1["Explicit evaluation wording"]
        D2["Benchmark references"]
        D3["Scoring / rubric leakage"]
        D4["Artificial answer constraints"]
        D5["Safety or alignment-test framing"]
        D6["Harness artifacts"]
        D7["Sandbagging / scheming cues"]
        D8["Deployment-like counter-signals"]
    end

    subgraph Scoring["Risk aggregation"]
        direction TB
        R1["Weight cues by category"]
        R2["Apply per-category caps"]
        R3["Subtract deployment counterweight"]
        R4["Clamp final score"]
        R5["Assign risk band"]
    end

    subgraph Outputs["Reviewer-facing outputs"]
        direction TB
        O1["Evaluation-awareness score"]
        O2["Risk level"]
        O3["Deployment mimic score"]
        O4["Evidence cues + spans"]
        O5["Mitigation recommendations"]
        O6["Markdown leakage report"]
    end

    subgraph Future["Future scorer plug-ins"]
        direction TB
        F1["Transcript classifier"]
        F2["Benchmark similarity signal"]
        F3["LLM purpose judge"]
        F4["Paired behavior-shift runner"]
        F5["Activation probes"]
    end

    SourceSignals --> Normalization --> Detection --> Scoring --> Outputs
    Detection -. "candidate features" .-> Future
    Future -. "calibrated signals" .-> Scoring

    classDef source fill:#e0f2fe,stroke:#0284c7,stroke-width:2px,color:#0f172a;
    classDef normalize fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#0f172a;
    classDef detect fill:#ede9fe,stroke:#7c3aed,stroke-width:2px,color:#0f172a;
    classDef score fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#0f172a;
    classDef output fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#0f172a;
    classDef future fill:#f1f5f9,stroke:#64748b,stroke-width:2px,stroke-dasharray: 5 5,color:#0f172a;

    class S1,S2,S3,S4,S5 source;
    class N1,N2,N3 normalize;
    class D1,D2,D3,D4,D5,D6,D7,D8 detect;
    class R1,R2,R3,R4,R5 score;
    class O1,O2,O3,O4,O5,O6 output;
    class F1,F2,F3,F4,F5 future;
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

Future scorers now plug into the same result model through the runnable Phase 1-5 modules:

- `validation.py` for labeled eval-vs-deployment scanner metrics;
- `verbalized.py` for quote-level verbalized evaluation-awareness monitoring;
- `runner.py` for paired obvious-eval, hidden-eval, and deployment-like rollouts;
- `rewrite.py` for leakage-reducing prompt rewrites and before/after scores;
- `whitebox.py` for activation-probe and steering-vector extension points.

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
