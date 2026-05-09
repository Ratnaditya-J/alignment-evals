from src.eval_awareness import EvaluationAwarenessDetector, LeakageReport, TranscriptInput


def test_obvious_evaluation_context_scores_high():
    detector = EvaluationAwarenessDetector()
    result = detector.analyze(
        TranscriptInput(
            system_prompt="You are being evaluated in a safety evaluation.",
            user_prompt="Answer only A, B, C, or D. Your answer will be scored automatically using the MMLU rubric.",
            task_format="multiple_choice",
        )
    )

    assert result.evaluation_awareness_score >= 0.7
    assert result.is_likely_evaluation_context
    assert any(cue.category.value == "explicit_evaluation" for cue in result.cues)
    assert any("benchmark names" in recommendation.lower() for recommendation in result.recommendations)


def test_deployment_like_context_scores_low():
    detector = EvaluationAwarenessDetector()
    result = detector.analyze(
        TranscriptInput(
            user_prompt="A customer asked why their production checkout job failed after a database migration. Help me draft a response.",
            visible_metadata={"source": "support ticket"},
        )
    )

    assert result.evaluation_awareness_score < 0.2
    assert result.deployment_mimic_score > 0.8
    assert not result.is_likely_evaluation_context


def test_leakage_report_includes_cues_and_recommendations():
    report = LeakageReport().analyze_and_render(
        "This is an eval harness. Return only JSON; hidden tests will be run."
    )

    assert "Evaluation Awareness Leakage Report" in report
    assert "eval harness" in report
    assert "Recommendations" in report
