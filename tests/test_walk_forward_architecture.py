import pytest

from app.config.config import Config
from app.decision.confidence import normalize_final_confidence
from app.decision.decision_reliability import assess_decision_reliability
from app.optimization.fitness import normalize_wf_score


def test_negative_fitness_normalizes_to_zero():
    assert normalize_wf_score(-5, stability_constant=10) == 0.0


def test_zero_fitness_normalizes_to_zero():
    assert normalize_wf_score(0, stability_constant=10) == 0.0


def test_positive_fitness_normalizes_to_half():
    assert normalize_wf_score(10, stability_constant=10) == 0.5


def test_large_fitness_normalizes_below_one():
    score = normalize_wf_score(1000, stability_constant=10)
    assert score == pytest.approx(1000 / 1010, rel=1e-3)
    assert 0.0 <= score <= 1.0


def test_confidence_scaling_uses_normalized_score():
    base_confidence = 0.8
    normalized_score = 0.5
    rank_weight = 0.5
    raw_confidence = base_confidence * normalized_score * rank_weight
    final_confidence = normalize_final_confidence(raw_confidence)
    assert final_confidence == pytest.approx(0.2)


def test_zero_trust_disables_trade():
    final_confidence = normalize_final_confidence(0.0)
    result = assess_decision_reliability(final_confidence, wf_score=0.0)
    assert result.trade_allowed is False


def test_confidence_pipeline_integration_cases():
    base_confidence = 0.8
    rank_weight = 0.5

    for raw_fitness in (-5, 10, 1000):
        normalized_score = normalize_wf_score(raw_fitness, stability_constant=10)
        raw_confidence = base_confidence * normalized_score * rank_weight
        final_confidence = normalize_final_confidence(raw_confidence)

        assert 0.0 <= final_confidence <= 1.0

        result = assess_decision_reliability(
            final_confidence,
            wf_score=normalized_score,
        )

        if normalized_score == 0.0:
            assert result.trade_allowed is False
        else:
            assert result.trade_allowed == (
                final_confidence >= Config.CONFIDENCE_NO_TRADE_THRESHOLD
            )
