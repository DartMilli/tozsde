"""
Comprehensive tests for recommendation_builder and decision_builder modules.
(Sprint 10 Week 2 - Issue #5 Coverage Improvement)

Tests:
- build_recommendation() with various confidence levels
- Strength classification (STRONG/NORMAL/WEAK/NO_TRADE)
- NO-TRADE override logic
- build_explanation() for HU and EN languages
- compute_decision_quality() scoring
- weighted_ensemble_decision() logic
- Edge cases and boundary conditions

Coverage Target: recommendation_builder 32% → 70%+
                decision_builder 41% → 75%+
"""

import pytest
from unittest.mock import patch, MagicMock
from app.decision.recommendation_builder import (
    build_recommendation,
    build_explanation,
)
from app.decision.decision_builder import (
    weighted_ensemble_decision,
    compute_decision_quality,
)
from app.config.config import Config


class TestBuildRecommendationBasic:
    """Test basic recommendation building."""

    def test_buy_signal_normal_strength(self):
        """Should build NORMAL BUY recommendation."""
        payload = {
            "ticker": "AAPL",
            "avg_confidence": 0.65,
            "avg_wf_score": 0.5,
            "action_code": 1,  # BUY
            "ensemble_quality": "MIXED",
        }

        result = build_recommendation(payload)

        assert result["action_code"] == 1
        assert result["action"] == Config.ACTION_LABELS[Config.LANG][1]
        assert result["strength"] == "NORMAL"
        assert result["no_trade"] is False
        assert result["confidence"] == 0.65

    def test_sell_signal_normal_strength(self):
        """Should build NORMAL SELL recommendation."""
        payload = {
            "ticker": "MSFT",
            "avg_confidence": 0.60,
            "avg_wf_score": 0.45,
            "action_code": 2,  # SELL
            "ensemble_quality": "MIXED",
        }

        result = build_recommendation(payload)

        assert result["action_code"] == 2
        assert result["action"] == Config.ACTION_LABELS[Config.LANG][2]
        assert result["strength"] == "NORMAL"
        assert result["no_trade"] is False

    def test_hold_signal(self):
        """Should build HOLD recommendation."""
        payload = {
            "ticker": "JNJ",
            "avg_confidence": 0.45,
            "avg_wf_score": 0.3,
            "action_code": 0,  # HOLD
            "ensemble_quality": "CHAOTIC",
        }

        result = build_recommendation(payload)

        assert result["action_code"] == 0
        assert result["action"] == Config.ACTION_LABELS[Config.LANG][0]
        assert result["no_trade"] is False


class TestRecommendationStrengthClassification:
    """Test strength classification logic."""

    def test_strong_buy_high_confidence_stable_ensemble(self):
        """Should classify as STRONG BUY."""
        payload = {
            "ticker": "AAPL",
            "avg_confidence": 0.85,  # >= STRONG_CONFIDENCE_THRESHOLD
            "avg_wf_score": 0.75,  # >= STRONG_WF_THRESHOLD
            "action_code": 1,
            "ensemble_quality": "STABLE",
        }

        result = build_recommendation(payload)

        assert result["strength"] == "STRONG"
        assert result["action_code"] == 1

    def test_weak_buy_low_confidence(self):
        """Should classify as WEAK BUY."""
        payload = {
            "ticker": "MSFT",
            "avg_confidence": 0.35,  # < WEAK_CONFIDENCE_THRESHOLD (0.4)
            "avg_wf_score": 0.5,
            "action_code": 1,
            "ensemble_quality": "MIXED",
        }

        result = build_recommendation(payload)

        assert result["strength"] == "WEAK"
        assert result["action_code"] == 1

    def test_strong_sell_with_conditions(self):
        """Should classify as STRONG SELL."""
        payload = {
            "ticker": "VOO",
            "avg_confidence": 0.82,
            "avg_wf_score": 0.70,
            "action_code": 2,
            "ensemble_quality": "STABLE",
        }

        result = build_recommendation(payload)

        assert result["strength"] == "STRONG"
        assert result["action_code"] == 2


class TestNoTradeOverride:
    """Test NO-TRADE policy override logic."""

    def test_low_confidence_triggers_no_trade(self):
        """Should override BUY with NO_TRADE when confidence too low."""
        payload = {
            "ticker": "QQQ",
            "avg_confidence": 0.20,  # < CONFIDENCE_NO_TRADE_THRESHOLD (0.25)
            "avg_wf_score": 0.5,
            "action_code": 1,  # BUY
            "ensemble_quality": "MIXED",
        }

        result = build_recommendation(payload)

        assert result["no_trade"] is True
        assert result["action_code"] == 0  # HOLD
        assert result["strength"] == "NO_TRADE"
        assert result["no_trade_reason"] == "LOW_CONFIDENCE"

    def test_no_trade_preserves_original_action(self):
        """Should preserve original action when no_trade triggered."""
        payload = {
            "ticker": "SPY",
            "avg_confidence": 0.15,  # < CONFIDENCE_NO_TRADE_THRESHOLD (0.25)
            "avg_wf_score": 0.4,
            "action_code": 2,  # SELL
            "ensemble_quality": "CHAOTIC",
        }

        result = build_recommendation(payload)

        assert result["no_trade"] is True
        assert result["original_action"] == 2
        assert result["action_code"] == 0

    def test_normal_confidence_no_no_trade(self):
        """Should not trigger NO_TRADE with normal confidence."""
        payload = {
            "ticker": "XOM",
            "avg_confidence": 0.55,
            "avg_wf_score": 0.5,
            "action_code": 1,
            "ensemble_quality": "MIXED",
        }

        result = build_recommendation(payload)

        assert result["no_trade"] is False
        assert result["no_trade_reason"] is None


class TestBuildExplanationBasic:
    """Test explanation building."""

    def test_explanation_hu_language(self):
        """Should build Hungarian explanation."""
        payload = {
            "ticker": "AAPL",
            "avg_confidence": 0.70,
            "avg_wf_score": 0.60,
            "ensemble_quality": "STABLE",
            "model_votes": [
                {
                    "model_name": "Model1",
                    "action_label": "BUY",
                    "confidence": 0.75,
                    "wf_score": 0.65,
                }
            ],
        }

        decision = {
            "action": "BUY",
            "strength": "STRONG",
            "no_trade": False,
            "quality_score": 0.72,
        }

        result = build_explanation(payload, decision)

        assert "hu" in result
        assert "en" in result
        assert "meta" in result
        assert "AAPL" in result["hu"]
        assert "STRONG" in result["hu"]

    def test_explanation_en_language(self):
        """Should build English explanation."""
        payload = {
            "ticker": "MSFT",
            "avg_confidence": 0.65,
            "avg_wf_score": 0.55,
            "ensemble_quality": "MIXED",
            "model_votes": [],
        }

        decision = {
            "action": "BUY",
            "strength": "NORMAL",
            "no_trade": False,
            "quality_score": 0.65,
        }

        result = build_explanation(payload, decision)

        assert "en" in result
        assert "MSFT" in result["en"]
        assert "NORMAL" in result["en"]

    def test_explanation_includes_model_votes(self):
        """Should include model votes in explanation."""
        payload = {
            "ticker": "JNJ",
            "avg_confidence": 0.60,
            "avg_wf_score": 0.50,
            "ensemble_quality": "STABLE",
            "model_votes": [
                {
                    "model_name": "ModelA",
                    "action_label": "BUY",
                    "confidence": 0.70,
                    "wf_score": 0.60,
                },
                {
                    "model_name": "ModelB",
                    "action_label": "SELL",
                    "confidence": 0.50,
                    "wf_score": 0.40,
                },
            ],
        }

        decision = {
            "action": "BUY",
            "strength": "NORMAL",
            "no_trade": False,
            "quality_score": 0.68,
        }

        result = build_explanation(payload, decision)

        assert "ModelA" in result["hu"]
        assert "ModelB" in result["hu"]
        assert "conf=" in result["hu"]
        assert "wf=" in result["hu"]


class TestExplanationNoTradeScenarios:
    """Test explanation building with NO_TRADE scenarios."""

    def test_explanation_low_confidence_no_trade(self):
        """Should explain LOW_CONFIDENCE no-trade."""
        payload = {
            "ticker": "SPY",
            "avg_confidence": 0.30,
            "avg_wf_score": 0.40,
            "ensemble_quality": "CHAOTIC",
            "model_votes": [],
        }

        decision = {
            "action": "HOLD",
            "strength": "NO_TRADE",
            "no_trade": True,
            "no_trade_reason": "LOW_CONFIDENCE",
            "original_action": 1,
            "quality_score": 0.35,
        }

        result = build_explanation(payload, decision)

        assert "LOW_CONFIDENCE" in result["hu"] or "policy" in result["hu"].lower()
        assert "LOW_CONFIDENCE" in result["en"] or "policy" in result["en"].lower()

    def test_explanation_low_wf_score_reason(self):
        """Should include low WF score as reason."""
        payload = {
            "ticker": "VOO",
            "avg_confidence": 0.65,
            "avg_wf_score": 0.30,  # Low WF
            "ensemble_quality": "MIXED",
            "model_votes": [],
        }

        decision = {
            "action": "BUY",
            "strength": "NORMAL",
            "no_trade": False,
            "quality_score": 0.55,
        }

        result = build_explanation(payload, decision)

        # Should mention low walk-forward stability
        assert "walk-forward" in result["hu"].lower() or "0.30" in result["hu"]
        assert "walk-forward" in result["en"].lower() or "0.30" in result["en"]

    def test_explanation_chaotic_ensemble_reason(self):
        """Should mention CHAOTIC ensemble quality."""
        payload = {
            "ticker": "QQQ",
            "avg_confidence": 0.60,
            "avg_wf_score": 0.55,
            "ensemble_quality": "CHAOTIC",
            "model_votes": [],
        }

        decision = {
            "action": "BUY",
            "strength": "NORMAL",
            "no_trade": False,
            "quality_score": 0.58,
        }

        result = build_explanation(payload, decision)

        assert "CHAOTIC" in result["hu"] or "eltérnek" in result["hu"]
        assert "CHAOTIC" in result["en"] or "divergent" in result["en"]


class TestComputeDecisionQuality:
    """Test decision quality scoring."""

    def test_quality_perfect_conditions(self):
        """Should score high with perfect conditions."""
        payload = {
            "avg_confidence": 0.90,
            "avg_wf_score": 0.85,
            "ensemble_quality": "STABLE",
        }

        score = compute_decision_quality(payload)

        assert score > 0.80
        assert isinstance(score, float)

    def test_quality_low_confidence(self):
        """Should score lower with low confidence."""
        payload = {
            "avg_confidence": 0.40,
            "avg_wf_score": 0.70,
            "ensemble_quality": "STABLE",
        }

        score = compute_decision_quality(payload)

        assert score < 0.65

    def test_quality_chaotic_ensemble(self):
        """Should score lower with CHAOTIC ensemble."""
        payload = {
            "avg_confidence": 0.70,
            "avg_wf_score": 0.65,
            "ensemble_quality": "CHAOTIC",
        }

        score = compute_decision_quality(payload)

        assert score < 0.70

    def test_quality_mixed_ensemble(self):
        """Should score medium with MIXED ensemble."""
        payload = {
            "avg_confidence": 0.70,
            "avg_wf_score": 0.65,
            "ensemble_quality": "MIXED",
        }

        score = compute_decision_quality(payload)

        assert 0.50 < score < 0.70

    def test_quality_missing_fields_defaults(self):
        """Should handle missing fields with defaults."""
        payload = {}

        score = compute_decision_quality(payload)

        assert 0.0 <= score <= 1.0
        assert isinstance(score, float)


class TestWeightedEnsembleDecision:
    """Test weighted ensemble decision logic."""

    def test_weighted_ensemble_single_model(self):
        """Should handle single model vote."""
        payload = {
            "model_votes": [
                {
                    "model_path": "model1.pkl",
                    "action": 1,  # BUY
                    "confidence": 0.80,
                    "wf_score": 0.70,
                }
            ]
        }

        reliability_scores = {"model1.pkl": 0.85}

        result = weighted_ensemble_decision(payload, reliability_scores)

        assert result["action_code"] == 1
        assert result["weighted_confidence"] > 0.0
        assert result["dominance"] > 0.5

    def test_weighted_ensemble_conflicting_votes(self):
        """Should resolve conflicting votes based on weight."""
        payload = {
            "model_votes": [
                {
                    "model_path": "model1.pkl",
                    "action": 1,  # BUY
                    "confidence": 0.90,
                    "wf_score": 0.80,
                },
                {
                    "model_path": "model2.pkl",
                    "action": 2,  # SELL
                    "confidence": 0.50,
                    "wf_score": 0.50,
                },
            ]
        }

        reliability_scores = {
            "model1.pkl": 0.90,
            "model2.pkl": 0.60,
        }

        result = weighted_ensemble_decision(payload, reliability_scores)

        # Model 1 has higher weight, should win
        assert result["action_code"] == 1

    def test_weighted_ensemble_equal_votes(self):
        """Should handle equally weighted votes."""
        payload = {
            "model_votes": [
                {
                    "model_path": "model1.pkl",
                    "action": 1,
                    "confidence": 0.70,
                    "wf_score": 0.60,
                },
                {
                    "model_path": "model2.pkl",
                    "action": 1,
                    "confidence": 0.70,
                    "wf_score": 0.60,
                },
            ]
        }

        reliability_scores = {
            "model1.pkl": 0.80,
            "model2.pkl": 0.80,
        }

        result = weighted_ensemble_decision(payload, reliability_scores)

        assert result["action_code"] == 1
        assert result["weighted_confidence"] > 0.0

    def test_weighted_ensemble_missing_reliability(self):
        """Should use default reliability for missing models."""
        payload = {
            "model_votes": [
                {
                    "model_path": "unknown_model.pkl",
                    "action": 1,
                    "confidence": 0.75,
                    "wf_score": 0.65,
                }
            ]
        }

        reliability_scores = {}  # Empty

        result = weighted_ensemble_decision(payload, reliability_scores)

        # Should still produce result with default reliability
        assert result["action_code"] == 1
        assert "vote_debug" in result

    def test_weighted_ensemble_missing_wf_score(self):
        """Should handle missing WF scores."""
        payload = {
            "model_votes": [
                {
                    "model_path": "model1.pkl",
                    "action": 1,
                    "confidence": 0.80,
                    # wf_score missing
                }
            ]
        }

        reliability_scores = {"model1.pkl": 0.85}

        result = weighted_ensemble_decision(payload, reliability_scores)

        assert result["action_code"] == 1
        # Should have used default wf_score of 1.0
        assert result["weighted_confidence"] > 0.0

    def test_weighted_ensemble_zero_total_weight(self):
        """Should handle edge case of zero total weight."""
        payload = {
            "model_votes": [
                {
                    "model_path": "model1.pkl",
                    "action": 1,
                    "confidence": 0.0,
                    "wf_score": 0.0,
                }
            ]
        }

        reliability_scores = {"model1.pkl": 0.0}

        result = weighted_ensemble_decision(payload, reliability_scores)

        assert result["weighted_confidence"] == 0.0


class TestBoundaryConditions:
    """Test boundary and edge case conditions."""

    def test_extreme_high_confidence(self):
        """Should handle confidence = 1.0."""
        payload = {
            "ticker": "TEST",
            "avg_confidence": 1.0,
            "avg_wf_score": 1.0,
            "action_code": 1,
            "ensemble_quality": "STABLE",
        }

        result = build_recommendation(payload)

        assert result["strength"] == "STRONG"
        assert result["no_trade"] is False

    def test_extreme_low_confidence(self):
        """Should handle confidence = 0.0."""
        payload = {
            "ticker": "TEST",
            "avg_confidence": 0.0,
            "avg_wf_score": 0.0,
            "action_code": 1,
            "ensemble_quality": "CHAOTIC",
        }

        result = build_recommendation(payload)

        assert result["no_trade"] is True

    def test_none_wf_score(self):
        """Should handle None WF score."""
        payload = {
            "ticker": "TEST",
            "avg_confidence": 0.70,
            "avg_wf_score": None,
            "action_code": 1,
            "ensemble_quality": "MIXED",
        }

        result = build_recommendation(payload)

        assert result["strength"] == "NORMAL"  # Can't be STRONG without WF score
        assert result["wf_score"] is None

    def test_empty_model_votes(self):
        """Should handle empty model votes."""
        payload = {
            "ticker": "TEST",
            "avg_confidence": 0.65,
            "avg_wf_score": 0.55,
            "ensemble_quality": "MIXED",
            "model_votes": [],
        }

        decision = {
            "action": "BUY",
            "strength": "NORMAL",
            "no_trade": False,
            "quality_score": 0.65,
        }

        result = build_explanation(payload, decision)

        assert "hu" in result
        assert "en" in result


class TestRecommendationQualityScore:
    """Test quality score computation in recommendations."""

    def test_recommendation_includes_quality_score(self):
        """Should include computed quality score."""
        payload = {
            "ticker": "AAPL",
            "avg_confidence": 0.75,
            "avg_wf_score": 0.65,
            "action_code": 1,
            "ensemble_quality": "STABLE",
        }

        result = build_recommendation(payload)

        assert "quality_score" in result
        assert result["quality_score"] > 0.0

    def test_quality_score_reflects_components(self):
        """Quality score should reflect input components."""
        high_quality_payload = {
            "ticker": "TEST1",
            "avg_confidence": 0.90,
            "avg_wf_score": 0.85,
            "action_code": 1,
            "ensemble_quality": "STABLE",
        }

        low_quality_payload = {
            "ticker": "TEST2",
            "avg_confidence": 0.40,
            "avg_wf_score": 0.35,
            "action_code": 1,
            "ensemble_quality": "CHAOTIC",
        }

        high_result = build_recommendation(high_quality_payload)
        low_result = build_recommendation(low_quality_payload)

        assert high_result["quality_score"] > low_result["quality_score"]


class TestExplanationMetadata:
    """Test explanation metadata."""

    def test_explanation_meta_structure(self):
        """Should have proper meta structure."""
        payload = {
            "ticker": "MSFT",
            "avg_confidence": 0.65,
            "avg_wf_score": 0.55,
            "ensemble_quality": "MIXED",
            "model_votes": [],
        }

        decision = {
            "action": "BUY",
            "strength": "NORMAL",
            "no_trade": False,
            "quality_score": 0.65,
        }

        result = build_explanation(payload, decision)

        assert "meta" in result
        assert "reasons_hu" in result["meta"]
        assert "reasons_en" in result["meta"]
        assert "quality_score" in result["meta"]

    def test_explanation_reasons_bilingual(self):
        """Should have reasons in both languages."""
        payload = {
            "ticker": "JNJ",
            "avg_confidence": 0.30,
            "avg_wf_score": 0.25,
            "ensemble_quality": "CHAOTIC",
            "model_votes": [],
        }

        decision = {
            "action": "HOLD",
            "strength": "NO_TRADE",
            "no_trade": True,
            "no_trade_reason": "LOW_CONFIDENCE",
            "original_action": 1,
            "quality_score": 0.30,
        }

        result = build_explanation(payload, decision)

        # Both languages should have reasons
        assert len(result["meta"]["reasons_hu"]) > 0
        assert len(result["meta"]["reasons_en"]) > 0
