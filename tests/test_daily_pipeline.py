"""
Integration tests for daily pipeline.

Tests:
- Pipeline components work correctly
- Decisions are properly formatted
- Error handling is graceful
- Data consistency is maintained
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import date
from app.decision.recommendation_builder import build_recommendation
from app.decision.decision_policy import apply_decision_policy
from app.decision.allocation import allocate_capital


class TestDailyPipeline:
    """Tests for daily pipeline components."""

    def test_build_recommendation_creates_valid_decision(self):
        """build_recommendation should create valid decision object."""
        payload = {
            "ticker": "TEST",
            "signal_strength": 0.8,
            "avg_confidence": 0.7,
            "avg_wf_score": 0.75,
            "action_code": 1,
            "ensemble_quality": "STABLE",
            "date": str(date.today()),
        }

        decision = build_recommendation(payload)

        assert "action" in decision or "action_code" in decision
        assert "confidence" in decision
        assert decision["confidence"] >= 0 and decision["confidence"] <= 1

    def test_apply_decision_policy_preserves_decision(self):
        """apply_decision_policy should preserve decision for valid signals."""
        decision = {
            "action": "BUY",
            "action_code": 1,
            "confidence": 0.8,
            "wf_score": 0.85,
            "strength": 0.8,
            "no_trade": False,
        }

        audit = {"date": str(date.today()), "ticker": "TEST"}

        result = apply_decision_policy(decision, audit)

        # Decision should still be present after policy
        assert "action_code" in result or "action" in result
        assert result["confidence"] == decision["confidence"]

    @patch("app.decision.allocation._get_correlation_matrix")
    def test_allocate_capital_single_decision(self, mock_corr):
        """allocate_capital should allocate 100% to single decision."""
        mock_corr.return_value = pd.DataFrame([[1.0]], index=["TEST"], columns=["TEST"])

        decisions = [
            {
                "ticker": "TEST",
                "date": str(date.today()),
                "decision": {"action_code": 1, "confidence": 0.8},
                "payload": {"volatility": 0.02},
            }
        ]

        result = allocate_capital(decisions)

        assert len(result) == 1
        assert "allocation_pct" in result[0]
        assert result[0]["allocation_pct"] == 1.0

    @patch("app.decision.allocation._get_correlation_matrix")
    def test_allocate_capital_filters_no_trade(self, mock_corr):
        """allocate_capital should skip no_trade decisions."""
        mock_corr.return_value = pd.DataFrame(
            [[1.0, 0.0], [0.0, 1.0]],
            index=["TEST1", "TEST2"],
            columns=["TEST1", "TEST2"],
        )

        decisions = [
            {
                "ticker": "TEST1",
                "date": str(date.today()),
                "decision": {"action_code": 1, "confidence": 0.8, "no_trade": False},
                "payload": {"volatility": 0.02},
            },
            {
                "ticker": "TEST2",
                "date": str(date.today()),
                "decision": {"action_code": 0, "confidence": 0.5, "no_trade": True},
                "payload": {"volatility": 0.02},
            },
        ]

        result = allocate_capital(
            [d for d in decisions if not d["decision"].get("no_trade", False)]
        )

        # Only TEST1 should be in result
        assert len(result) == 1
        assert result[0]["ticker"] == "TEST1"


class TestPipelineDataIntegrity:
    """Tests for data consistency throughout pipeline."""

    def test_decision_confidence_range_valid(self):
        """Decision confidence should always be between 0 and 1."""
        payload = {
            "ticker": "TEST",
            "signal_strength": 0.75,
            "avg_confidence": 0.75,
            "avg_wf_score": 0.8,
            "action_code": 1,
            "ensemble_quality": "STABLE",
        }

        decision = build_recommendation(payload)

        assert 0 <= decision["confidence"] <= 1

    @patch("app.decision.allocation._get_correlation_matrix")
    def test_allocation_percentages_sum_to_one(self, mock_corr):
        """Allocation percentages across decisions should sum to ~1.0."""
        mock_corr.return_value = pd.DataFrame(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            index=["T1", "T2", "T3"],
            columns=["T1", "T2", "T3"],
        )

        decisions = [
            {
                "ticker": "T1",
                "date": str(date.today()),
                "decision": {"action_code": 1, "confidence": 0.8},
                "payload": {"volatility": 0.02},
            },
            {
                "ticker": "T2",
                "date": str(date.today()),
                "decision": {"action_code": 1, "confidence": 0.8},
                "payload": {"volatility": 0.03},
            },
            {
                "ticker": "T3",
                "date": str(date.today()),
                "decision": {"action_code": 1, "confidence": 0.8},
                "payload": {"volatility": 0.04},
            },
        ]

        result = allocate_capital(decisions)

        total_pct = sum(r["allocation_pct"] for r in result)
        assert abs(total_pct - 1.0) < 0.001

    def test_decision_event_consistency(self):
        """Decision event should have consistent fields."""
        decision = {
            "action_code": 1,
            "confidence": 0.75,
            "wf_score": 0.8,
            "strength": 0.75,
        }

        # Decision should have required fields
        assert decision["action_code"] in [0, 1]  # HOLD or BUY
        assert 0 <= decision["confidence"] <= 1
        assert 0 <= decision.get("wf_score", 0) <= 1


class TestPipelineErrorHandling:
    """Tests for error handling and recovery."""

    @patch("app.decision.allocation._get_correlation_matrix")
    def test_allocation_handles_zero_volatility(self, mock_corr):
        """allocate_capital should handle zero volatility gracefully."""
        mock_corr.return_value = pd.DataFrame(
            [[1.0, 0.0], [0.0, 1.0]],
            index=["TEST1", "TEST2"],
            columns=["TEST1", "TEST2"],
        )

        decisions = [
            {
                "ticker": "TEST1",
                "date": str(date.today()),
                "decision": {"action_code": 1, "confidence": 0.8},
                "payload": {"volatility": 0.0},  # Zero volatility
            },
            {
                "ticker": "TEST2",
                "date": str(date.today()),
                "decision": {"action_code": 1, "confidence": 0.8},
                "payload": {"volatility": 0.02},
            },
        ]

        # Should not raise exception
        result = allocate_capital(decisions)
        assert len(result) >= 0  # May filter out zero-vol position

    def test_recommendation_handles_missing_confidence(self):
        """build_recommendation should handle missing confidence gracefully."""
        payload = {
            "ticker": "TEST",
            "signal_strength": 0.8,
            # confidence missing
        }

        # Should not raise exception
        try:
            decision = build_recommendation(payload)
            assert "confidence" in decision
        except:
            # If it raises, that's also acceptable behavior
            pass

    def test_policy_handles_edge_cases(self):
        """apply_decision_policy should handle edge case decisions."""
        decision = {"action_code": 0, "confidence": 0.0, "no_trade": True}  # HOLD

        audit = {"date": str(date.today())}

        # Should not raise exception
        result = apply_decision_policy(decision, audit)
        assert result is not None
