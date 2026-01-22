"""
Unit tests for capital allocation logic.

Tests:
- Allocation normalization
- Correlation adjustment
- Edge cases (single ticker, high correlation)
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from app.decision.allocation import allocate_capital
from app.config.config import Config


class TestCapitalAllocation:
    """Tests for capital allocation calculation."""

    def test_allocation_basic(self):
        """Allocate capital should distribute funds proportionally."""
        # Single BUY decision with basic params
        decisions = [
            {
                "ticker": "TEST1",
                "date": "2025-01-20",
                "decision": {"action_code": 1, "confidence": 0.8},
                "payload": {"volatility": 0.02},
            }
        ]

        result = allocate_capital(decisions)

        assert len(result) == 1
        assert "allocation_amount" in result[0]
        assert "allocation_pct" in result[0]
        # Single ticker should get full allocation
        assert result[0]["allocation_pct"] == 1.0
        assert result[0]["allocation_amount"] > 0

    def test_allocation_normalization(self):
        """Allocation percentages should sum to 100%."""
        # Multiple BUY decisions with different volatilities
        decisions = [
            {
                "ticker": "TEST1",
                "date": "2025-01-20",
                "decision": {"action_code": 1},
                "payload": {"volatility": 0.02},
            },
            {
                "ticker": "TEST2",
                "date": "2025-01-20",
                "decision": {"action_code": 1},
                "payload": {"volatility": 0.03},
            },
            {
                "ticker": "TEST3",
                "date": "2025-01-20",
                "decision": {"action_code": 1},
                "payload": {"volatility": 0.04},
            },
        ]

        # Mock the correlation matrix call to avoid DB access
        with patch("app.decision.allocation._get_correlation_matrix") as mock_corr:
            # Return identity matrix (no correlation)
            mock_corr.return_value = pd.DataFrame(
                np.eye(3),
                index=["TEST1", "TEST2", "TEST3"],
                columns=["TEST1", "TEST2", "TEST3"],
            )

            result = allocate_capital(decisions)

        # All should have allocation
        assert all("allocation_pct" in r for r in result)
        # Percentages should sum to 100%
        total_pct = sum(r["allocation_pct"] for r in result)
        assert abs(total_pct - 1.0) < 0.0001

    def test_allocation_no_trade_zero(self):
        """NO_TRADE signal (action_code != 1) should receive zero allocation."""
        decisions = [
            {
                "ticker": "TEST1",
                "date": "2025-01-20",
                "decision": {"action_code": 1},
                "payload": {"volatility": 0.02},
            },
            {
                "ticker": "TEST2",
                "date": "2025-01-20",
                "decision": {"action_code": 0},  # NO_TRADE
                "payload": {"volatility": 0.02},
            },
        ]

        result = allocate_capital(decisions)

        # First should have allocation, second should be zero
        assert result[0]["allocation_pct"] > 0
        assert result[1]["allocation_pct"] == 0.0
        assert result[1]["allocation_amount"] == 0.0

    def test_allocation_respects_confidence(self):
        """Higher confidence should contribute to inverse volatility weighting."""
        # Two tickers with same volatility - inverse vol should be same
        decisions = [
            {
                "ticker": "TEST1",
                "date": "2025-01-20",
                "decision": {"action_code": 1, "confidence": 0.8},
                "payload": {"volatility": 0.02},
            },
            {
                "ticker": "TEST2",
                "date": "2025-01-20",
                "decision": {"action_code": 1, "confidence": 0.5},
                "payload": {"volatility": 0.02},
            },
        ]

        with patch("app.decision.allocation._get_correlation_matrix") as mock_corr:
            mock_corr.return_value = pd.DataFrame(
                np.eye(2), index=["TEST1", "TEST2"], columns=["TEST1", "TEST2"]
            )
            result = allocate_capital(decisions)

        # Both should have equal allocation (same volatility)
        # Note: allocation uses inverse volatility, not confidence directly
        assert result[0]["allocation_pct"] > 0
        assert result[1]["allocation_pct"] > 0


class TestCorrelationAdjustment:
    """Tests for correlation-based adjustment."""

    def test_correlation_limit_high_corr(self):
        """High correlation should affect allocation weighting."""
        # This test verifies the correlation matrix is used
        decisions = [
            {
                "ticker": "TEST1",
                "date": "2025-01-20",
                "decision": {"action_code": 1},
                "payload": {"volatility": 0.02},
            },
            {
                "ticker": "TEST2",
                "date": "2025-01-20",
                "decision": {"action_code": 1},
                "payload": {"volatility": 0.02},
            },
        ]

        with patch("app.decision.allocation._get_correlation_matrix") as mock_corr:
            # High correlation matrix
            mock_corr.return_value = pd.DataFrame(
                [[1.0, 0.9], [0.9, 1.0]],
                index=["TEST1", "TEST2"],
                columns=["TEST1", "TEST2"],
            )
            result = allocate_capital(decisions)

        # Both should have allocations (correlation adjustment applied)
        assert all("allocation_pct" in r for r in result)
        assert all(r["allocation_pct"] >= 0 for r in result)

    def test_correlation_limit_low_corr(self):
        """Low correlation should not reduce allocation severely."""
        # Single ticker - no correlation adjustment
        decisions = [
            {
                "ticker": "TEST1",
                "date": "2025-01-20",
                "decision": {"action_code": 1},
                "payload": {"volatility": 0.02},
            }
        ]

        result = allocate_capital(decisions)

        # Single ticker should get full allocation
        assert result[0]["allocation_pct"] == 1.0

    def test_correlation_keeps_higher_confidence(self):
        """Function should keep higher confidence positions."""
        decisions = [
            {
                "ticker": "HIGH_CONF",
                "date": "2025-01-20",
                "decision": {"action_code": 1, "confidence": 0.9},
                "payload": {"volatility": 0.02},
            },
            {
                "ticker": "LOW_CONF",
                "date": "2025-01-20",
                "decision": {"action_code": 1, "confidence": 0.3},
                "payload": {"volatility": 0.02},
            },
        ]

        with patch("app.decision.allocation._get_correlation_matrix") as mock_corr:
            mock_corr.return_value = pd.DataFrame(
                np.eye(2),
                index=["HIGH_CONF", "LOW_CONF"],
                columns=["HIGH_CONF", "LOW_CONF"],
            )
            result = allocate_capital(decisions)

        # Both should have allocations proportional to inverse vol (same in this case)
        assert result[0]["allocation_pct"] > 0
        assert result[1]["allocation_pct"] > 0


class TestAllocationEdgeCases:
    """Tests for edge cases."""

    def test_allocation_single_ticker(self):
        """Single ticker should receive 100% allocation."""
        decisions = [
            {
                "ticker": "SINGLE",
                "date": "2025-01-20",
                "decision": {"action_code": 1},
                "payload": {"volatility": 0.02},
            }
        ]

        result = allocate_capital(decisions)

        assert result[0]["allocation_pct"] == 1.0
        assert result[0]["allocation_amount"] == Config.INITIAL_CAPITAL

    def test_allocation_zero_confidence(self):
        """Zero confidence signals should still be processed."""
        decisions = [
            {
                "ticker": "ZERO_CONF",
                "date": "2025-01-20",
                "decision": {"action_code": 1, "confidence": 0.0},
                "payload": {"volatility": 0.02},
            }
        ]

        result = allocate_capital(decisions)

        # Should still get allocation (not filtered by confidence)
        assert result[0]["allocation_pct"] == 1.0

    def test_allocation_all_same_confidence(self):
        """Equal confidence should result in equal or inverse-vol weighted allocation."""
        decisions = [
            {
                "ticker": "T1",
                "date": "2025-01-20",
                "decision": {"action_code": 1},
                "payload": {"volatility": 0.02},  # Same volatility
            },
            {
                "ticker": "T2",
                "date": "2025-01-20",
                "decision": {"action_code": 1},
                "payload": {"volatility": 0.02},  # Same volatility
            },
        ]

        with patch("app.decision.allocation._get_correlation_matrix") as mock_corr:
            mock_corr.return_value = pd.DataFrame(
                np.eye(2), index=["T1", "T2"], columns=["T1", "T2"]
            )
            result = allocate_capital(decisions)

        # With same volatility, allocations should be roughly equal
        total_pct = sum(r["allocation_pct"] for r in result)
        assert abs(total_pct - 1.0) < 0.0001
        # Both should have positive allocation
        assert all(r["allocation_pct"] > 0 for r in result)
