"""
Sprint 3.2 - Correlation Threshold Enforcement Tests

Tests for:
  - High correlation pair detection
  - Reducing allocation for correlated pairs
  - Keeping higher-confidence asset at full allocation
  - Edge cases (no correlation, single ticker)
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from app.decision.allocation import enforce_correlation_limits


class TestCorrelationThresholdEnforcement:
    """Test enforce_correlation_limits function."""

    def test_no_correlated_pairs(self):
        """Verify function handles uncorrelated assets."""
        decisions = [
            {
                "ticker": "VOO",
                "allocation_amount": 50000,
                "allocation_pct": 0.5,
                "decision": {"confidence": 0.8},
            },
            {
                "ticker": "BND",
                "allocation_amount": 50000,
                "allocation_pct": 0.5,
                "decision": {"confidence": 0.75},
            },
        ]

        # Mock low correlation (0.1)
        mock_corr_matrix = pd.DataFrame(
            [[1.0, 0.1], [0.1, 1.0]],
            index=["VOO", "BND"],
            columns=["VOO", "BND"],
        )

        with patch("app.decision.allocation._get_correlation_matrix") as mock_get:
            mock_get.return_value = mock_corr_matrix
            result = enforce_correlation_limits(decisions, max_correlation=0.7)

        # Should be unchanged
        assert result[0]["allocation_amount"] == 50000, "VOO should be unchanged"
        assert result[1]["allocation_amount"] == 50000, "BND should be unchanged"

    def test_high_correlation_pair_reduce_weaker(self):
        """Verify high correlation pair reduces lower-confidence asset."""
        decisions = [
            {
                "ticker": "VOO",
                "allocation_amount": 50000,
                "allocation_pct": 0.5,
                "decision": {"confidence": 0.85},  # Higher confidence
            },
            {
                "ticker": "SPY",
                "allocation_amount": 50000,
                "allocation_pct": 0.5,
                "decision": {"confidence": 0.70},  # Lower confidence
            },
        ]

        # Mock high correlation (0.9)
        mock_corr_matrix = pd.DataFrame(
            [[1.0, 0.9], [0.9, 1.0]],
            index=["VOO", "SPY"],
            columns=["VOO", "SPY"],
        )

        with patch("app.decision.allocation._get_correlation_matrix") as mock_get:
            mock_get.return_value = mock_corr_matrix
            result = enforce_correlation_limits(decisions, max_correlation=0.7)

        # VOO (higher confidence) should stay the same
        assert result[0]["allocation_amount"] == 50000, "VOO should stay unchanged"

        # SPY (lower confidence) should be reduced to 50%
        assert result[1]["allocation_amount"] == 25000, "SPY should be reduced by 50%"
        assert result[1]["allocation_pct"] == 0.25, "SPY pct should be halved"

    def test_high_correlation_multiple_pairs(self):
        """Verify handling of multiple correlated pairs."""
        decisions = [
            {
                "ticker": "VOO",
                "allocation_amount": 40000,
                "allocation_pct": 0.4,
                "decision": {"confidence": 0.85},
            },
            {
                "ticker": "SPY",
                "allocation_amount": 30000,
                "allocation_pct": 0.3,
                "decision": {"confidence": 0.70},
            },
            {
                "ticker": "BND",
                "allocation_amount": 30000,
                "allocation_pct": 0.3,
                "decision": {"confidence": 0.75},
            },
        ]

        # VOO-SPY correlated (0.95), BND independent (0.2)
        mock_corr_matrix = pd.DataFrame(
            [[1.0, 0.95, 0.2], [0.95, 1.0, 0.15], [0.2, 0.15, 1.0]],
            index=["VOO", "SPY", "BND"],
            columns=["VOO", "SPY", "BND"],
        )

        with patch("app.decision.allocation._get_correlation_matrix") as mock_get:
            mock_get.return_value = mock_corr_matrix
            result = enforce_correlation_limits(decisions, max_correlation=0.7)

        # VOO stays unchanged
        assert result[0]["allocation_amount"] == 40000, "VOO unchanged"

        # SPY reduced (correlated with VOO and lower confidence)
        assert result[1]["allocation_amount"] == 15000, "SPY reduced by 50%"

        # BND stays unchanged (not highly correlated)
        assert result[2]["allocation_amount"] == 30000, "BND unchanged"

    def test_single_ticker_no_change(self):
        """Verify single ticker is unchanged."""
        decisions = [
            {
                "ticker": "VOO",
                "allocation_amount": 100000,
                "allocation_pct": 1.0,
                "decision": {"confidence": 0.8},
            },
        ]

        result = enforce_correlation_limits(decisions)

        assert result[0]["allocation_amount"] == 100000, "Single ticker unchanged"

    def test_no_allocated_positions(self):
        """Verify function handles no allocated positions."""
        decisions = [
            {
                "ticker": "VOO",
                "allocation_amount": 0,
                "allocation_pct": 0.0,
                "decision": {"confidence": 0.8},
            },
        ]

        result = enforce_correlation_limits(decisions)

        assert result[0]["allocation_amount"] == 0, "No positions unchanged"

    def test_correlation_limit_threshold(self):
        """Verify only pairs above threshold are adjusted."""
        decisions = [
            {
                "ticker": "VOO",
                "allocation_amount": 50000,
                "allocation_pct": 0.5,
                "decision": {"confidence": 0.8},
            },
            {
                "ticker": "SPY",
                "allocation_amount": 50000,
                "allocation_pct": 0.5,
                "decision": {"confidence": 0.75},
            },
        ]

        # Correlation exactly at threshold (0.7)
        mock_corr_matrix = pd.DataFrame(
            [[1.0, 0.7], [0.7, 1.0]],
            index=["VOO", "SPY"],
            columns=["VOO", "SPY"],
        )

        with patch("app.decision.allocation._get_correlation_matrix") as mock_get:
            mock_get.return_value = mock_corr_matrix
            result = enforce_correlation_limits(decisions, max_correlation=0.7)

        # At threshold, should NOT reduce (only > threshold)
        assert result[0]["allocation_amount"] == 50000, "VOO unchanged at threshold"
        assert result[1]["allocation_amount"] == 50000, "SPY unchanged at threshold"

    def test_correlation_limit_above_threshold(self):
        """Verify pairs above threshold are adjusted."""
        decisions = [
            {
                "ticker": "VOO",
                "allocation_amount": 50000,
                "allocation_pct": 0.5,
                "decision": {"confidence": 0.8},
            },
            {
                "ticker": "SPY",
                "allocation_amount": 50000,
                "allocation_pct": 0.5,
                "decision": {"confidence": 0.75},
            },
        ]

        # Correlation above threshold (0.71 > 0.7)
        mock_corr_matrix = pd.DataFrame(
            [[1.0, 0.71], [0.71, 1.0]],
            index=["VOO", "SPY"],
            columns=["VOO", "SPY"],
        )

        with patch("app.decision.allocation._get_correlation_matrix") as mock_get:
            mock_get.return_value = mock_corr_matrix
            result = enforce_correlation_limits(decisions, max_correlation=0.7)

        # Above threshold, should reduce
        assert result[1]["allocation_amount"] == 25000, "SPY reduced above threshold"

    def test_metadata_update_on_adjustment(self):
        """Verify decision metadata is updated when correlation adjustment occurs."""
        decisions = [
            {
                "ticker": "VOO",
                "allocation_amount": 50000,
                "allocation_pct": 0.5,
                "decision": {"confidence": 0.85},
            },
            {
                "ticker": "SPY",
                "allocation_amount": 50000,
                "allocation_pct": 0.5,
                "decision": {"confidence": 0.70},
            },
        ]

        mock_corr_matrix = pd.DataFrame(
            [[1.0, 0.9], [0.9, 1.0]],
            index=["VOO", "SPY"],
            columns=["VOO", "SPY"],
        )

        with patch("app.decision.allocation._get_correlation_matrix") as mock_get:
            mock_get.return_value = mock_corr_matrix
            result = enforce_correlation_limits(decisions, max_correlation=0.7)

        # Check that weaker decision has adjustment flag
        assert (
            result[1]["decision"].get("correlation_adjustment") is True
        ), "SPY decision should have correlation_adjustment flag"


class TestCorrelationLimitIntegration:
    """Test integration with allocation workflow."""

    def test_preserve_zero_allocation(self):
        """Verify zero-allocation positions are preserved."""
        decisions = [
            {
                "ticker": "VOO",
                "allocation_amount": 100000,
                "allocation_pct": 1.0,
                "decision": {"confidence": 0.8},
            },
            {
                "ticker": "SPY",
                "allocation_amount": 0,
                "allocation_pct": 0.0,
                "decision": {"confidence": 0.75},
            },
        ]

        mock_corr_matrix = pd.DataFrame(
            [[1.0, 0.95], [0.95, 1.0]],
            index=["VOO", "SPY"],
            columns=["VOO", "SPY"],
        )

        with patch("app.decision.allocation._get_correlation_matrix") as mock_get:
            mock_get.return_value = mock_corr_matrix
            result = enforce_correlation_limits(decisions, max_correlation=0.7)

        # SPY already has no allocation, should stay 0
        assert result[1]["allocation_amount"] == 0, "Zero allocation preserved"

    def test_error_handling(self):
        """Verify function handles correlation matrix errors gracefully."""
        decisions = [
            {
                "ticker": "VOO",
                "allocation_amount": 50000,
                "allocation_pct": 0.5,
                "decision": {"confidence": 0.8},
            },
        ]

        # Mock an error in correlation matrix retrieval
        with patch("app.decision.allocation._get_correlation_matrix") as mock_get:
            mock_get.side_effect = Exception("Database error")
            result = enforce_correlation_limits(decisions)

        # Should return unchanged decisions (not crash)
        assert result[0]["allocation_amount"] == 50000, "Returns unchanged on error"
