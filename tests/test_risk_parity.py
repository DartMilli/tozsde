"""
Sprint 3.1 - Risk Parity Allocation Tests

Tests for:
  - Volatility computation from price history
  - Inverse-volatility weight calculation
  - Capital allocation with risk parity
  - Edge cases (no tradeable positions, single ticker)
"""

import pytest
import numpy as np
from unittest.mock import MagicMock
from dataclasses import replace

from app.decision.risk_parity import RiskParityAllocator, apply_risk_parity


class TestRiskParityAllocator:
    """Test RiskParityAllocator class."""

    def test_allocator_initialization(self):
        """Verify allocator initializes with correct lookback_days."""
        allocator = RiskParityAllocator(lookback_days=30)
        assert allocator.lookback_days == 30

    def test_allocator_default_lookback(self):
        """Verify allocator defaults to 60 lookback days."""
        allocator = RiskParityAllocator()
        assert allocator.lookback_days == 60


class TestVolatilityComputation:
    """Test volatility calculation."""

    def test_compute_volatilities_simple(self):
        """Verify volatility computation on constant prices."""
        allocator = RiskParityAllocator()

        # Constant prices = zero volatility
        price_history = {
            "VOO": np.array([100.0] * 100),
            "SPY": np.array([100.0] * 100),
        }

        vols = allocator._compute_volatilities(price_history)

        # With constant prices, returns are 0, std is 0, annualized is 0
        # But code has floor at 0.01
        assert "VOO" in vols
        assert "SPY" in vols
        assert vols["VOO"] == 0.01  # Floor
        assert vols["SPY"] == 0.01

    def test_compute_volatilities_trending_up(self):
        """Verify volatility computation on trending prices."""
        allocator = RiskParityAllocator()

        # Steady trend upward = low volatility
        prices_up = np.array([100.0 + i for i in range(50)])
        # Random walk = higher volatility
        np.random.seed(42)
        prices_random = 100.0 + np.cumsum(np.random.normal(0, 1, 50))

        price_history = {
            "VOO": prices_up,
            "SPY": prices_random,
        }

        vols = allocator._compute_volatilities(price_history)

        # VOO should have lower volatility than SPY
        assert (
            vols["VOO"] < vols["SPY"]
        ), "Trending should have lower vol than random walk"

    def test_compute_volatilities_insufficient_data(self):
        """Verify volatility handling with insufficient data."""
        allocator = RiskParityAllocator()

        # Less than 2 prices
        price_history = {
            "VOO": np.array([100.0]),
        }

        vols = allocator._compute_volatilities(price_history)

        # Should return default high volatility
        assert vols["VOO"] == 0.1, "Insufficient data should get default volatility"


class TestInverseVolatilityWeights:
    """Test inverse-volatility weight calculation."""

    def test_weights_normalize_to_one(self):
        """Verify weights sum to 1.0."""
        allocator = RiskParityAllocator()

        volatilities = {
            "VOO": 0.15,
            "SPY": 0.20,
            "QQQ": 0.25,
        }
        tickers = ["VOO", "SPY", "QQQ"]

        weights = allocator._compute_inverse_volatility_weights(volatilities, tickers)

        assert abs(weights.sum() - 1.0) < 1e-6, "Weights should sum to 1.0"

    def test_weights_inverse_relationship(self):
        """Verify lower volatility gets higher weight."""
        allocator = RiskParityAllocator()

        volatilities = {
            "VOO": 0.10,  # Low vol = should get ~50%
            "SPY": 0.20,  # High vol = should get ~25%
        }
        tickers = ["VOO", "SPY"]

        weights = allocator._compute_inverse_volatility_weights(volatilities, tickers)

        # VOO (lower vol) should get more weight than SPY
        voo_weight = weights[0]
        spy_weight = weights[1]

        assert voo_weight > spy_weight, "Lower volatility should get higher weight"
        assert voo_weight > 0.5, "Low-vol asset should get >50% in 2-asset portfolio"

    def test_weights_equal_volatility(self):
        """Verify equal weights when volatility is equal."""
        allocator = RiskParityAllocator()

        volatilities = {
            "VOO": 0.15,
            "SPY": 0.15,
            "QQQ": 0.15,
        }
        tickers = ["VOO", "SPY", "QQQ"]

        weights = allocator._compute_inverse_volatility_weights(volatilities, tickers)

        # Should be ~1/3 each
        expected = 1.0 / 3
        for weight in weights:
            assert abs(weight - expected) < 0.01, "Equal vols should get equal weights"


class TestAllocationApplication:
    """Test capital allocation application."""

    def test_apply_allocation_amounts(self, test_settings):
        """Verify allocation amounts are applied to decisions."""
        settings = replace(test_settings, INITIAL_CAPITAL=100000)
        allocator = RiskParityAllocator(settings=settings)

        decisions = [
            {"ticker": "VOO", "action_code": 1, "action": "BUY", "confidence": 0.8},
            {"ticker": "SPY", "action_code": 1, "action": "BUY", "confidence": 0.75},
        ]

        weights = np.array([0.6, 0.4])
        tradeable_indices = [0, 1]

        result = allocator._apply_allocation(decisions, weights, tradeable_indices)

        # Capital * 0.95 * weight = allocation
        capital = 100000 * 0.95

        assert "allocation_amount" in result[0], "Should have allocation_amount"
        assert "allocation_pct" in result[0], "Should have allocation_pct"

        # Check amounts
        expected_voo = capital * 0.6
        assert (
            abs(result[0]["allocation_amount"] - expected_voo) < 1
        ), "VOO allocation mismatch"

    def test_apply_allocation_preserves_others(self):
        """Verify allocation doesn't modify non-tradeable fields."""
        allocator = RiskParityAllocator()

        decisions = [
            {
                "ticker": "VOO",
                "action_code": 1,
                "action": "BUY",
                "confidence": 0.8,
                "custom_field": "should_persist",
            },
        ]

        weights = np.array([1.0])
        tradeable_indices = [0]

        result = allocator._apply_allocation(decisions, weights, tradeable_indices)

        assert (
            result[0]["custom_field"] == "should_persist"
        ), "Custom fields should persist"


class TestFullAllocation:
    """Test full allocation workflow."""

    def test_allocate_no_tradeable(self):
        """Verify allocation handles no tradeable positions."""
        allocator = RiskParityAllocator()

        decisions = [
            {"ticker": "VOO", "action_code": 0, "action": "HOLD"},  # No trade
        ]

        price_history = {
            "VOO": np.array([100.0 + i for i in range(50)]),
        }

        result = allocator.allocate(decisions, price_history)

        # Should return unchanged
        assert result[0].get("allocation_amount") is None, "No allocation for no-trade"

    def test_allocate_multiple_tickers(self, test_settings):
        """Verify full allocation with multiple tickers."""
        settings = replace(test_settings, INITIAL_CAPITAL=100000)
        allocator = RiskParityAllocator(settings=settings)

        decisions = [
            {"ticker": "VOO", "action_code": 1, "action": "BUY", "confidence": 0.8},
            {"ticker": "SPY", "action_code": 1, "action": "BUY", "confidence": 0.75},
            {"ticker": "QQQ", "action_code": 1, "action": "BUY", "confidence": 0.7},
        ]

        # Create price histories with different volatilities
        np.random.seed(42)
        price_history = {
            "VOO": 100.0 + np.cumsum(np.random.normal(0, 0.5, 50)),  # Low vol
            "SPY": 100.0 + np.cumsum(np.random.normal(0, 1.0, 50)),  # Medium vol
            "QQQ": 100.0 + np.cumsum(np.random.normal(0, 1.5, 50)),  # High vol
        }

        result = allocator.allocate(decisions, price_history)

        # All should have allocations
        assert result[0]["allocation_amount"] > 0
        assert result[1]["allocation_amount"] > 0
        assert result[2]["allocation_amount"] > 0

        # VOO (lowest vol) should get largest allocation
        assert result[0]["allocation_amount"] > result[2]["allocation_amount"]


class TestConvenienceFunction:
    """Test apply_risk_parity convenience function."""

    def test_apply_risk_parity_function(self, test_settings, monkeypatch):
        """Verify apply_risk_parity convenience function works."""
        decisions = [
            {"ticker": "VOO", "action_code": 1, "action": "BUY", "confidence": 0.8},
        ]

        price_history = {
            "VOO": np.array([100.0 + i for i in range(50)]),
        }

        settings = replace(test_settings, INITIAL_CAPITAL=100000)
        monkeypatch.setattr("app.decision.risk_parity.build_settings", lambda: settings)
        result = apply_risk_parity(decisions, price_history)

        assert "allocation_amount" in result[0], "Should apply allocation"
        assert result[0]["allocation_amount"] > 0, "Allocation should be positive"
