"""
Unit tests for fitness functions and genetic optimizer.

Tests:
- Fitness function calculations
- Edge cases (zero trades, all losses)
- Overfitting penalties
- Walk-forward fitness aggregation
"""

import pytest
import numpy as np
import pandas as pd
from app.optimization.fitness import fitness_walk_forward, fitness_single
from app.reporting.metrics import WalkForwardMetrics


class MockMetrics:
    """Mock metrics object for testing."""

    def __init__(
        self,
        trade_count=50,
        net_profit=1000.0,
        max_drawdown=0.1,
        winrate=0.55,
        equity_curve=None,
    ):
        self.trade_count = trade_count
        self.net_profit = net_profit
        self.max_drawdown = max_drawdown
        self.winrate = winrate
        self.equity_curve = equity_curve or [1000.0, 1010.0, 1020.0]


class TestFitnessFunction:
    """Tests for single-period fitness calculation."""

    def test_fitness_zero_trades(self):
        """Fitness with few trades should return negative infinity."""
        metrics = MockMetrics(trade_count=5, net_profit=1000.0)
        result = fitness_single(metrics)
        # Should penalize low trade count (minimum 30 trades required)
        assert result == -1e12 or result < -1000

    def test_fitness_positive_return(self):
        """Fitness should be positive for profitable trades."""
        metrics = MockMetrics(
            trade_count=50, net_profit=1000.0, max_drawdown=0.1, winrate=0.6
        )
        result = fitness_single(metrics)
        # Should be positive with good metrics
        assert result > 0

    def test_fitness_negative_return(self):
        """Fitness should penalize losses."""
        metrics = MockMetrics(
            trade_count=50, net_profit=-500.0, max_drawdown=0.3, winrate=0.4
        )
        result = fitness_single(metrics)
        # Should be very negative with losses
        assert result < -500

    def test_fitness_high_sharpe(self):
        """Fitness should reward high Sharpe ratio (high winrate)."""
        high_winrate = MockMetrics(
            trade_count=50, net_profit=1000.0, max_drawdown=0.05, winrate=0.75
        )
        low_winrate = MockMetrics(
            trade_count=50, net_profit=1000.0, max_drawdown=0.05, winrate=0.51
        )

        fitness_high = fitness_single(high_winrate)
        fitness_low = fitness_single(low_winrate)

        # Higher winrate should produce higher fitness
        assert fitness_high > fitness_low


class TestWalkForwardFitness:
    """Tests for walk-forward fitness aggregation."""

    def test_wf_fitness_basic(self):
        """Walk-forward fitness should aggregate window results."""
        # Create a mock walk-forward metrics object
        wf_metrics = WalkForwardMetrics(
            avg_profit=500.0,
            avg_dd=0.1,
            profit_std=50.0,
            dd_std=0.01,
            negative_fold_ratio=0.2,
        )

        result = fitness_walk_forward(wf_metrics)
        # Should be positive with good aggregated metrics
        assert isinstance(result, float)
        assert result > -1e12  # Not completely penalized

    def test_wf_fitness_consistency(self):
        """Same inputs should produce same fitness (deterministic)."""
        wf_metrics = WalkForwardMetrics(
            avg_profit=500.0,
            avg_dd=0.1,
            profit_std=50.0,
            dd_std=0.01,
            negative_fold_ratio=0.2,
        )

        result1 = fitness_walk_forward(wf_metrics)
        result2 = fitness_walk_forward(wf_metrics)

        # Must be deterministic
        assert result1 == result2

    def test_wf_fitness_degradation_penalty(self):
        """Degrading performance across windows should reduce fitness."""
        # Good consistency
        good_metrics = WalkForwardMetrics(
            avg_profit=500.0,
            avg_dd=0.1,
            profit_std=10.0,  # Low std = consistent
            dd_std=0.005,
            negative_fold_ratio=0.1,
        )

        # Poor consistency (high variance)
        poor_metrics = WalkForwardMetrics(
            avg_profit=500.0,
            avg_dd=0.1,
            profit_std=200.0,  # High std = degrading
            dd_std=0.05,
            negative_fold_ratio=0.4,
        )

        fitness_good = fitness_walk_forward(good_metrics)
        fitness_poor = fitness_walk_forward(poor_metrics)

        # Consistent performance should be better than degrading
        assert fitness_good > fitness_poor


class TestOverfittingPenalty:
    """Tests for overfitting detection and penalties."""

    def test_overfitting_penalty_applied(self):
        """High variance across windows should reduce fitness."""
        wf_metrics = WalkForwardMetrics(
            avg_profit=500.0,
            avg_dd=0.1,
            profit_std=300.0,  # Very high variance
            dd_std=0.1,
            negative_fold_ratio=0.5,
        )

        result = fitness_walk_forward(wf_metrics)
        # High overfitting should reduce fitness significantly
        assert result < 200  # Much lower than avg_profit

    def test_overfitting_threshold(self):
        """High negative fold ratio should heavily penalize fitness."""
        wf_metrics = WalkForwardMetrics(
            avg_profit=500.0,
            avg_dd=0.1,
            profit_std=50.0,
            dd_std=0.01,
            negative_fold_ratio=0.6,  # More than 50% folds negative
        )

        result = fitness_walk_forward(wf_metrics)
        # Should return negative infinity when too many negative folds
        assert result == -1e12
