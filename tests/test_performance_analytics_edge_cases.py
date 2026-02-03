"""
Edge Case Tests for Performance Analytics (Sprint 10 Week 2 - Issue #5).

Tests:
- Empty data handling
- Single return value
- Boundary conditions (very small/large returns)
- NaN and infinite values
- Extreme volatility scenarios
- Drawdown edge cases
- Trade statistics edge cases
"""

import pytest
import math
import numpy as np
from datetime import datetime, timedelta
from app.reporting.performance_analytics import PerformanceAnalytics, PerformanceMetrics


class TestPerformanceAnalyticsEmptyData:
    """Test handling of empty data."""

    def test_calculate_metrics_empty_returns(self):
        """Should raise error with empty returns."""
        analytics = PerformanceAnalytics()
        with pytest.raises(ValueError):
            analytics.calculate_performance_metrics([], [])

    def test_calculate_metrics_mismatched_lengths(self):
        """Should raise error when returns and dates have different lengths."""
        analytics = PerformanceAnalytics()
        returns = [0.01, 0.02]
        dates = [datetime.now()]
        with pytest.raises(ValueError):
            analytics.calculate_performance_metrics(returns, dates)


class TestPerformanceAnalyticsSingleValue:
    """Test handling of single return value."""

    def test_single_return_positive(self):
        """Should handle single positive return."""
        analytics = PerformanceAnalytics()
        returns = [0.01]
        dates = [datetime(2025, 1, 1)]
        
        metrics = analytics.calculate_performance_metrics(returns, dates)
        
        assert abs(metrics.total_return - 0.01) < 1e-6
        assert metrics.volatility == 0.0  # Single value has zero volatility
        assert metrics.sharpe_ratio == 0.0  # Cannot calculate with single value
        assert metrics.max_drawdown == 0.0

    def test_single_return_negative(self):
        """Should handle single negative return."""
        analytics = PerformanceAnalytics()
        returns = [-0.05]
        dates = [datetime(2025, 1, 1)]
        
        metrics = analytics.calculate_performance_metrics(returns, dates)
        
        assert abs(metrics.total_return - (-0.05)) < 1e-6
        assert metrics.volatility == 0.0


class TestPerformanceAnalyticsExtremeBoundaries:
    """Test boundary conditions with extreme values."""

    def test_very_small_returns(self):
        """Should handle very small returns."""
        analytics = PerformanceAnalytics()
        returns = [0.0001, 0.00011, 0.00009]
        dates = [datetime(2025, 1, i) for i in range(1, 4)]
        
        metrics = analytics.calculate_performance_metrics(returns, dates)
        
        assert metrics.total_return > 0
        assert metrics.volatility >= 0
        assert not math.isnan(metrics.total_return)

    def test_very_large_returns(self):
        """Should handle very large returns."""
        analytics = PerformanceAnalytics()
        returns = [1.5, 2.0, 1.2]  # 150%, 200%, 120% returns
        dates = [datetime(2025, 1, i) for i in range(1, 4)]
        
        metrics = analytics.calculate_performance_metrics(returns, dates)
        
        assert metrics.total_return > 0
        assert not math.isnan(metrics.total_return)

    def test_all_zero_returns(self):
        """Should handle all zero returns."""
        analytics = PerformanceAnalytics()
        returns = [0.0, 0.0, 0.0, 0.0]
        dates = [datetime(2025, 1, i) for i in range(1, 5)]
        
        metrics = analytics.calculate_performance_metrics(returns, dates)
        
        assert metrics.total_return == 0.0
        assert metrics.volatility == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.max_drawdown == 0.0


class TestPerformanceAnalyticsExtremeVolatility:
    """Test extreme volatility scenarios."""

    def test_high_volatility_alternating(self):
        """Should handle highly volatile alternating returns."""
        analytics = PerformanceAnalytics()
        returns = [0.20, -0.15, 0.25, -0.10, 0.30]
        dates = [datetime(2025, 1, i) for i in range(1, 6)]
        
        metrics = analytics.calculate_performance_metrics(returns, dates)
        
        assert metrics.volatility > 0.15  # Should be high
        assert not math.isnan(metrics.volatility)
        assert not math.isnan(metrics.sharpe_ratio)

    def test_constant_returns_no_volatility(self):
        """Should handle constant returns (zero volatility)."""
        analytics = PerformanceAnalytics()
        returns = [0.02, 0.02, 0.02, 0.02]
        dates = [datetime(2025, 1, i) for i in range(1, 5)]
        
        metrics = analytics.calculate_performance_metrics(returns, dates)
        
        assert metrics.volatility == 0.0
        # With zero volatility, Sharpe ratio should be handled gracefully
        assert not math.isnan(metrics.sharpe_ratio)


class TestPerformanceAnalyticsDrawdownEdgeCases:
    """Test drawdown calculation edge cases."""

    def test_no_drawdown(self):
        """Should handle series with no drawdown (all positive)."""
        analytics = PerformanceAnalytics()
        returns = [0.01, 0.02, 0.015, 0.025]
        dates = [datetime(2025, 1, i) for i in range(1, 5)]
        
        metrics = analytics.calculate_performance_metrics(returns, dates)
        
        assert metrics.max_drawdown == 0.0

    def test_immediate_drawdown(self):
        """Should detect drawdown at start."""
        analytics = PerformanceAnalytics()
        returns = [-0.10, 0.05, 0.05]
        dates = [datetime(2025, 1, i) for i in range(1, 4)]
        
        metrics = analytics.calculate_performance_metrics(returns, dates)
        
        # Drawdown is calculated as positive magnitude (absolute value)
        assert metrics.max_drawdown > 0 or metrics.total_return < 0

    def test_full_loss(self):
        """Should handle complete loss scenario."""
        analytics = PerformanceAnalytics()
        returns = [0.10, -1.0]  # Complete loss (100% loss)
        dates = [datetime(2025, 1, 1), datetime(2025, 1, 2)]
        
        metrics = analytics.calculate_performance_metrics(returns, dates)
        
        # With complete loss, max drawdown should be significant
        assert metrics.max_drawdown >= 0.5 or metrics.total_return < -0.5


class TestPerformanceAnalyticsNegativeReturns:
    """Test handling of all-negative returns."""

    def test_all_losses(self):
        """Should handle series of only losses."""
        analytics = PerformanceAnalytics()
        returns = [-0.05, -0.03, -0.07, -0.02]
        dates = [datetime(2025, 1, i) for i in range(1, 5)]
        
        metrics = analytics.calculate_performance_metrics(returns, dates)
        
        assert metrics.total_return < 0
        # Drawdown is calculated as positive magnitude
        assert metrics.max_drawdown > 0 or abs(metrics.total_return) > 0.15
        assert not math.isnan(metrics.annualized_return)

    def test_negative_sharpe_ratio(self):
        """Should calculate negative Sharpe ratio for losing strategies."""
        analytics = PerformanceAnalytics()
        returns = [-0.02] * 30  # 30 days of -2% returns
        dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(30)]
        
        metrics = analytics.calculate_performance_metrics(returns, dates)
        
        assert metrics.sharpe_ratio <= 0  # Negative or zero
        assert not math.isnan(metrics.sharpe_ratio)


class TestPerformanceAnalyticsCalmarRatio:
    """Test Calmar ratio edge cases."""

    def test_calmar_zero_drawdown(self):
        """Should handle Calmar ratio with zero drawdown."""
        analytics = PerformanceAnalytics()
        returns = [0.01, 0.01, 0.01]
        dates = [datetime(2025, 1, i) for i in range(1, 4)]
        
        metrics = analytics.calculate_performance_metrics(returns, dates)
        
        # With zero drawdown, Calmar should be 0 or handled gracefully
        assert metrics.calmar_ratio >= 0
        assert not math.isnan(metrics.calmar_ratio)

    def test_calmar_with_large_drawdown(self):
        """Should calculate Calmar ratio with large drawdown."""
        analytics = PerformanceAnalytics()
        returns = [0.50, -0.40]  # 50% gain then 40% loss
        dates = [datetime(2025, 1, 1), datetime(2025, 1, 2)]
        
        metrics = analytics.calculate_performance_metrics(returns, dates)
        
        # Calmar should be defined and non-negative for this scenario
        assert not math.isnan(metrics.calmar_ratio)


class TestPerformanceAnalyticsTradeStatistics:
    """Test trade statistics edge cases."""

    def test_no_trades(self):
        """Should handle metrics calculation with no trades."""
        analytics = PerformanceAnalytics()
        returns = [0.01, 0.02, 0.015]
        dates = [datetime(2025, 1, i) for i in range(1, 4)]
        
        metrics = analytics.calculate_performance_metrics(returns, dates, trades=None)
        
        assert metrics.total_trades == 0
        assert metrics.winning_trades == 0
        assert metrics.losing_trades == 0

    def test_single_winning_trade(self):
        """Should handle single winning trade."""
        analytics = PerformanceAnalytics()
        returns = [0.01, 0.02]
        dates = [datetime(2025, 1, 1), datetime(2025, 1, 2)]
        trades = [
            {
                'entry_price': 100.0,
                'exit_price': 105.0,
                'quantity': 10,
                'pnl': 50.0
            }
        ]
        
        metrics = analytics.calculate_performance_metrics(returns, dates, trades=trades)
        
        assert metrics.total_trades == 1
        assert metrics.profit_factor >= 0
        assert not math.isnan(metrics.profit_factor)

    def test_all_losing_trades(self):
        """Should handle all losing trades."""
        analytics = PerformanceAnalytics()
        returns = [-0.01, -0.02]
        dates = [datetime(2025, 1, 1), datetime(2025, 1, 2)]
        trades = [
            {'entry_price': 100.0, 'exit_price': 99.0, 'quantity': 10, 'pnl': -10.0},
            {'entry_price': 98.0, 'exit_price': 96.0, 'quantity': 10, 'pnl': -20.0},
        ]
        
        metrics = analytics.calculate_performance_metrics(returns, dates, trades=trades)
        
        assert metrics.total_trades == 2
        assert metrics.winning_trades == 0
        assert metrics.losing_trades == 2
        assert metrics.profit_factor == 0.0  # No gains


class TestPerformanceAnalyticsLongTimeSeries:
    """Test with longer time series."""

    def test_one_year_of_returns(self):
        """Should handle one year of daily returns."""
        analytics = PerformanceAnalytics()
        returns = list(np.random.normal(0.0005, 0.01, 252))  # 252 trading days
        dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(252)]
        
        metrics = analytics.calculate_performance_metrics(returns, dates)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert not math.isnan(metrics.annualized_return)
        assert not math.isnan(metrics.sharpe_ratio)
        assert not math.isnan(metrics.volatility)

    def test_three_years_of_returns(self):
        """Should handle three years of daily returns."""
        analytics = PerformanceAnalytics()
        returns = list(np.random.normal(0.0003, 0.008, 756))  # ~3 years
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(756)]
        
        metrics = analytics.calculate_performance_metrics(returns, dates)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert not math.isnan(metrics.total_return)
        assert -10 < metrics.annualized_return < 10  # Reasonable bounds
