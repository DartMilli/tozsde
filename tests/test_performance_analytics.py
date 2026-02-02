"""
Tests for PerformanceAnalytics module.
"""

import pytest
from datetime import datetime, timedelta
from app.reporting.performance_analytics import (
    PerformanceAnalytics,
    PerformanceMetrics,
    DrawdownAnalysis,
    RollingMetrics
)


@pytest.fixture
def analytics():
    """Create a test analytics instance."""
    return PerformanceAnalytics(risk_free_rate=0.02)


@pytest.fixture
def sample_returns():
    """Create sample return data."""
    # 20 days of returns
    returns = [0.01, -0.005, 0.015, -0.01, 0.02, 0.005, -0.015, 0.01, 0.008, -0.012,
               0.01, 0.005, -0.008, 0.015, -0.005, 0.01, 0.012, -0.01, 0.005, 0.015]
    dates = [datetime.now() - timedelta(days=20-i) for i in range(20)]
    return returns, dates


class TestPerformanceMetricsCalculation:
    """Test performance metrics calculation."""
    
    def test_calculate_metrics_basic(self, analytics, sample_returns):
        """Test basic metrics calculation."""
        returns, dates = sample_returns
        
        metrics = analytics.calculate_performance_metrics(returns, dates)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_return != 0
        assert metrics.annualized_return is not None
        assert metrics.volatility >= 0
        assert metrics.max_drawdown >= 0
    
    def test_metrics_with_trades(self, analytics, sample_returns):
        """Test metrics calculation with trade data."""
        returns, dates = sample_returns
        trades = [
            {'pnl': 100},
            {'pnl': -50},
            {'pnl': 150},
            {'pnl': -30}
        ]
        
        metrics = analytics.calculate_performance_metrics(returns, dates, trades)
        
        assert metrics.total_trades == 4
        assert metrics.winning_trades == 2
        assert metrics.losing_trades == 2
        assert metrics.win_rate == 0.5
    
    def test_empty_returns_raises_error(self, analytics):
        """Test error on empty returns."""
        with pytest.raises(ValueError):
            analytics.calculate_performance_metrics([], [])
    
    def test_mismatched_lengths_raises_error(self, analytics):
        """Test error on mismatched array lengths."""
        with pytest.raises(ValueError):
            analytics.calculate_performance_metrics([0.01, 0.02], [datetime.now()])


class TestReturnCalculations:
    """Test return calculation methods."""
    
    def test_total_return_positive(self, analytics):
        """Test total return calculation with positive returns."""
        returns = [0.10, 0.05, 0.08]
        total = analytics._calculate_total_return(returns)
        
        # (1.10 * 1.05 * 1.08) - 1 = 0.2484
        assert total == pytest.approx(0.2484, rel=0.01)
    
    def test_total_return_mixed(self, analytics):
        """Test total return with mixed returns."""
        returns = [0.10, -0.05, 0.08]
        total = analytics._calculate_total_return(returns)
        
        assert total > 0  # Net positive
    
    def test_annualized_return(self, analytics):
        """Test annualized return calculation."""
        returns = [0.01] * 365  # 1% daily for a year
        dates = [datetime.now() - timedelta(days=365-i) for i in range(365)]
        
        annualized = analytics._calculate_annualized_return(returns, dates)
        
        assert annualized > 3.0  # Should be significantly positive


class TestVolatilityCalculation:
    """Test volatility calculation."""
    
    def test_volatility_positive(self, analytics):
        """Test volatility is always positive."""
        returns = [0.01, -0.01, 0.02, -0.02, 0.015]
        dates = [datetime.now() - timedelta(days=5-i) for i in range(5)]
        
        vol = analytics._calculate_volatility(returns, dates)
        
        assert vol > 0
    
    def test_volatility_increases_with_variance(self, analytics):
        """Test higher variance increases volatility."""
        low_var_returns = [0.001] * 10
        high_var_returns = [0.05, -0.05, 0.06, -0.06, 0.04, -0.04, 0.05, -0.05, 0.03, -0.03]
        dates = [datetime.now() - timedelta(days=10-i) for i in range(10)]
        
        low_vol = analytics._calculate_volatility(low_var_returns, dates)
        high_vol = analytics._calculate_volatility(high_var_returns, dates)
        
        assert high_vol > low_vol


class TestRiskAdjustedRatios:
    """Test risk-adjusted ratio calculations."""
    
    def test_sharpe_ratio_positive(self, analytics, sample_returns):
        """Test Sharpe ratio with positive returns."""
        returns, dates = sample_returns
        
        sharpe = analytics._calculate_sharpe_ratio(returns, dates)
        
        assert isinstance(sharpe, float)
    
    def test_sharpe_zero_volatility(self, analytics):
        """Test Sharpe ratio with zero volatility."""
        returns = [0.0] * 10
        dates = [datetime.now() - timedelta(days=10-i) for i in range(10)]
        
        sharpe = analytics._calculate_sharpe_ratio(returns, dates)
        
        assert sharpe == 0.0
    
    def test_sortino_ratio(self, analytics, sample_returns):
        """Test Sortino ratio calculation."""
        returns, dates = sample_returns
        
        sortino = analytics._calculate_sortino_ratio(returns, dates)
        
        assert isinstance(sortino, float)
    
    def test_sortino_only_positive_returns(self, analytics):
        """Test Sortino ratio with only positive returns."""
        returns = [0.01, 0.02, 0.015, 0.01]
        dates = [datetime.now() - timedelta(days=4-i) for i in range(4)]
        
        sortino = analytics._calculate_sortino_ratio(returns, dates)
        
        # Should be very high or infinite
        assert sortino > 0 or sortino == float('inf')


class TestDrawdownAnalysis:
    """Test drawdown analysis."""
    
    def test_drawdown_analysis_basic(self, analytics, sample_returns):
        """Test basic drawdown analysis."""
        returns, dates = sample_returns
        
        dd_analysis = analytics.analyze_drawdowns(returns, dates)
        
        assert isinstance(dd_analysis, DrawdownAnalysis)
        assert dd_analysis.max_drawdown >= 0
        assert dd_analysis.current_drawdown >= 0
    
    def test_no_drawdown_all_positive(self, analytics):
        """Test drawdown with all positive returns."""
        returns = [0.01] * 10
        dates = [datetime.now() - timedelta(days=10-i) for i in range(10)]
        
        dd_analysis = analytics.analyze_drawdowns(returns, dates)
        
        assert dd_analysis.max_drawdown == 0.0
        assert len(dd_analysis.drawdowns) == 0
    
    def test_drawdown_detection(self, analytics):
        """Test drawdown detection."""
        # Create returns with clear drawdown
        returns = [0.05, 0.05, -0.10, -0.05, -0.05, 0.15, 0.10]
        dates = [datetime.now() - timedelta(days=7-i) for i in range(7)]
        
        dd_analysis = analytics.analyze_drawdowns(returns, dates)
        
        assert dd_analysis.max_drawdown > 0
        assert len(dd_analysis.drawdowns) > 0


class TestRollingMetrics:
    """Test rolling window metrics."""
    
    def test_rolling_metrics_basic(self, analytics):
        """Test basic rolling metrics calculation."""
        returns = [0.01] * 50
        dates = [datetime.now() - timedelta(days=50-i) for i in range(50)]
        
        rolling = analytics.calculate_rolling_metrics(returns, dates, window_days=10)
        
        assert isinstance(rolling, RollingMetrics)
        assert rolling.window_size_days == 10
        assert len(rolling.returns) > 0
        assert len(rolling.volatilities) > 0
        assert len(rolling.sharpe_ratios) > 0
    
    def test_rolling_insufficient_data(self, analytics):
        """Test rolling metrics with insufficient data."""
        returns = [0.01] * 5
        dates = [datetime.now() - timedelta(days=5-i) for i in range(5)]
        
        rolling = analytics.calculate_rolling_metrics(returns, dates, window_days=10)
        
        assert len(rolling.returns) == 0  # Not enough data
    
    def test_rolling_window_alignment(self, analytics):
        """Test rolling window data alignment."""
        returns = [0.01] * 40
        dates = [datetime.now() - timedelta(days=40-i) for i in range(40)]
        
        rolling = analytics.calculate_rolling_metrics(returns, dates, window_days=20)
        
        # Should have 21 rolling windows (40 - 20 + 1)
        assert len(rolling.returns) == 21
        assert len(rolling.dates) == 21


class TestTradeStatistics:
    """Test trade statistics calculations."""
    
    def test_trade_statistics_basic(self, analytics):
        """Test basic trade statistics."""
        trades = [
            {'pnl': 100},
            {'pnl': -50},
            {'pnl': 150},
            {'pnl': -30},
            {'pnl': 80}
        ]
        
        win_rate, profit_factor, total, winning, losing = \
            analytics._calculate_trade_statistics(trades)
        
        assert total == 5
        assert winning == 3
        assert losing == 2
        assert win_rate == 0.6
        assert profit_factor > 1.0  # Net profitable
    
    def test_trade_extremes(self, analytics):
        """Test trade extremes calculation."""
        trades = [
            {'pnl': 100},
            {'pnl': -50},
            {'pnl': 200},
            {'pnl': -20}
        ]
        
        avg_win, avg_loss, best, worst = analytics._calculate_trade_extremes(trades)
        
        assert avg_win == 150.0  # (100 + 200) / 2
        assert avg_loss == -35.0  # (-50 + -20) / 2
        assert best == 200
        assert worst == -50
    
    def test_empty_trades(self, analytics):
        """Test trade statistics with empty trades."""
        win_rate, profit_factor, total, winning, losing = \
            analytics._calculate_trade_statistics([])
        
        assert total == 0
        assert winning == 0
        assert losing == 0
        assert win_rate == 0.0


class TestMaxDrawdownCalculation:
    """Test maximum drawdown calculation."""
    
    def test_max_drawdown_basic(self, analytics):
        """Test maximum drawdown calculation."""
        # Create returns with known drawdown
        returns = [0.10, 0.05, -0.15, -0.10, 0.20]
        
        max_dd = analytics._calculate_max_drawdown(returns)
        
        assert max_dd > 0
        assert max_dd <= 1.0  # Drawdown is percentage
    
    def test_max_drawdown_no_losses(self, analytics):
        """Test max drawdown with no losses."""
        returns = [0.05, 0.10, 0.08, 0.12]
        
        max_dd = analytics._calculate_max_drawdown(returns)
        
        assert max_dd == 0.0
    
    def test_max_drawdown_all_losses(self, analytics):
        """Test max drawdown with all losses."""
        returns = [-0.05, -0.10, -0.08]
        
        max_dd = analytics._calculate_max_drawdown(returns)
        
        assert max_dd > 0.20  # Should be significant


class TestEdgeCases:
    """Test edge cases."""
    
    def test_single_return(self, analytics):
        """Test calculations with single return."""
        returns = [0.05]
        dates = [datetime.now()]
        
        total = analytics._calculate_total_return(returns)
        assert total == 0.05
    
    def test_zero_returns(self, analytics):
        """Test calculations with all zero returns."""
        returns = [0.0] * 10
        dates = [datetime.now() - timedelta(days=10-i) for i in range(10)]
        
        metrics = analytics.calculate_performance_metrics(returns, dates)
        
        assert metrics.total_return == 0.0
        assert metrics.volatility == 0.0
    
    def test_extreme_positive_returns(self, analytics):
        """Test with extreme positive returns."""
        returns = [1.0, 0.5, 0.8]  # 100%, 50%, 80% daily returns
        dates = [datetime.now() - timedelta(days=3-i) for i in range(3)]
        
        metrics = analytics.calculate_performance_metrics(returns, dates)
        
        assert metrics.total_return > 4.0  # Should be very high
    
    def test_extreme_negative_returns(self, analytics):
        """Test with extreme negative returns."""
        returns = [-0.20, -0.15, -0.10]
        dates = [datetime.now() - timedelta(days=3-i) for i in range(3)]
        
        metrics = analytics.calculate_performance_metrics(returns, dates)
        
        assert metrics.total_return < 0


class TestRiskFreeRate:
    """Test risk-free rate handling."""
    
    def test_custom_risk_free_rate(self):
        """Test custom risk-free rate."""
        analytics = PerformanceAnalytics(risk_free_rate=0.05)
        
        assert analytics.risk_free_rate == 0.05
    
    def test_zero_risk_free_rate(self):
        """Test with zero risk-free rate."""
        analytics = PerformanceAnalytics(risk_free_rate=0.0)
        returns = [0.01] * 10
        dates = [datetime.now() - timedelta(days=10-i) for i in range(10)]
        
        sharpe = analytics._calculate_sharpe_ratio(returns, dates)
        
        # Sharpe should be higher with zero risk-free rate
        assert sharpe > 0
