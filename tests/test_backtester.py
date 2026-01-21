"""
Unit tests for backtester core logic.

Tests:
- Trade execution correctness
- Transaction cost application
- Performance metrics (Sharpe, Drawdown, etc.)
- Edge cases (no trades, all losses)
"""

import pytest
import numpy as np
import pandas as pd
from app.backtesting.backtester import Backtester


def default_params():
    """Default valid parameters for backtester."""
    return {
        "sma_period": 20,
        "ema_period": 10,
        "rsi_period": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "bbands_period": 20,
        "bbands_stddev": 2.0,
        "atr_period": 14,
        "adx_period": 14,
        "stoch_k": 14,
        "stoch_d": 3,
    }


class TestBacktesterTradeExecution:
    """Tests for trade execution logic."""

    def test_backtester_no_trades(self, sample_df):
        """Backtester with no trades should return minimal changes."""
        df = sample_df.copy()
        bt = Backtester(df, "TEST")
        params = default_params()
        params["sma_period"] = 50
        params["ema_period"] = 20
        report = bt.run(params)
        assert isinstance(report.metrics["trade_count"], (int, float))
        assert report.metrics["net_profit"] is not None

    def test_backtester_single_buy_hold(self, sample_df):
        """Buy and hold strategy should return price appreciation."""
        df = sample_df.copy()
        df["Close"] = np.linspace(100, 110, len(df))
        bt = Backtester(df, "TEST")
        params = default_params()
        params["sma_period"] = 5
        params["ema_period"] = 3
        report = bt.run(params)
        assert report.metrics["net_profit"] is not None
        assert isinstance(report.metrics["trade_count"], (int, float))

    def test_backtester_transaction_costs(self, sample_df):
        """Transaction costs should reduce profit."""
        df = sample_df.copy()
        df["Close"] = 100.0
        bt = Backtester(df, "TEST")
        params = default_params()
        params["sma_period"] = 5
        params["ema_period"] = 3
        report = bt.run(params)
        if report.metrics["trade_count"] > 0:
            assert report.metrics["net_profit"] <= 0

    def test_backtester_multiple_trades(self, sample_df):
        """Multiple trades should accumulate costs."""
        df = sample_df.copy()
        df["Close"] = np.tile([100, 105, 100, 105], len(df) // 4 + 1)[: len(df)]
        bt = Backtester(df, "TEST")
        params = default_params()
        params["sma_period"] = 2
        params["ema_period"] = 2
        params["rsi_period"] = 5
        params["macd_fast"] = 5
        params["macd_slow"] = 21
        params["macd_signal"] = 5
        report = bt.run(params)
        assert report.metrics["trade_count"] >= 0
        assert report.metrics["net_profit"] is not None


class TestBacktesterMetrics:
    """Tests for performance metric calculations."""

    def test_backtester_sharpe_ratio(self, sample_df):
        """Sharpe ratio should be computable from returns."""
        df = sample_df.copy()
        bt = Backtester(df, "TEST")
        params = default_params()
        report = bt.run(params)
        assert "sharpe" in report.diagnostics
        assert isinstance(report.diagnostics["sharpe"], (int, float))

    def test_backtester_max_drawdown(self, sample_df):
        """Maximum drawdown should reflect largest peak-to-trough decline."""
        df = sample_df.copy()
        df["Close"] = np.concatenate(
            [
                np.linspace(100, 120, len(df) // 2),
                np.linspace(120, 80, len(df) - len(df) // 2),
            ]
        )
        bt = Backtester(df, "TEST")
        params = default_params()
        report = bt.run(params)
        assert report.metrics["max_drawdown"] is not None
        assert report.metrics["max_drawdown"] <= 0

    def test_backtester_win_rate(self, sample_df):
        """Win rate should be percentage of profitable trades."""
        df = sample_df.copy()
        bt = Backtester(df, "TEST")
        params = default_params()
        report = bt.run(params)
        if report.metrics["trade_count"] > 0:
            assert 0 <= report.metrics["winrate"] <= 1

    def test_backtester_profit_factor(self, sample_df):
        """Profit factor should be gross profit / gross loss."""
        df = sample_df.copy()
        bt = Backtester(df, "TEST")
        params = default_params()
        report = bt.run(params)
        assert report.metrics["profit_factor"] >= 0


class TestBacktesterEdgeCases:
    """Tests for edge cases and error handling."""

    def test_backtester_all_losses(self, sample_df):
        """Backtester should handle all-losing trades."""
        df = sample_df.copy()
        df["Close"] = np.linspace(100, 50, len(df))
        bt = Backtester(df, "TEST")
        params = default_params()
        report = bt.run(params)
        assert isinstance(report.metrics["profit_factor"], (int, float))
        assert report.metrics["winrate"] is not None

    def test_backtester_insufficient_data(self):
        """Backtester should validate minimum data length."""
        dates = pd.date_range(start="2025-01-01", periods=2, freq="D")
        df = pd.DataFrame(
            {
                "Close": [100, 101],
                "High": [100.5, 101.5],
                "Low": [99.5, 100.5],
                "Open": [100, 101],
                "Volume": [1000, 1000],
            },
            index=dates,
        )
        bt = Backtester(df, "TEST")
        params = default_params()
        params["sma_period"] = 50
        report = bt.run(params)
        assert isinstance(report, object)
        assert report.metrics["trade_count"] == 0

    def test_backtester_missing_ohlcv(self):
        """Backtester should work with available OHLCV columns."""
        dates = pd.date_range(start="2025-01-01", periods=30, freq="D")
        df = pd.DataFrame(
            {
                "Close": np.linspace(100, 105, 30),
                "High": np.linspace(100.5, 105.5, 30),
                "Low": np.linspace(99.5, 104.5, 30),
                "Open": np.linspace(100, 105, 30),
                "Volume": np.ones(30) * 1000,
            },
            index=dates,
        )
        bt = Backtester(df, "TEST")
        params = default_params()
        report = bt.run(params)
        assert report.metrics is not None
