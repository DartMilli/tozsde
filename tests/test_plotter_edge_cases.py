"""
plotter edge case tests
Focus: get_candle_img_buffer(), get_equity_curve_buffer(), get_drawdown_curve_buffer(), 
       plot_gradient_scatter(), plot_strategy_colored_scatter() with unusual/invalid input
"""

import pytest
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from io import BytesIO
import base64

from app.config.config import Config
from app.reporting.plotter import (
    get_candle_img_buffer,
    get_equity_curve_buffer,
    get_drawdown_curve_buffer,
    plot_gradient_scatter,
    plot_strategy_colored_scatter
)


class TestGetCandleImgBuffer:
    """Edge cases for get_candle_img_buffer()"""

    def test_empty_dataframe(self):
        """Empty DataFrame should raise KeyError due to missing columns"""
        df = pd.DataFrame()
        indicators = {}
        # Empty df has no 'Close' column, will raise KeyError
        with pytest.raises(KeyError):
            get_candle_img_buffer(df, indicators)

    def test_minimal_dataframe(self):
        """Single row DataFrame"""
        dates = pd.date_range(start="2024-01-01", periods=1)
        df = pd.DataFrame({
            "Open": [100],
            "High": [105],
            "Low": [99],
            "Close": [102],
            "Volume": [1000]
        }, index=dates)
        
        indicators = {
            "SMA": pd.Series([100], index=dates, dtype=float),
            "EMA": pd.Series([100], index=dates, dtype=float),
            "BB_upper": pd.Series([105], index=dates, dtype=float),
            "BB_middle": pd.Series([100], index=dates, dtype=float),
            "BB_lower": pd.Series([95], index=dates, dtype=float),
            "RSI": pd.Series([50], index=dates, dtype=float),
            "MACD": pd.Series([0], index=dates, dtype=float),
            "MACD_SIGNAL": pd.Series([0], index=dates, dtype=float),
            "STOCH_K": pd.Series([50], index=dates, dtype=float),
            "STOCH_D": pd.Series([50], index=dates, dtype=float),
            "ATR": pd.Series([2], index=dates, dtype=float),
            "ADX": pd.Series([25], index=dates, dtype=float),
            "PLUS_DI": pd.Series([20], index=dates, dtype=float),
            "MINUS_DI": pd.Series([20], index=dates, dtype=float)
        }
        
        result = get_candle_img_buffer(df, indicators)
        assert isinstance(result, BytesIO)
        assert result.tell() == 0  # Buffer pointer at start

    def test_missing_indicators(self):
        """Missing some or all indicator keys"""
        dates = pd.date_range(start="2024-01-01", periods=10)
        df = pd.DataFrame({
            "Open": [100] * 10,
            "High": [105] * 10,
            "Low": [99] * 10,
            "Close": [102] * 10,
            "Volume": [1000] * 10
        }, index=dates)
        
        # Only partial indicators
        indicators = {
            "SMA": pd.Series([100] * 10, index=dates, dtype=float),
            "EMA": pd.Series([100] * 10, index=dates, dtype=float)
        }
        
        # Should raise KeyError for missing indicator
        with pytest.raises(KeyError):
            get_candle_img_buffer(df, indicators)

    def test_signals_none(self):
        """No signals provided"""
        dates = pd.date_range(start="2024-01-01", periods=5)
        df = pd.DataFrame({
            "Open": [100, 101, 102, 103, 104],
            "High": [105, 106, 107, 108, 109],
            "Low": [99, 100, 101, 102, 103],
            "Close": [102, 103, 104, 105, 106],
            "Volume": [1000] * 5
        }, index=dates)
        
        indicators = {
            "SMA": pd.Series([100] * 5, index=dates, dtype=float),
            "EMA": pd.Series([100] * 5, index=dates, dtype=float),
            "BB_upper": pd.Series([105] * 5, index=dates, dtype=float),
            "BB_middle": pd.Series([100] * 5, index=dates, dtype=float),
            "BB_lower": pd.Series([95] * 5, index=dates, dtype=float),
            "RSI": pd.Series([50] * 5, index=dates, dtype=float),
            "MACD": pd.Series([0] * 5, index=dates, dtype=float),
            "MACD_SIGNAL": pd.Series([0] * 5, index=dates, dtype=float),
            "STOCH_K": pd.Series([50] * 5, index=dates, dtype=float),
            "STOCH_D": pd.Series([50] * 5, index=dates, dtype=float),
            "ATR": pd.Series([2] * 5, index=dates, dtype=float),
            "ADX": pd.Series([25] * 5, index=dates, dtype=float),
            "PLUS_DI": pd.Series([20] * 5, index=dates, dtype=float),
            "MINUS_DI": pd.Series([20] * 5, index=dates, dtype=float)
        }
        
        result = get_candle_img_buffer(df, indicators, signals=None)
        assert isinstance(result, BytesIO)

    def test_signals_with_valid_dates(self):
        """Signals with valid dates in df"""
        dates = pd.date_range(start="2024-01-01", periods=5)
        df = pd.DataFrame({
            "Open": [100, 101, 102, 103, 104],
            "High": [105, 106, 107, 108, 109],
            "Low": [99, 100, 101, 102, 103],
            "Close": [102, 103, 104, 105, 106],
            "Volume": [1000] * 5
        }, index=dates)
        
        indicators = {
            "SMA": pd.Series([100] * 5, index=dates, dtype=float),
            "EMA": pd.Series([100] * 5, index=dates, dtype=float),
            "BB_upper": pd.Series([105] * 5, index=dates, dtype=float),
            "BB_middle": pd.Series([100] * 5, index=dates, dtype=float),
            "BB_lower": pd.Series([95] * 5, index=dates, dtype=float),
            "RSI": pd.Series([50] * 5, index=dates, dtype=float),
            "MACD": pd.Series([0] * 5, index=dates, dtype=float),
            "MACD_SIGNAL": pd.Series([0] * 5, index=dates, dtype=float),
            "STOCH_K": pd.Series([50] * 5, index=dates, dtype=float),
            "STOCH_D": pd.Series([50] * 5, index=dates, dtype=float),
            "ATR": pd.Series([2] * 5, index=dates, dtype=float),
            "ADX": pd.Series([25] * 5, index=dates, dtype=float),
            "PLUS_DI": pd.Series([20] * 5, index=dates, dtype=float),
            "MINUS_DI": pd.Series([20] * 5, index=dates, dtype=float)
        }
        
        signals = ["BUY on 2024-01-02", "SELL on 2024-01-04"]
        result = get_candle_img_buffer(df, indicators, signals=signals)
        assert isinstance(result, BytesIO)

    def test_signals_with_invalid_dates(self):
        """Signals with dates not in df"""
        dates = pd.date_range(start="2024-01-01", periods=5)
        df = pd.DataFrame({
            "Open": [100, 101, 102, 103, 104],
            "High": [105, 106, 107, 108, 109],
            "Low": [99, 100, 101, 102, 103],
            "Close": [102, 103, 104, 105, 106],
            "Volume": [1000] * 5
        }, index=dates)
        
        indicators = {
            "SMA": pd.Series([100] * 5, index=dates, dtype=float),
            "EMA": pd.Series([100] * 5, index=dates, dtype=float),
            "BB_upper": pd.Series([105] * 5, index=dates, dtype=float),
            "BB_middle": pd.Series([100] * 5, index=dates, dtype=float),
            "BB_lower": pd.Series([95] * 5, index=dates, dtype=float),
            "RSI": pd.Series([50] * 5, index=dates, dtype=float),
            "MACD": pd.Series([0] * 5, index=dates, dtype=float),
            "MACD_SIGNAL": pd.Series([0] * 5, index=dates, dtype=float),
            "STOCH_K": pd.Series([50] * 5, index=dates, dtype=float),
            "STOCH_D": pd.Series([50] * 5, index=dates, dtype=float),
            "ATR": pd.Series([2] * 5, index=dates, dtype=float),
            "ADX": pd.Series([25] * 5, index=dates, dtype=float),
            "PLUS_DI": pd.Series([20] * 5, index=dates, dtype=float),
            "MINUS_DI": pd.Series([20] * 5, index=dates, dtype=float)
        }
        
        # Invalid dates skip signal markers, but still crash on 's' parameter
        signals = ["BUY on 2025-01-01", "SELL on 2025-12-31"]  # Future dates not in df
        # No signals added (dates not in df), but code still has 's' parameter issue if any markers exist
        # Actually with invalid dates, no markers added, so this should work
        result = get_candle_img_buffer(df, indicators, signals=signals)
        assert isinstance(result, BytesIO)  # Should handle gracefully since no valid signals


class TestGetEquityCurveBuffer:
    """Edge cases for get_equity_curve_buffer()"""

    def test_empty_equity_curve(self):
        """Empty equity curve dict"""
        equity_curve = {"date": [], "portfolio_value": []}
        result = get_equity_curve_buffer("TEST", equity_curve)
        assert isinstance(result, str)  # Base64 string
        assert len(result) > 0

    def test_single_point(self):
        """Single data point"""
        equity_curve = {
            "date": [datetime(2024, 1, 1)],
            "portfolio_value": [10000]
        }
        result = get_equity_curve_buffer("TEST", equity_curve)
        assert isinstance(result, str)
        # Decode to verify it's valid base64
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_negative_values(self):
        """Negative portfolio values (losses)"""
        dates = [datetime(2024, 1, i) for i in range(1, 6)]
        equity_curve = {
            "date": dates,
            "portfolio_value": [10000, 8000, 5000, -1000, -5000]
        }
        result = get_equity_curve_buffer("TEST", equity_curve)
        assert isinstance(result, str)


class TestGetDrawdownCurveBuffer:
    """Edge cases for get_drawdown_curve_buffer()"""

    def test_empty_series(self):
        """Empty equity curve series"""
        equity_curve = pd.Series([], dtype=float)
        result = get_drawdown_curve_buffer(equity_curve)
        assert isinstance(result, str)

    def test_single_value(self):
        """Single value series"""
        equity_curve = pd.Series([10000], dtype=float)
        result = get_drawdown_curve_buffer(equity_curve)
        assert isinstance(result, str)

    def test_no_drawdown(self):
        """Always increasing equity (no drawdown)"""
        equity_curve = pd.Series([10000, 11000, 12000, 13000], dtype=float)
        result = get_drawdown_curve_buffer(equity_curve)
        assert isinstance(result, str)

    def test_extreme_drawdown(self):
        """100% drawdown scenario"""
        equity_curve = pd.Series([10000, 5000, 0], dtype=float)
        result = get_drawdown_curve_buffer(equity_curve)
        assert isinstance(result, str)


class TestPlotGradientScatter:
    """Edge cases for plot_gradient_scatter()"""

    @patch.object(Config, "CHART_DIR", "/tmp/charts")
    @patch("app.reporting.plotter.plt.savefig")
    @patch("app.reporting.plotter.plt.close")
    def test_empty_subset(self, mock_close, mock_savefig):
        """Empty DataFrame should return early"""
        subset = pd.DataFrame()
        plot_gradient_scatter(subset, "TEST", "DQN")
        # Should return without plotting
        mock_savefig.assert_not_called()

    @patch.object(Config, "CHART_DIR", "/tmp/charts")
    @patch("matplotlib.figure.Figure.savefig")
    @patch("app.reporting.plotter.plt.close")
    def test_single_row(self, mock_close, mock_savefig):
        """Single data point"""
        subset = pd.DataFrame({
            "sharpe_ratio": [1.5],
            "final_portfolio_value": [15000],
            "composite_score": [0.8],
            "reward_strategy": ["profit"]
        })
        plot_gradient_scatter(subset, "TEST", "DQN")
        mock_savefig.assert_called_once()

    @patch.object(Config, "CHART_DIR", "/tmp/charts")
    @patch("matplotlib.figure.Figure.savefig")
    @patch("app.reporting.plotter.plt.close")
    def test_less_than_3_rows(self, mock_close, mock_savefig):
        """Less than 3 rows (top3 annotation edge case)"""
        subset = pd.DataFrame({
            "sharpe_ratio": [1.5, 2.0],
            "final_portfolio_value": [15000, 20000],
            "composite_score": [0.8, 0.9],
            "reward_strategy": ["profit", "sharpe"]
        })
        plot_gradient_scatter(subset, "TEST", "DQN")
        mock_savefig.assert_called_once()

    @patch.object(Config, "CHART_DIR", "/tmp/charts")
    @patch("matplotlib.figure.Figure.savefig")
    @patch("app.reporting.plotter.plt.close")
    def test_identical_scores(self, mock_close, mock_savefig):
        """All same composite scores (colormap edge case)"""
        subset = pd.DataFrame({
            "sharpe_ratio": [1.5, 2.0, 2.5],
            "final_portfolio_value": [15000, 20000, 25000],
            "composite_score": [0.8, 0.8, 0.8],
            "reward_strategy": ["profit", "sharpe", "drawdown"]
        })
        plot_gradient_scatter(subset, "TEST", "DQN")
        mock_savefig.assert_called_once()


class TestPlotStrategyColoredScatter:
    """Edge cases for plot_strategy_colored_scatter()"""

    @patch.object(Config, "CHART_DIR", "/tmp/charts")
    @patch("app.reporting.plotter.plt.savefig")
    @patch("app.reporting.plotter.plt.close")
    def test_empty_subset(self, mock_close, mock_savefig):
        """Empty DataFrame should return early"""
        subset = pd.DataFrame()
        plot_strategy_colored_scatter(subset, "TEST", "DQN")
        mock_savefig.assert_not_called()

    @patch.object(Config, "CHART_DIR", "/tmp/charts")
    @patch("matplotlib.figure.Figure.savefig")
    @patch("app.reporting.plotter.plt.close")
    def test_single_strategy(self, mock_close, mock_savefig):
        """Only one strategy type"""
        subset = pd.DataFrame({
            "sharpe_ratio": [1.5, 2.0, 2.5],
            "final_portfolio_value": [15000, 20000, 25000],
            "composite_score": [0.7, 0.8, 0.9],
            "reward_strategy": ["profit", "profit", "profit"]
        })
        plot_strategy_colored_scatter(subset, "TEST", "DQN")
        mock_savefig.assert_called_once()

    @patch.object(Config, "CHART_DIR", "/tmp/charts")
    @patch("matplotlib.figure.Figure.savefig")
    @patch("app.reporting.plotter.plt.close")
    def test_many_strategies(self, mock_close, mock_savefig):
        """More than 10 strategies (colormap limit)"""
        strategies = [f"strategy_{i}" for i in range(15)]
        subset = pd.DataFrame({
            "sharpe_ratio": np.random.rand(15),
            "final_portfolio_value": np.random.rand(15) * 10000 + 10000,
            "composite_score": np.random.rand(15),
            "reward_strategy": strategies
        })
        plot_strategy_colored_scatter(subset, "TEST", "DQN")
        mock_savefig.assert_called_once()

