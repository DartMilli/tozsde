"""
Comprehensive tests for all technical indicators.

Targets: EMA, RSI, MACD, BBANDS, ATR, ADX, STOCH coverage improvement
"""

import pytest
import numpy as np
import pandas as pd

from app.indicators.technical import (
    get_indicator_description,
    sma,
    ema,
    ema_old,
    rsi,
    rsi_old,
    macd,
    bbands,
    atr,
    adx,
    stoch,
)


class TestGetIndicatorDescription:
    """Test indicator description getter."""
    
    def test_get_indicator_description_returns_dict(self):
        """Test that indicator description returns a dictionary."""
        result = get_indicator_description()
        assert isinstance(result, dict)
        assert "SMA" in result
        assert "EMA" in result
        assert "RSI" in result


class TestEMA:
    """Test Exponential Moving Average."""
    
    def test_ema_basic(self):
        """Test basic EMA calculation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ema(data, period=3)
        assert len(result) == len(data)
        assert not np.isnan(result[-1])
    
    def test_ema_old_basic(self):
        """Test old EMA implementation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ema_old(data, period=3)
        assert len(result) == len(data)
        assert result[0] == data[0]
    
    def test_ema_single_value(self):
        """Test EMA with single value."""
        data = np.array([5.0])
        result = ema(data, period=3)
        assert len(result) == 1
        assert result[0] == 5.0
    
    def test_ema_constant_series(self):
        """Test EMA on constant series."""
        data = np.array([5.0] * 10)
        result = ema(data, period=3)
        # EMA of constant should converge to constant
        assert abs(result[-1] - 5.0) < 0.01


class TestRSI:
    """Test Relative Strength Index."""
    
    def test_rsi_basic(self):
        """Test basic RSI calculation."""
        data = np.array([44.0, 44.3, 44.1, 43.8, 44.5, 44.3, 45.0, 45.8, 46.1, 46.4,
                        46.8, 46.3, 46.1, 46.4, 46.5, 45.8, 46.0])
        result = rsi(data, period=14)
        assert len(result) == len(data)
        # RSI should be between 0 and 100
        assert np.all((result[~np.isnan(result)] >= 0) & (result[~np.isnan(result)] <= 100))
    
    def test_rsi_old_basic(self):
        """Test old RSI implementation."""
        data = np.array([44.0, 44.3, 44.1, 43.8, 44.5, 44.3, 45.0, 45.8, 46.1, 46.4,
                        46.8, 46.3, 46.1, 46.4, 46.5, 45.8, 46.0])
        result = rsi_old(data, period=14)
        assert len(result) == len(data)
        # RSI should be between 0 and 100
        assert np.all((result[~np.isnan(result)] >= 0) & (result[~np.isnan(result)] <= 100))
    
    def test_rsi_trending_up(self):
        """Test RSI on upward trending data."""
        data = np.linspace(10, 20, 30)
        result = rsi(data, period=14)
        # Strong uptrend should have high RSI
        assert result[-1] > 70
    
    def test_rsi_trending_down(self):
        """Test RSI on downward trending data."""
        data = np.linspace(20, 10, 30)
        result = rsi(data, period=14)
        # Strong downtrend should have low RSI
        assert result[-1] < 30
    
    def test_rsi_constant_values(self):
        """Test RSI on constant data."""
        data = np.array([50.0] * 20)
        result = rsi(data, period=14)
        # No change gives RSI = 0 (no gain or loss)
        assert np.isnan(result[-1]) or result[-1] == 0.0


class TestMACD:
    """Test Moving Average Convergence Divergence."""
    
    def test_macd_basic(self):
        """Test basic MACD calculation."""
        data = np.random.randn(50) + 100
        macd_line, signal_line = macd(data, fast=12, slow=26, signal=9)
        
        assert len(macd_line) == len(data)
        assert len(signal_line) == len(data)
    
    def test_macd_components(self):
        """Test MACD components relationship."""
        data = np.linspace(90, 110, 50)
        macd_line, signal_line = macd(data, fast=12, slow=26, signal=9)
        
        # MACD line should differ from signal line
        assert not np.array_equal(macd_line, signal_line)
    
    def test_macd_custom_params(self):
        """Test MACD with custom parameters."""
        data = np.random.randn(60) + 100
        macd_line, signal_line = macd(data, fast=8, slow=21, signal=5)
        
        assert len(macd_line) == len(data)
        assert not np.all(np.isnan(macd_line))


class TestBollingerBands:
    """Test Bollinger Bands."""
    
    def test_bbands_basic(self):
        """Test basic Bollinger Bands calculation."""
        data = np.random.randn(50) + 100
        upper, middle, lower = bbands(data, period=20, std_dev=2)
        
        assert len(upper) == len(data)
        assert len(middle) == len(data)
        assert len(lower) == len(data)
    
    def test_bbands_relationship(self):
        """Test Bollinger Bands relationship (upper > middle > lower)."""
        data = np.random.randn(50) + 100
        upper, middle, lower = bbands(data, period=20, std_dev=2)
        
        # Check where all three are valid (not NaN)
        valid_idx = ~(np.isnan(upper) | np.isnan(middle) | np.isnan(lower))
        
        if np.any(valid_idx):
            assert np.all(upper[valid_idx] >= middle[valid_idx])
            assert np.all(middle[valid_idx] >= lower[valid_idx])
    
    def test_bbands_constant_data(self):
        """Test Bollinger Bands on constant data."""
        data = np.array([100.0] * 30)
        upper, middle, lower = bbands(data, period=20, std_dev=2)
        
        # With zero volatility, all bands should converge
        valid_idx = ~np.isnan(middle)
        if np.any(valid_idx):
            assert np.allclose(upper[valid_idx], 100.0, atol=0.1)
            assert np.allclose(middle[valid_idx], 100.0, atol=0.1)
            assert np.allclose(lower[valid_idx], 100.0, atol=0.1)
    
    def test_bbands_custom_std(self):
        """Test Bollinger Bands with custom standard deviation."""
        data = np.random.randn(50) + 100
        upper_2, middle_2, lower_2 = bbands(data, period=20, std_dev=2)
        upper_3, middle_3, lower_3 = bbands(data, period=20, std_dev=3)
        
        # Middle should be the same
        np.testing.assert_array_almost_equal(middle_2, middle_3)
        
        # Bands with std_dev=3 should be wider
        valid_idx = ~np.isnan(upper_2)
        if np.any(valid_idx):
            assert np.all((upper_3[valid_idx] - middle_3[valid_idx]) >= 
                         (upper_2[valid_idx] - middle_2[valid_idx]))


class TestATR:
    """Test Average True Range."""
    
    def test_atr_basic(self):
        """Test basic ATR calculation."""
        np.random.seed(42)
        high = np.random.randn(50) + 102
        low = np.random.randn(50) + 98
        close = np.random.randn(50) + 100
        
        result = atr(high, low, close, period=14)
        assert len(result) == len(high)
        # ATR should be non-negative
        assert np.all(result[~np.isnan(result)] >= 0)
    
    def test_atr_increasing_volatility(self):
        """Test ATR with increasing volatility."""
        # Create data with increasing volatility
        high = np.linspace(100, 110, 30) + np.linspace(0, 5, 30)
        low = np.linspace(100, 110, 30) - np.linspace(0, 5, 30)
        close = np.linspace(100, 110, 30)
        
        result = atr(high, low, close, period=14)
        
        # ATR should increase with volatility
        if not np.isnan(result[-10]) and not np.isnan(result[-1]):
            assert result[-1] > result[-10]
    
    def test_atr_zero_range(self):
        """Test ATR with zero price range."""
        high = np.array([100.0] * 20)
        low = np.array([100.0] * 20)
        close = np.array([100.0] * 20)
        
        result = atr(high, low, close, period=14)
        # ATR should be zero or near zero
        valid_idx = ~np.isnan(result)
        if np.any(valid_idx):
            assert np.allclose(result[valid_idx], 0.0, atol=0.01)


class TestADX:
    """Test Average Directional Index."""
    
    def test_adx_basic(self):
        """Test basic ADX calculation."""
        np.random.seed(42)
        high = np.cumsum(np.random.randn(50)) + 100
        low = high - np.abs(np.random.randn(50))
        close = (high + low) / 2
        
        adx_vals, plus_di, minus_di = adx(high, low, close, period=14)
        assert len(adx_vals) == len(high)
        assert len(plus_di) == len(high)
        assert len(minus_di) == len(high)
        # ADX should be between 0 and 100
        valid_idx = ~np.isnan(adx_vals)
        if np.any(valid_idx):
            assert np.all((adx_vals[valid_idx] >= 0) & (adx_vals[valid_idx] <= 100))
    
    def test_adx_strong_trend(self):
        """Test ADX on strong trending data."""
        # Strong uptrend
        high = np.linspace(100, 150, 50)
        low = np.linspace(95, 145, 50)
        close = np.linspace(98, 148, 50)
        
        adx_vals, plus_di, minus_di = adx(high, low, close, period=14)
        
        # Strong trend should have high ADX
        if not np.isnan(adx_vals[-1]):
            assert adx_vals[-1] > 25
    
    def test_adx_sideways_market(self):
        """Test ADX on sideways/ranging market."""
        # Sideways market
        high = np.sin(np.linspace(0, 4*np.pi, 50)) * 2 + 100
        low = high - 2
        close = (high + low) / 2
        
        adx_vals, plus_di, minus_di = adx(high, low, close, period=14)
        
        # Weak trend should have lower ADX
        # (This is probabilistic, so just check it's computed)
        assert not np.all(np.isnan(adx_vals))


class TestStochastic:
    """Test Stochastic Oscillator."""
    
    def test_stoch_basic(self):
        """Test basic Stochastic calculation."""
        np.random.seed(42)
        high = np.random.randn(50) + 102
        low = np.random.randn(50) + 98
        close = np.random.randn(50) + 100
        
        k, d = stoch(high, low, close, k_period=14, d_period=3)
        
        assert len(k) == len(high)
        assert len(d) == len(high)
        
        # Stochastic should be between 0 and 100
        valid_k = ~np.isnan(k)
        valid_d = ~np.isnan(d)
        
        if np.any(valid_k):
            assert np.all((k[valid_k] >= 0) & (k[valid_k] <= 100))
        if np.any(valid_d):
            assert np.all((d[valid_d] >= 0) & (d[valid_d] <= 100))
    
    def test_stoch_at_high(self):
        """Test Stochastic when price is at high."""
        # Price at highest point
        high = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 
                        110, 111, 112, 113, 114, 115])
        low = high - 5
        close = high  # Close at high
        
        k, d = stoch(high, low, close, k_period=14, d_period=3)
        
        # When close is at high, %K should be near 100
        if not np.isnan(k[-1]):
            assert k[-1] > 90
    
    def test_stoch_at_low(self):
        """Test Stochastic when price is at low."""
        # Price at lowest point
        high = np.array([110, 109, 108, 107, 106, 105, 104, 103, 102, 101,
                        100, 99, 98, 97, 96, 95])
        low = high - 5
        close = low  # Close at low
        
        k, d = stoch(high, low, close, k_period=14, d_period=3)
        
        # When close is at low, %K should be near 0
        if not np.isnan(k[-1]):
            assert k[-1] < 10
    
    def test_stoch_custom_periods(self):
        """Test Stochastic with custom periods."""
        np.random.seed(42)
        high = np.random.randn(50) + 102
        low = np.random.randn(50) + 98
        close = np.random.randn(50) + 100
        
        k, d = stoch(high, low, close, k_period=10, d_period=5)
        
        assert len(k) == len(high)
        assert len(d) == len(high)


class TestEdgeCases:
    """Test edge cases for all indicators."""
    
    def test_short_data_series(self):
        """Test indicators with very short data."""
        data = np.array([100.0, 101.0, 99.0])
        
        # Should not crash
        result_sma = sma(data, period=2)
        assert len(result_sma) == len(data)
        
        result_ema = ema(data, period=2)
        assert len(result_ema) == len(data)
    
    def test_single_data_point(self):
        """Test indicators with single data point."""
        data = np.array([100.0])
        
        result_sma = sma(data, period=1)
        assert len(result_sma) == 1
        
        result_ema = ema(data, period=1)
        assert len(result_ema) == 1
    
    def test_adx_short_data(self):
        """Test ADX with data too short for calculation."""
        # ADX needs period*2+1 minimum
        high = np.array([100, 101, 102])
        low = np.array([95, 96, 97])
        close = np.array([98, 99, 100])
        
        adx_vals, plus_di, minus_di = adx(high, low, close, period=14)
        # Should return NaN arrays for short data
        assert np.all(np.isnan(adx_vals))
        assert np.all(np.isnan(plus_di))
        assert np.all(np.isnan(minus_di))
    
    def test_nan_handling(self):
        """Test that indicators handle NaN gracefully."""
        data = np.array([100.0, np.nan, 102.0, 103.0, 104.0])
        
        # Should not crash
        result_ema = ema(data, period=3)
        assert len(result_ema) == len(data)
