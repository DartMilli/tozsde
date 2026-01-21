"""
Unit tests for technical indicators module.

Tests:
- SMA (Simple Moving Average) correctness
- Edge cases (short windows, NaN values)
- Performance (large datasets)
"""

import pytest
import numpy as np
import pandas as pd
from app.indicators.technical import sma


class TestSMA:
    """Tests for Simple Moving Average calculation."""

    def test_sma_basic_calculation(self):
        """SMA of [1,2,3,4,5] over 3 periods = [NaN, NaN, 2, 3, 4]."""
        data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(data, 3)
        expected = np.array([np.nan, np.nan, 2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_sma_window_larger_than_data(self):
        """SMA with window > data length should still compute but pad heavily."""
        data = pd.Series([1.0, 2.0, 3.0])
        result = sma(data, 10)
        # With window=10 and data length=3
        # np.convolve valid mode gives: max(M,N) - min(M,N) + 1 = 10 - 3 + 1 = 8 values
        # Padded with (window-1)=9 NaNs = 17 total
        assert len(result) == 17  # 9 NaNs + 8 computed values
        assert np.sum(np.isnan(result[:9])) == 9  # First 9 are NaN
        # Remaining 8 values should be equal (all convolved window positions give same result)
        np.testing.assert_array_almost_equal(result[9:], 0.6, decimal=5)

    def test_sma_constant_series(self):
        """SMA of constant series should equal the constant."""
        data = pd.Series([5.0] * 10)
        result = sma(data, 3)
        # First 2 values are NaN (window-1), rest should be 5.0
        np.testing.assert_array_almost_equal(result[2:], 5.0, decimal=5)

    def test_sma_single_element(self):
        """SMA with window=1 should equal the input."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(data, 1)
        np.testing.assert_array_almost_equal(result, data, decimal=5)

    def test_sma_with_nan_values(self):
        """SMA should handle NaN gracefully."""
        # SMA result starts with (period-1) NaN values
        data = np.array([np.nan, 1.0, 2.0, 3.0, 4.0])
        result = sma(data, 3)
        # Should have at least 2 NaN at start (from window) plus the input NaN
        assert np.isnan(result[0]) and np.isnan(result[1]) and np.isnan(result[2])


class TestIndicatorPerformance:
    """Performance tests for indicator calculations."""

    def test_sma_large_dataset(self):
        """SMA should compute efficiently on large datasets."""
        data = np.random.randn(10000)
        import time

        start = time.time()
        result = sma(data, 50)
        elapsed = time.time() - start
        # Should complete in <1 second (generous for varied systems)
        assert elapsed < 1.0
        assert len(result) == len(data)
        # First 49 should be NaN, rest should have values
        assert np.sum(np.isnan(result[:49])) == 49
