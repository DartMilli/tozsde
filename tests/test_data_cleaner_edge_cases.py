"""
data_cleaner edge case tests
Focus: sanitize_dataframe() and prepare_df() with unusual/invalid input
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from app.data_access.data_cleaner import sanitize_dataframe, prepare_df


class TestSanitizeDataframeEdgeCases:
    """Edge cases for sanitize_dataframe()"""

    def test_empty_dataframe(self):
        """Empty DataFrame should return empty but structured result"""
        df = pd.DataFrame()
        result = sanitize_dataframe(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_missing_all_expected_columns(self):
        """DataFrame with no OHLCV columns"""
        df = pd.DataFrame({"random": [1, 2, 3], "other": [4, 5, 6]})
        result = sanitize_dataframe(df)
        assert len(result.columns) == 0
        assert len(result) == 3  # rows preserved but no columns

    def test_partial_expected_columns(self):
        """Only some OHLCV columns present"""
        df = pd.DataFrame({
            "Open": [100, 101, 102],
            "Close": [99, 100, 101],
            "SomeOther": [1, 2, 3]
        })
        result = sanitize_dataframe(df)
        assert list(result.columns) == ["Open", "Close"]
        assert len(result) == 3

    def test_date_column_as_regular_column(self):
        """Date column exists but not as index"""
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "Open": [100, 101, 102],
            "Close": [99, 100, 101]
        })
        result = sanitize_dataframe(df, index_col="date")
        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result) == 3
        assert "date" not in result.columns

    def test_invalid_date_strings(self):
        """Invalid date strings should be coerced to NaT"""
        df = pd.DataFrame({
            "date": ["2024-01-01", "not-a-date", "2024-01-03"],
            "Open": [100, 101, 102],
            "Close": [99, 100, 101]
        })
        result = sanitize_dataframe(df, index_col="date")
        # NaT values remain in index
        assert result.index.isna().sum() == 1

    def test_non_numeric_values_coerced(self):
        """Non-numeric values should become NaN"""
        df = pd.DataFrame({
            "Open": [100, "abc", 102],
            "Close": [99, 100, "xyz"],
            "Volume": [1000, 2000, 3000]
        })
        result = sanitize_dataframe(df)
        assert pd.isna(result.iloc[1]["Open"])
        assert pd.isna(result.iloc[2]["Close"])
        assert result["Volume"].iloc[0] == 1000

    def test_mixed_int_float_conversion(self):
        """Mixed int and float values should all become float"""
        df = pd.DataFrame({
            "Open": [100, 101.5, 102],
            "Close": [99.5, 100, 101],
            "Volume": [1000, 2000, 3000]
        })
        result = sanitize_dataframe(df)
        # pd.to_numeric with errors='coerce' converts to float64 if mixed, int64 if all ints
        assert result["Open"].dtype == np.float64
        assert result["Close"].dtype == np.float64
        # Volume has all integers, so pd.to_numeric keeps it as int64
        assert result["Volume"].dtype in (np.int64, np.float64)

    def test_unsorted_dates_get_sorted(self):
        """Unsorted date index should be sorted"""
        df = pd.DataFrame({
            "date": ["2024-01-03", "2024-01-01", "2024-01-02"],
            "Open": [102, 100, 101],
            "Close": [101, 99, 100]
        })
        result = sanitize_dataframe(df, index_col="date")
        assert result.iloc[0]["Open"] == 100  # 2024-01-01
        assert result.iloc[1]["Open"] == 101  # 2024-01-02
        assert result.iloc[2]["Open"] == 102  # 2024-01-03

    def test_duplicate_dates(self):
        """Duplicate dates should be preserved"""
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "Open": [100, 101, 102],
            "Close": [99, 100, 101]
        })
        result = sanitize_dataframe(df, index_col="date")
        assert len(result) == 3  # All rows preserved

    def test_all_nan_column(self):
        """Column with all NaN values"""
        df = pd.DataFrame({
            "Open": [np.nan, np.nan, np.nan],
            "Close": [99, 100, 101]
        })
        result = sanitize_dataframe(df)
        assert "Open" in result.columns
        assert result["Open"].isna().all()

    def test_extreme_values(self):
        """Very large and very small numbers"""
        df = pd.DataFrame({
            "Open": [1e10, 1e-10, 1e5],
            "Close": [1e10, 1e-10, 1e5]
        })
        result = sanitize_dataframe(df)
        assert result["Open"].iloc[0] == 1e10
        assert result["Open"].iloc[1] == 1e-10

    def test_negative_prices(self):
        """Negative values should be preserved (validation not cleaner's job)"""
        df = pd.DataFrame({
            "Open": [-100, 101, 102],
            "Close": [-99, 100, 101]
        })
        result = sanitize_dataframe(df)
        assert result["Open"].iloc[0] == -100


class TestPrepareDfEdgeCases:
    """Edge cases for prepare_df()"""

    @patch("app.data_access.data_cleaner.get_params")
    @patch("app.data_access.data_cleaner.compute_signals")
    def test_empty_dataframe(self, mock_compute, mock_get_params):
        """Empty DataFrame should handle gracefully"""
        df = pd.DataFrame()
        mock_get_params.return_value = {"sma_period": 20}
        mock_compute.return_value = ([], {})
        
        result = prepare_df(df, "TEST")
        assert len(result) == 0

    @patch("app.data_access.data_cleaner.get_params")
    @patch("app.data_access.data_cleaner.compute_signals")
    def test_params_none_uses_default(self, mock_compute, mock_get_params):
        """None params should trigger get_params()"""
        df = pd.DataFrame({"Close": [100, 101, 102]})
        mock_get_params.return_value = {"sma_period": 20}
        mock_compute.return_value = ([], {"SMA": pd.Series([100, 101, 102], dtype=float)})
        
        result = prepare_df(df, "TEST", params=None)
        mock_get_params.assert_called_once_with("TEST")

    @patch("app.data_access.data_cleaner.get_params")
    @patch("app.data_access.data_cleaner.compute_signals")
    def test_params_provided_skips_default(self, mock_compute, mock_get_params):
        """Provided params should NOT call get_params()"""
        df = pd.DataFrame({"Close": [100, 101, 102]})
        custom_params = {"sma_period": 50}
        mock_compute.return_value = ([], {"SMA": pd.Series([100, 101, 102], dtype=float)})
        
        result = prepare_df(df, "TEST", params=custom_params)
        mock_get_params.assert_not_called()
        mock_compute.assert_called_once_with(df, "TEST", custom_params)

    @patch("app.data_access.data_cleaner.get_params")
    @patch("app.data_access.data_cleaner.compute_signals")
    def test_indicators_added_as_columns(self, mock_compute, mock_get_params):
        """Indicators from compute_signals should become DataFrame columns"""
        df = pd.DataFrame({"Close": [100, 101, 102]})
        mock_get_params.return_value = {"sma_period": 20}
        mock_compute.return_value = ([], {
            "SMA": pd.Series([100, 101, 102], dtype=float),
            "RSI": pd.Series([50, 55, 60], dtype=float)
        })
        
        result = prepare_df(df, "TEST")
        assert "SMA" in result.columns
        assert "RSI" in result.columns
        assert result["SMA"].iloc[0] == 100

    @patch("app.data_access.data_cleaner.get_params")
    @patch("app.data_access.data_cleaner.compute_signals")
    def test_dropna_removes_incomplete_rows(self, mock_compute, mock_get_params):
        """Rows with NaN indicators should be dropped"""
        df = pd.DataFrame({"Close": [100, 101, 102, 103, 104]})
        mock_get_params.return_value = {"sma_period": 20}
        # First 2 indicators have NaN (warmup period)
        mock_compute.return_value = ([], {
            "SMA": pd.Series([np.nan, np.nan, 102, 103, 104], dtype=float)
        })
        
        result = prepare_df(df, "TEST")
        assert len(result) == 3  # First 2 rows dropped

    @patch("app.data_access.data_cleaner.get_params")
    @patch("app.data_access.data_cleaner.compute_signals")
    def test_no_indicators_returned(self, mock_compute, mock_get_params):
        """Empty indicators dict should not break"""
        df = pd.DataFrame({"Close": [100, 101, 102]})
        mock_get_params.return_value = {"sma_period": 20}
        mock_compute.return_value = ([], {})
        
        result = prepare_df(df, "TEST")
        assert len(result) == 3  # Original data preserved

    @patch("app.data_access.data_cleaner.get_params")
    @patch("app.data_access.data_cleaner.compute_signals")
    def test_mismatched_indicator_length(self, mock_compute, mock_get_params):
        """Indicator series shorter than df should align correctly"""
        df = pd.DataFrame({"Close": [100, 101, 102, 103, 104]})
        mock_get_params.return_value = {"sma_period": 20}
        # Indicator with only 3 values
        short_series = pd.Series([102, 103, 104], dtype=float)
        mock_compute.return_value = ([], {"SMA": short_series})
        
        result = prepare_df(df, "TEST")
        # pandas alignment should create NaNs for missing indices
        assert len(result) <= 3  # dropna() removes misaligned rows
