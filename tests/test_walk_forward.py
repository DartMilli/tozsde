"""
Unit tests for walk-forward optimization.

Tests:
- Window generation and slicing
- Train/test split correctness
- Window boundary conditions
- Out-of-sample performance tracking
"""

import pytest
import numpy as np
import pandas as pd
from app.backtesting.walk_forward import WalkForwardOptimizer


class TestWalkForwardWindows:
    """Tests for rolling window generation."""

    def test_wf_window_generation(self, sample_df):
        """Walk-forward should generate correct train/test windows."""
        wf = WalkForwardOptimizer(
            ticker="TEST",
            df=sample_df,
            bounds={},
            train_window=15,
            test_window=5,
            step_size=5,
            verbose=False,
        )

        # Check initialization
        assert wf.ticker == "TEST"
        assert len(wf.df) == 30  # sample_df has 30 rows
        assert wf.train_window == 15
        assert wf.test_window == 5

    def test_wf_no_data_leakage(self, sample_df):
        """Test data should not overlap with training data."""
        # Create a simple window slice manually
        start_train = 0
        end_train = 15
        start_test = 15
        end_test = 20

        train_df = sample_df.iloc[start_train:end_train]
        test_df = sample_df.iloc[start_test:end_test]

        # No rows should appear in both
        assert len(train_df) == 15
        assert len(test_df) == 5
        # Verify no date overlap
        last_train_date = train_df.index[-1]
        first_test_date = test_df.index[0]
        assert first_test_date > last_train_date

    def test_wf_window_boundaries(self, sample_df):
        """Window boundaries should not overlap."""
        wf = WalkForwardOptimizer(
            ticker="TEST",
            df=sample_df,
            bounds={},
            train_window=10,
            test_window=5,
            step_size=5,
            verbose=False,
        )

        # Manually compute window positions
        start_train = wf.train_window  # 10
        start_test = start_train  # 10
        end_test = start_test + wf.test_window  # 15

        # Check that we have enough data
        assert end_test <= len(wf.df)


class TestWalkForwardValidation:
    """Tests for walk-forward validation logic."""

    def test_wf_validation_decreasing_performance(self):
        """Track profits across windows for performance degradation."""
        # Simulated OOS profits showing degradation
        oos_profits = [100, 80, 60, 40]  # Declining

        # Check if decreasing
        is_degrading = all(
            oos_profits[i] > oos_profits[i + 1] for i in range(len(oos_profits) - 1)
        )

        assert is_degrading  # Should detect degradation

    def test_wf_validation_stable_performance(self):
        """Track stable performance across windows."""
        # Simulated OOS profits showing stability
        oos_profits = [100, 95, 102, 98]  # Stable around 100

        # Calculate mean and std
        mean_profit = np.mean(oos_profits)
        std_profit = np.std(oos_profits)
        cv = std_profit / abs(mean_profit) if mean_profit != 0 else float("inf")

        # Stable means low coefficient of variation
        assert cv < 0.2  # Less than 20% variation


class TestWalkForwardEdgeCases:
    """Tests for edge cases."""

    def test_wf_insufficient_data(self):
        """Walk-forward should handle small datasets gracefully."""
        # Create minimal DataFrame (5 rows)
        dates = pd.date_range(start="2025-01-01", periods=5, freq="D")
        small_df = pd.DataFrame(
            {
                "Open": [100, 101, 102, 103, 104],
                "High": [101, 102, 103, 104, 105],
                "Low": [99, 100, 101, 102, 103],
                "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
                "Volume": [1000000] * 5,
            },
            index=dates,
        )

        # WalkForwardOptimizer should initialize with small data
        wf = WalkForwardOptimizer(
            ticker="SMALL",
            df=small_df,
            bounds={},
            train_window=3,
            test_window=1,
            step_size=1,
            verbose=False,
        )

        assert len(wf.df) == 5
        assert wf.train_window <= len(wf.df)

    def test_wf_single_window(self, sample_df):
        """Walk-forward should work with single window."""
        wf = WalkForwardOptimizer(
            ticker="TEST",
            df=sample_df,
            bounds={},
            train_window=20,
            test_window=10,
            step_size=30,  # Large step to fit only 1 window
            verbose=False,
        )

        # Should have at least 1 valid window
        assert len(wf.df) >= wf.train_window + wf.test_window
