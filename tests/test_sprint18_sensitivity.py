"""S18C – Pure-function coverage for execution_sensitivity.py & shadow_compare.py."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers – fake trade object
# ---------------------------------------------------------------------------


@dataclass
class _FakeTrade:
    entry_idx: int
    exit_idx: int


# ---------------------------------------------------------------------------
# execution_sensitivity – _paired_t_test
# ---------------------------------------------------------------------------


class TestPairedTTest:
    def _fn(self):
        from app.validation.execution_sensitivity import _paired_t_test

        return _paired_t_test

    def test_empty_lists(self):
        fn = self._fn()
        result = fn([], [])
        assert result["n"] == 0
        assert result["t_stat"] is None
        assert result["approx"] is True

    def test_single_pair(self):
        fn = self._fn()
        result = fn([1.0], [0.5])
        assert result["n"] == 1
        assert result["t_stat"] is None  # n < 2

    def test_proper_pairs(self):
        fn = self._fn()
        result = fn([1.0, 2.0, 3.0], [0.5, 1.5, 2.5])
        assert result["n"] == 3
        assert result["t_stat"] is not None
        assert result["p_value"] is not None

    def test_identical_pairs_zero_std(self):
        fn = self._fn()
        result = fn([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
        assert result["t_stat"] == pytest.approx(0.0)
        assert result["p_value"] == pytest.approx(1.0)

    def test_mismatched_lengths_uses_shortest(self):
        fn = self._fn()
        result = fn([1.0, 2.0, 3.0], [0.5, 1.5])
        # zip stops at the shorter list → 2 pairs
        assert result["n"] == 2

    def test_with_none_values_skipped(self):
        fn = self._fn()
        result = fn([1.0, None, 3.0], [0.5, 1.5, 2.5])
        assert result["n"] == 2  # None pair skipped


# ---------------------------------------------------------------------------
# execution_sensitivity – _variant_prices
# ---------------------------------------------------------------------------


class TestVariantPrices:
    def _fn(self):
        from app.validation.execution_sensitivity import _variant_prices

        return _variant_prices

    def test_close_reference(self):
        fn = self._fn()
        closes = np.array([100.0, 102.0, 104.0])
        opens = np.array([99.0, 101.0, 103.0])
        trade = _FakeTrade(entry_idx=0, exit_idx=2)
        from app.validation.execution_sensitivity import ExecutionVariant

        result = fn(trade, closes, opens, ExecutionVariant.CLOSE_REFERENCE)
        assert result == pytest.approx((100.0, 104.0))

    def test_same_bar_open(self):
        fn = self._fn()
        closes = np.array([100.0, 102.0, 104.0])
        opens = np.array([99.0, 101.0, 103.0])
        trade = _FakeTrade(entry_idx=0, exit_idx=2)
        from app.validation.execution_sensitivity import ExecutionVariant

        result = fn(trade, closes, opens, ExecutionVariant.SAME_BAR_OPEN)
        assert result == pytest.approx((99.0, 103.0))

    def test_next_open_shift_valid(self):
        fn = self._fn()
        closes = np.array([100.0, 102.0, 104.0])
        opens = np.array([99.0, 101.0, 103.0])
        trade = _FakeTrade(entry_idx=0, exit_idx=1)
        from app.validation.execution_sensitivity import ExecutionVariant

        result = fn(trade, closes, opens, ExecutionVariant.NEXT_OPEN_SHIFT)
        assert result == pytest.approx((101.0, 103.0))

    def test_next_open_shift_out_of_bounds(self):
        fn = self._fn()
        closes = np.array([100.0, 102.0])
        opens = np.array([99.0, 101.0])
        trade = _FakeTrade(entry_idx=0, exit_idx=1)
        from app.validation.execution_sensitivity import ExecutionVariant

        result = fn(trade, closes, opens, ExecutionVariant.NEXT_OPEN_SHIFT)
        assert result is None

    def test_hybrid_test_valid(self):
        fn = self._fn()
        closes = np.array([100.0, 102.0, 104.0])
        opens = np.array([99.0, 101.0, 103.0])
        trade = _FakeTrade(entry_idx=0, exit_idx=1)
        from app.validation.execution_sensitivity import ExecutionVariant

        result = fn(trade, closes, opens, ExecutionVariant.HYBRID_TEST)
        assert result == pytest.approx((100.0, 103.0))

    def test_hybrid_test_out_of_bounds(self):
        fn = self._fn()
        closes = np.array([100.0, 102.0])
        opens = np.array([99.0, 101.0])
        trade = _FakeTrade(entry_idx=0, exit_idx=1)
        from app.validation.execution_sensitivity import ExecutionVariant

        result = fn(trade, closes, opens, ExecutionVariant.HYBRID_TEST)
        assert result is None

    def test_unknown_variant_returns_none(self):
        fn = self._fn()
        closes = np.array([100.0, 102.0])
        opens = np.array([99.0, 101.0])
        trade = _FakeTrade(entry_idx=0, exit_idx=1)
        result = fn(trade, closes, opens, "UNKNOWN_VARIANT")
        assert result is None


# ---------------------------------------------------------------------------
# execution_sensitivity – _execution_returns
# ---------------------------------------------------------------------------


class TestExecutionReturns:
    def test_basic_returns(self):
        from app.validation.execution_sensitivity import (
            _execution_returns,
            ExecutionVariant,
        )

        closes = np.array([100.0, 110.0, 120.0])
        opens = np.array([99.0, 109.0, 119.0])
        trades = [_FakeTrade(entry_idx=0, exit_idx=2)]
        returns = _execution_returns(
            trades, closes, opens, 0.0, 0.0, 0.0, ExecutionVariant.CLOSE_REFERENCE
        )
        assert len(returns) == 1
        # sell/buy - 1 = 120/100 - 1 = 0.20
        assert returns[0] == pytest.approx(0.20, abs=1e-6)

    def test_none_when_price_is_none(self):
        from app.validation.execution_sensitivity import (
            _execution_returns,
            ExecutionVariant,
        )

        # next_open_shift on last bar → out of bounds → None
        closes = np.array([100.0, 110.0])
        opens = np.array([99.0, 109.0])
        trades = [_FakeTrade(entry_idx=0, exit_idx=1)]
        returns = _execution_returns(
            trades, closes, opens, 0.0, 0.0, 0.0, ExecutionVariant.NEXT_OPEN_SHIFT
        )
        assert returns[0] is None

    def test_with_slippage_fee(self):
        from app.validation.execution_sensitivity import (
            _execution_returns,
            ExecutionVariant,
        )

        closes = np.array([100.0, 105.0])
        opens = np.array([99.0, 104.0])
        trades = [_FakeTrade(entry_idx=0, exit_idx=1)]
        returns_no_fee = _execution_returns(
            trades, closes, opens, 0.0, 0.0, 0.0, ExecutionVariant.CLOSE_REFERENCE
        )
        returns_with_fee = _execution_returns(
            trades, closes, opens, 0.001, 0.001, 0.001, ExecutionVariant.CLOSE_REFERENCE
        )
        # Returns with fees should be lower
        assert returns_with_fee[0] < returns_no_fee[0]


# ---------------------------------------------------------------------------
# execution_sensitivity – _equity_metrics
# ---------------------------------------------------------------------------


class TestEquityMetrics:
    def test_empty_returns(self):
        from app.validation.execution_sensitivity import _equity_metrics

        result = _equity_metrics([], 10000.0, 0.1)
        assert result["trade_count"] == 0
        assert result["total_return"] == 0.0

    def test_all_none_returns(self):
        from app.validation.execution_sensitivity import _equity_metrics

        result = _equity_metrics([None, None], 10000.0, 0.1)
        assert result["trade_count"] == 0

    def test_single_return(self):
        from app.validation.execution_sensitivity import _equity_metrics

        result = _equity_metrics([0.10], 10000.0, 0.1)
        assert result["trade_count"] == 1
        assert result["avg_trade_return"] == pytest.approx(0.10, abs=1e-6)
        assert result["std_trade_return"] == pytest.approx(0.0)

    def test_multiple_returns(self):
        from app.validation.execution_sensitivity import _equity_metrics

        result = _equity_metrics([0.05, 0.10, -0.02], 10000.0, 0.1)
        assert result["trade_count"] == 3
        assert result["avg_trade_return"] == pytest.approx(
            (0.05 + 0.10 - 0.02) / 3, abs=1e-6
        )
        assert result["max_drawdown"] <= 0.0

    def test_compounding_mode(self):
        from app.validation.execution_sensitivity import _equity_metrics

        result = _equity_metrics([0.05, 0.10], 10000.0, None)
        assert result["trade_count"] == 2


# ---------------------------------------------------------------------------
# execution_sensitivity – _gap_stats
# ---------------------------------------------------------------------------


class TestGapStats:
    def test_no_trades(self):
        from app.validation.execution_sensitivity import _gap_stats

        closes = np.array([100.0, 102.0, 104.0])
        opens = np.array([99.0, 101.0, 103.0])
        result = _gap_stats([], closes, opens)
        assert result["mean_entry_gap"] == 0.0
        assert result["mean_exit_gap"] == 0.0

    def test_with_valid_trade(self):
        from app.validation.execution_sensitivity import _gap_stats

        closes = np.array([100.0, 102.0, 104.0, 106.0])
        opens = np.array([101.0, 103.0, 105.0, 107.0])
        trades = [_FakeTrade(entry_idx=0, exit_idx=1)]
        result = _gap_stats(trades, closes, opens)
        # entry_gap = opens[1]/closes[0] - 1 = 103/100 - 1 = 0.03
        assert result["mean_entry_gap"] == pytest.approx(0.03, abs=1e-6)

    def test_out_of_bounds_skipped(self):
        from app.validation.execution_sensitivity import _gap_stats

        closes = np.array([100.0, 102.0])
        opens = np.array([99.0, 101.0])
        trades = [_FakeTrade(entry_idx=0, exit_idx=1)]
        result = _gap_stats(trades, closes, opens)
        # Both entry+exit+1 are at boundary → skipped
        assert result["mean_entry_gap"] == 0.0


# ---------------------------------------------------------------------------
# ExecutionVariant constants
# ---------------------------------------------------------------------------


class TestExecutionVariantConstants:
    def test_constants_are_strings(self):
        from app.validation.execution_sensitivity import ExecutionVariant

        assert ExecutionVariant.CLOSE_REFERENCE == "close_reference"
        assert ExecutionVariant.SAME_BAR_OPEN == "same_bar_open"
        assert ExecutionVariant.NEXT_OPEN_SHIFT == "next_open_shift"
        assert ExecutionVariant.HYBRID_TEST == "hybrid_test"


# ---------------------------------------------------------------------------
# shadow_compare – _baseline_policy
# ---------------------------------------------------------------------------


class TestBaselinePolicy:
    def _fn(self):
        from app.validation.shadow_compare import _baseline_policy

        return _baseline_policy

    def test_oversold_returns_1(self):
        assert self._fn()(25.0) == 1

    def test_boundary_30_returns_1(self):
        assert self._fn()(29.99) == 1

    def test_overbought_returns_2(self):
        assert self._fn()(75.0) == 2

    def test_boundary_70_returns_2(self):
        assert self._fn()(70.01) == 2

    def test_neutral_returns_0(self):
        assert self._fn()(50.0) == 0

    def test_exactly_30_returns_0(self):
        # Not < 30, not > 70 → 0
        assert self._fn()(30.0) == 0

    def test_exactly_70_returns_0(self):
        assert self._fn()(70.0) == 0


# ---------------------------------------------------------------------------
# shadow_compare – _extract_rsi
# ---------------------------------------------------------------------------


class TestExtractRsi:
    def _fn(self):
        from app.validation.shadow_compare import _extract_rsi

        return _extract_rsi

    def test_uppercase_rsi(self):
        df = pd.DataFrame({"Close": [100.0], "RSI": [45.5]})
        assert self._fn()(df) == pytest.approx(45.5)

    def test_title_case_rsi(self):
        df = pd.DataFrame({"Close": [100.0], "Rsi": [60.0]})
        assert self._fn()(df) == pytest.approx(60.0)

    def test_lowercase_rsi(self):
        df = pd.DataFrame({"Close": [100.0], "rsi": [30.0]})
        assert self._fn()(df) == pytest.approx(30.0)

    def test_no_rsi_column(self):
        df = pd.DataFrame({"Close": [100.0], "Volume": [1000]})
        assert self._fn()(df) is None

    def test_rsi_bad_value_returns_none(self):
        df = pd.DataFrame({"RSI": ["bad"]})
        result = self._fn()(df)
        assert result is None


# ---------------------------------------------------------------------------
# shadow_compare – _TradeReturn dataclass (import coverage)
# ---------------------------------------------------------------------------


class TestShadowCompareImports:
    def test_baseline_policy_module_imports(self):
        import app.validation.shadow_compare as m

        assert callable(m._baseline_policy)
        assert callable(m._extract_rsi)
