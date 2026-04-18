"""S17: Validation helper coverage boost.

Targets (pure-logic functions):
- app/validation/metrics_core.py (22% → 90%+)
- app/validation/robustness_self_check.py (0% → 60%+)
- app/validation/post_upgrade_diagnostics.py helpers (0% → 30%+)
- app/validation/drift_monitor.py pure helpers (0% → 20%+)
- app/validation/risk_stress.py (22% → 50%+)
- app/validation/utils.py (46% → 80%+)
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# metrics_core.py — pure financial math
# ─────────────────────────────────────────────────────────────────────────────


def _equity(values):
    return pd.Series(values, dtype=float)


class TestMetricsCore:
    def test_compute_returns_basic(self):
        from app.validation.metrics_core import compute_returns

        eq = _equity([100, 110, 105])
        ret = compute_returns(eq)
        assert len(ret) == 2
        assert ret.iloc[0] == pytest.approx(0.10, abs=0.001)

    def test_compute_sharpe_positive(self):
        from app.validation.metrics_core import compute_sharpe

        # Use varied returns so std > 0
        returns = pd.Series([0.01, 0.02, -0.005, 0.015, 0.008] * 50, dtype=float)
        sharpe = compute_sharpe(returns)
        assert isinstance(sharpe, float)

    def test_compute_sharpe_zero_std(self):
        from app.validation.metrics_core import compute_sharpe

        returns = pd.Series([0.0] * 10, dtype=float)
        assert compute_sharpe(returns) == 0.0

    def test_compute_sortino_basic(self):
        from app.validation.metrics_core import compute_sortino

        returns = pd.Series([0.01, -0.005, 0.02, -0.01], dtype=float)
        sortino = compute_sortino(returns)
        assert isinstance(sortino, float)

    def test_compute_sortino_no_downside(self):
        from app.validation.metrics_core import compute_sortino

        # No negative returns → downside.std() == 0 → returns 0.0
        returns = pd.Series([0.01, 0.02], dtype=float)
        result = compute_sortino(returns)
        # May return 0.0 or nan depending on implementation
        assert result == 0.0 or (
            isinstance(result, float) and (result == 0.0 or result != result)
        )

    def test_compute_max_drawdown_negative(self):
        from app.validation.metrics_core import compute_max_drawdown

        eq = _equity([100, 120, 80, 90])
        dd = compute_max_drawdown(eq)
        assert dd < 0

    def test_compute_profit_factor_all_wins(self):
        from app.validation.metrics_core import compute_profit_factor

        trades = pd.Series([0.05, 0.03, 0.02], dtype=float)
        pf = compute_profit_factor(trades)
        assert pf == float("inf")

    def test_compute_profit_factor_mixed(self):
        from app.validation.metrics_core import compute_profit_factor

        trades = pd.Series([0.05, -0.02, 0.03, -0.01], dtype=float)
        pf = compute_profit_factor(trades)
        assert pf > 1.0

    def test_compute_calmar_positive(self):
        from app.validation.metrics_core import compute_calmar

        eq = _equity([100, 110, 108, 115, 120])
        calmar = compute_calmar(eq)
        assert isinstance(calmar, float)

    def test_compute_calmar_no_drawdown(self):
        from app.validation.metrics_core import compute_calmar

        eq = _equity([100, 110, 120])
        calmar = compute_calmar(eq)
        assert calmar == 0.0

    def test_compute_volatility_basic(self):
        from app.validation.metrics_core import compute_volatility

        returns = pd.Series([0.01, -0.01, 0.02, -0.02], dtype=float)
        vol = compute_volatility(returns)
        assert vol > 0

    def test_compute_all_metrics_without_trades(self):
        from app.validation.metrics_core import compute_all_metrics

        eq = _equity([100, 105, 103, 110])
        result = compute_all_metrics(eq)
        assert "sharpe" in result
        assert "max_drawdown" in result
        assert "calmar" in result
        assert "profit_factor" not in result

    def test_compute_all_metrics_with_trades(self):
        from app.validation.metrics_core import compute_all_metrics

        eq = _equity([100, 105, 103, 110])
        trades = pd.Series([0.05, -0.02, 0.03], dtype=float)
        result = compute_all_metrics(eq, trade_returns=trades)
        assert "profit_factor" in result

    def test_compute_trade_statistics_empty(self):
        from app.validation.metrics_core import compute_trade_statistics

        result = compute_trade_statistics(pd.Series([], dtype=float))
        assert result["trade_count"] == 0
        assert result["winrate"] == 0.0

    def test_compute_trade_statistics_basic(self):
        from app.validation.metrics_core import compute_trade_statistics

        trades = pd.Series([0.05, -0.02, 0.03, -0.01], dtype=float)
        result = compute_trade_statistics(trades)
        assert result["trade_count"] == 4
        assert 0.0 <= result["winrate"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# robustness_self_check.py — private helpers (no IO)
# ─────────────────────────────────────────────────────────────────────────────
class TestRobustnessSelfCheck:
    def test_constraint_enforcement_check(self):
        from app.validation.robustness_self_check import _test_constraint_enforcement

        result = _test_constraint_enforcement()
        assert hasattr(result, "name")
        assert result.name == "constraint_enforcement"
        assert isinstance(result.passed, bool)

    def test_stability_enforcement_check(self):
        from app.validation.robustness_self_check import _test_stability_enforcement

        result = _test_stability_enforcement()
        assert result.name == "stability_enforcement"
        assert isinstance(result.passed, bool)

    def test_self_check_result_dataclass(self):
        from app.validation.robustness_self_check import SelfCheckResult

        r = SelfCheckResult(name="x", passed=True, details={"k": 1})
        assert r.name == "x"
        assert r.passed is True
        assert r.details["k"] == 1

    def test_run_self_checks_mocked_engine_integrity(self):
        from app.validation.robustness_self_check import (
            run_self_checks,
            SelfCheckResult,
        )

        fake = SelfCheckResult(name="engine_integrity", passed=True, details={})
        with patch(
            "app.validation.robustness_self_check._test_engine_integrity",
            return_value=fake,
        ):
            result = run_self_checks()
        assert "status" in result
        assert "checks" in result

    def test_run_self_checks_failure(self):
        from app.validation.robustness_self_check import (
            run_self_checks,
            SelfCheckResult,
        )

        fake = SelfCheckResult(
            name="engine_integrity", passed=False, details={"error": "test"}
        )
        with patch(
            "app.validation.robustness_self_check._test_engine_integrity",
            return_value=fake,
        ):
            result = run_self_checks()
        assert result["status"] == "failed"


# ─────────────────────────────────────────────────────────────────────────────
# post_upgrade_diagnostics.py — pure helpers
# ─────────────────────────────────────────────────────────────────────────────
class TestPostUpgradeDiagnosticsHelpers:
    def test_load_validation_report(self):
        from app.validation.post_upgrade_diagnostics import _load_validation_report

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"status": "ok", "sharpe": 1.2}, f)
            fpath = f.name
        try:
            result = _load_validation_report(fpath)
            assert result["status"] == "ok"
        finally:
            os.unlink(fpath)

    def test_param_cv_mean_empty(self):
        from app.validation.post_upgrade_diagnostics import _param_cv_mean

        assert _param_cv_mean({}) is None
        assert _param_cv_mean({"ga_robustness": {}}) is None

    def test_param_cv_mean_with_values(self):
        from app.validation.post_upgrade_diagnostics import _param_cv_mean

        report = {"ga_robustness": {"param_cv": {"sma_fast": 0.1, "sma_slow": 0.2}}}
        result = _param_cv_mean(report)
        assert result == pytest.approx(0.15)

    def test_rolling_slope_short(self):
        from app.validation.post_upgrade_diagnostics import _rolling_slope

        assert _rolling_slope([1.0]) is None
        assert _rolling_slope([]) is None

    def test_rolling_slope_positive_trend(self):
        from app.validation.post_upgrade_diagnostics import _rolling_slope

        result = _rolling_slope([1.0, 2.0, 3.0, 4.0])
        assert result == pytest.approx(1.0)

    def test_rolling_slope_negative_trend(self):
        from app.validation.post_upgrade_diagnostics import _rolling_slope

        result = _rolling_slope([4.0, 3.0, 2.0, 1.0])
        assert result == pytest.approx(-1.0)

    def test_latest_report_path_no_dir(self):
        from app.validation.post_upgrade_diagnostics import _latest_report_path

        with patch("os.path.isdir", return_value=False):
            assert _latest_report_path() is None

    def test_latest_report_path_no_files(self):
        from app.validation.post_upgrade_diagnostics import _latest_report_path

        with patch("os.path.isdir", return_value=True), patch(
            "os.listdir", return_value=[]
        ):
            assert _latest_report_path() is None

    def test_latest_report_path_returns_last(self):
        from app.validation.post_upgrade_diagnostics import _latest_report_path

        with patch("os.path.isdir", return_value=True), patch(
            "os.listdir", return_value=["validation_a.json", "validation_b.json"]
        ):
            result = _latest_report_path()
        assert result is not None
        assert result.endswith("validation_b.json")

    def test_extract_metrics_empty_report(self):
        from app.validation.post_upgrade_diagnostics import _extract_metrics

        with patch(
            "app.validation.post_upgrade_diagnostics.analyze_trade_quality",
            return_value={},
        ):
            result = _extract_metrics({}, None)
        assert isinstance(result, dict)

    def test_extract_metrics_rolling_slope(self):
        from app.validation.post_upgrade_diagnostics import _extract_metrics

        report = {
            "walk_forward": {
                "mean_oos_sharpe": 0.8,
                "wf_summary": {"rolling_sharpe_trend": [1.0, 2.0, 3.0]},
            }
        }
        wf = {"wf_summary": {"rolling_sharpe_trend": [1.0, 2.0, 3.0]}}
        with patch(
            "app.validation.post_upgrade_diagnostics.analyze_trade_quality",
            return_value={},
        ):
            result = _extract_metrics(report, wf, compute_trade_quality=False)
        assert isinstance(result, dict)


# ─────────────────────────────────────────────────────────────────────────────
# drift_monitor.py — pure variant-price helper
# ─────────────────────────────────────────────────────────────────────────────
class TestDriftMonitorHelpers:
    def _make_trade(self):
        t = MagicMock()
        t.entry_idx = 1
        t.exit_idx = 3
        return t

    def test_close_reference_variant(self):
        from app.validation.drift_monitor import _variant_prices, ExecutionVariant

        closes = [100.0, 101.0, 102.0, 103.0, 104.0]
        opens = [99.0, 100.5, 101.5, 102.5, 103.5]
        t = self._make_trade()
        prices = _variant_prices(t, closes, opens, ExecutionVariant.CLOSE_REFERENCE)
        assert prices == (101.0, 103.0)

    def test_same_bar_open_variant(self):
        from app.validation.drift_monitor import _variant_prices, ExecutionVariant

        closes = [100.0, 101.0, 102.0, 103.0, 104.0]
        opens = [99.0, 100.5, 101.5, 102.5, 103.5]
        t = self._make_trade()
        prices = _variant_prices(t, closes, opens, ExecutionVariant.SAME_BAR_OPEN)
        assert prices == (100.5, 102.5)

    def test_next_open_shift_variant(self):
        from app.validation.drift_monitor import _variant_prices, ExecutionVariant

        closes = [100.0] * 10
        opens = [99.0, 100.5, 101.5, 102.5, 103.5, 104.0, 105.0, 106.0, 107.0, 108.0]
        t = self._make_trade()
        prices = _variant_prices(t, closes, opens, ExecutionVariant.NEXT_OPEN_SHIFT)
        assert prices is not None

    def test_next_open_shift_out_of_bounds(self):
        from app.validation.drift_monitor import _variant_prices, ExecutionVariant

        closes = [100.0, 101.0, 102.0, 103.0]
        opens = [99.0, 100.5, 101.5, 102.5]
        t = (
            self._make_trade()
        )  # entry_idx=1, exit_idx=3; shift would be at 4 which is out of bounds
        prices = _variant_prices(t, closes, opens, ExecutionVariant.NEXT_OPEN_SHIFT)
        assert prices is None

    def test_hybrid_test_variant(self):
        from app.validation.drift_monitor import _variant_prices, ExecutionVariant

        closes = [100.0, 101.0, 102.0, 103.0, 104.0]
        opens = [99.0, 100.5, 101.5, 102.5, 103.5]
        t = self._make_trade()
        prices = _variant_prices(t, closes, opens, ExecutionVariant.HYBRID_TEST)
        assert prices is not None

    def test_unknown_variant_returns_none(self):
        from app.validation.drift_monitor import _variant_prices

        t = self._make_trade()
        assert _variant_prices(t, [], [], "UNKNOWN") is None


# ─────────────────────────────────────────────────────────────────────────────
# validation/utils.py
# ─────────────────────────────────────────────────────────────────────────────
class TestValidationUtils:
    def test_get_validation_ticker_from_env(self):
        from app.validation.utils import get_validation_ticker

        with patch.dict("os.environ", {"VALIDATION_TICKER": "MSFT"}):
            assert get_validation_ticker() == "MSFT"

    def test_get_validation_ticker_default(self):
        from app.validation.utils import get_validation_ticker

        env = {k: v for k, v in os.environ.items() if k != "VALIDATION_TICKER"}
        with patch.dict("os.environ", env, clear=True):
            ticker = get_validation_ticker()
        assert isinstance(ticker, str)
        assert len(ticker) > 0

    def test_get_validation_window_returns_dates(self):
        from app.validation.utils import get_validation_window

        start, end = get_validation_window()
        assert isinstance(start, date)
        assert isinstance(end, date)
        assert end > start


# ─────────────────────────────────────────────────────────────────────────────
# risk_stress.py — if it has pure helpers
# ─────────────────────────────────────────────────────────────────────────────
class TestRiskStress:
    def test_import_succeeds(self):
        import app.validation.risk_stress as rs

        assert rs is not None

    def test_run_risk_stress_tests_no_data(self):
        from app.validation.risk_stress import run_risk_stress_tests

        with patch("app.validation.risk_stress.load_data", return_value=None), patch(
            "app.validation.risk_stress.get_validation_ticker", return_value="VOO"
        ), patch(
            "app.validation.risk_stress.get_validation_window",
            return_value=(
                __import__("datetime").date(2022, 1, 1),
                __import__("datetime").date(2022, 12, 31),
            ),
        ):
            result = run_risk_stress_tests()
        assert result["status"] == "no_data"
