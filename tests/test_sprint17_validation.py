"""S17C: Validation/governance coverage boost.

Targets:
- ValidationRunner (was 16%)
- pipeline_audit helpers (was 48%)
- data_integrity_check (already 90% - minor additions)
- execution_stress helpers (was 21%)
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# ValidationRunner
# ─────────────────────────────────────────────────────────────────────────────


class TestValidationRunner:
    def test_init_sets_mode(self):
        from app.validation.validation_runner import ValidationRunner

        vr = ValidationRunner(mode="quick")
        assert vr.mode == "quick"

    def test_run_bias_tests_sets_result(self):
        from app.validation.validation_runner import ValidationRunner

        vr = ValidationRunner(mode="quick")
        with patch(
            "app.validation.bias_tests.run_bias_tests", return_value={"ok": True}
        ):
            vr.run_bias_tests()
        assert "bias" in vr.results

    def test_run_ga_robustness_sets_result(self):
        from app.validation.validation_runner import ValidationRunner

        vr = ValidationRunner(mode="full")
        with patch(
            "app.validation.ga_robustness.run_ga_robustness_tests",
            return_value={"passed": True},
        ):
            vr.run_ga_robustness()
        assert "ga_robustness" in vr.results

    def test_run_risk_stress_sets_result(self):
        from app.validation.validation_runner import ValidationRunner

        vr = ValidationRunner(mode="full")
        with patch(
            "app.validation.risk_stress.run_risk_stress_tests",
            return_value={"stress": "ok"},
        ):
            vr.run_risk_stress()
        assert "risk" in vr.results

    def test_run_shadow_comparison_sets_result(self):
        from app.validation.validation_runner import ValidationRunner

        vr = ValidationRunner(mode="shadow")
        with patch(
            "app.validation.shadow_compare.run_shadow_comparison",
            return_value={"equity_diff": 0},
        ):
            vr.run_shadow_comparison()
        assert "shadow" in vr.results

    def test_run_execution_sensitivity_sets_result(self):
        from app.validation.validation_runner import ValidationRunner

        vr = ValidationRunner(mode="quick")
        with patch(
            "app.validation.execution_sensitivity.run_execution_sensitivity",
            return_value={"result": "ok"},
        ):
            vr.run_execution_sensitivity()
        assert "execution_sensitivity" in vr.results

    def test_update_engine_status_valid(self):
        from app.validation.validation_runner import ValidationRunner

        vr = ValidationRunner(mode="quick")
        vr.results = {
            "bias": {
                "engine_integrity": {
                    "trade_count_match": True,
                    "trade_index_match": True,
                },
                "relative_gap": 0.0,
            },
            "shadow": {"equity_diff": 0},
        }
        vr._update_engine_status()
        assert vr.results["engine_integrity"]["status"] == "ENGINE_VALID"
        assert vr.results["validation_status"] == "ENGINE_VALID"

    def test_update_engine_status_trade_count_mismatch(self):
        from app.validation.validation_runner import ValidationRunner

        vr = ValidationRunner(mode="quick")
        vr.results = {
            "bias": {
                "engine_integrity": {
                    "trade_count_match": False,
                    "trade_index_match": True,
                },
            },
            "shadow": {},
        }
        vr._update_engine_status()
        assert vr.results["engine_integrity"]["status"] == "ENGINE_INVALID"
        assert "trade_count_mismatch" in vr.results["engine_integrity"]["issues"]

    def test_update_engine_status_equity_diff(self):
        from app.validation.validation_runner import ValidationRunner

        vr = ValidationRunner(mode="quick")
        vr.results = {
            "bias": {"engine_integrity": {}},
            "shadow": {"equity_diff": 500.0},
        }
        vr._update_engine_status()
        assert vr.results["engine_integrity"]["status"] == "ENGINE_INVALID"

    def test_execute_quick_mode_calls_bias_and_sensitivity(self):
        from app.validation.validation_runner import ValidationRunner

        vr = ValidationRunner(mode="quick")
        with patch.object(vr, "run_bias_tests") as m_bias, patch.object(
            vr, "run_execution_sensitivity"
        ) as m_sens, patch.object(vr, "_update_engine_status"), patch.object(
            vr, "compute_final_score"
        ):
            with patch("app.validation.validation_runner.seed_deterministic"), patch(
                "app.validation.validation_runner.random"
            ):
                vr.execute()
        m_bias.assert_called_once()
        m_sens.assert_called_once()

    def test_compute_final_score_calls_compute_quant_score(self):
        from app.validation.validation_runner import ValidationRunner

        vr = ValidationRunner(mode="quick")
        vr.results = {"bias": {}}
        with patch(
            "app.validation.validation_runner.compute_quant_score",
            return_value={"score": 0.8},
        ) as mock_score:
            vr.compute_final_score()
        mock_score.assert_called_once_with(vr.results)
        assert "final_score" in vr.results


# ─────────────────────────────────────────────────────────────────────────────
# pipeline_audit helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestPipelineAuditHelpers:
    def _make_df(self, rows=50):
        idx = pd.date_range("2024-01-01", periods=rows, freq="B")
        return pd.DataFrame(
            {
                "Open": np.linspace(100, 120, rows),
                "High": np.linspace(101, 122, rows),
                "Low": np.linspace(99, 118, rows),
                "Close": np.linspace(100, 120, rows),
                "Volume": [1_000_000] * rows,
            },
            index=idx,
        )

    def test_lookback_from_params_basic(self):
        from app.validation.pipeline_audit import _lookback_from_params

        params = {"sma_period": 20, "rsi_period": 14, "macd_slow": 26, "macd_signal": 9}
        lb = _lookback_from_params(params)
        assert lb == max(20, 14, 26 + 9)  # 35

    def test_lookback_from_params_adx(self):
        from app.validation.pipeline_audit import _lookback_from_params

        params = {"adx_period": 14}
        lb = _lookback_from_params(params)
        assert lb == 14 * 2 + 1

    def test_audit_fold_returns_dict(self, test_settings):
        from app.validation.pipeline_audit import _audit_fold
        from app.validation import set_settings

        set_settings(test_settings)
        df = self._make_df()
        params = {"sma_period": 10}
        result = _audit_fold(df, "TEST", params, fold_id=0)
        assert isinstance(result, dict)
        assert "fold_id" in result
        assert result["fold_id"] == 0

    def test_audit_fold_empty_df(self, test_settings):
        from app.validation.pipeline_audit import _audit_fold
        from app.validation import set_settings

        set_settings(test_settings)
        empty = pd.DataFrame()
        result = _audit_fold(empty, "TEST", {}, fold_id=1)
        assert result["raw_rows"] == 0

    def test_run_pipeline_audit_no_data(self, test_settings):
        from app.validation.pipeline_audit import run_pipeline_audit
        from app.validation import set_settings

        set_settings(test_settings)
        with patch("app.validation.pipeline_audit.load_data", return_value=None), patch(
            "app.validation.pipeline_audit.get_validation_ticker", return_value="TEST"
        ), patch("app.validation.pipeline_audit.get_params", return_value={}), patch(
            "app.validation.pipeline_audit.get_validation_window",
            return_value=(pd.Timestamp("2022-01-01"), pd.Timestamp("2023-01-01")),
        ):
            result = run_pipeline_audit(ticker="TEST")
        assert result["status"] == "no_data"


# ─────────────────────────────────────────────────────────────────────────────
# execution_stress helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestExecutionStressHelpers:
    def _make_df(self, rows=60):
        idx = pd.date_range("2024-01-01", periods=rows, freq="B")
        close = np.linspace(100, 115, rows)
        return pd.DataFrame(
            {
                "Open": close * 0.99,
                "High": close * 1.02,
                "Low": close * 0.98,
                "Close": close,
                "Volume": [500_000] * rows,
            },
            index=idx,
        )

    def test_scenario_prices_close_reference(self):
        from app.validation.execution_stress import _scenario_prices, ExecutionScenario

        trade = MagicMock()
        trade.entry_idx = 5
        trade.exit_idx = 10
        closes = np.arange(20, dtype=float)
        opens = np.arange(20, dtype=float) - 0.5
        result = _scenario_prices(
            trade, closes, opens, ExecutionScenario.CLOSE_REFERENCE
        )
        assert result == (5.0, 10.0)

    def test_scenario_prices_baseline(self):
        from app.validation.execution_stress import _scenario_prices, ExecutionScenario

        trade = MagicMock()
        trade.entry_idx = 3
        trade.exit_idx = 7
        closes = np.linspace(100, 110, 20)
        opens = np.linspace(99, 109, 20)
        result = _scenario_prices(
            trade, closes, opens, ExecutionScenario.BASELINE_NEXT_OPEN
        )
        assert result is not None
        # entry_at=4, exit_at=8
        assert abs(result[0] - opens[4]) < 0.01

    def test_scenario_prices_out_of_bounds(self):
        from app.validation.execution_stress import _scenario_prices, ExecutionScenario

        trade = MagicMock()
        trade.entry_idx = 18
        trade.exit_idx = 19
        closes = np.linspace(100, 110, 20)
        opens = np.linspace(99, 109, 20)
        # entry_at=19, exit_at=20 — exit_at is out of bounds → fallback to closes
        result = _scenario_prices(
            trade, closes, opens, ExecutionScenario.BASELINE_NEXT_OPEN
        )
        # Should return closes fallback
        assert result is not None

    def test_equity_metrics_empty_trades(self):
        from app.validation.execution_stress import _equity_metrics

        result = _equity_metrics([], initial_capital=10000)
        assert result["trade_count"] == 0
        assert result["total_return"] == 0.0

    def test_equity_metrics_with_returns(self):
        from app.validation.execution_stress import _equity_metrics

        returns = [0.01, 0.02, -0.01, 0.015]
        result = _equity_metrics(returns, initial_capital=10000)
        assert result["trade_count"] == 4
        assert "sharpe" in result
        assert "max_drawdown" in result

    def test_equity_hash_deterministic(self):
        from app.validation.execution_stress import _equity_hash

        values = [100.0, 105.3, 102.1]
        h1 = _equity_hash(values)
        h2 = _equity_hash(values)
        assert h1 == h2

    def test_evaluate_execution_stress_no_trades(self, test_settings):
        from app.validation.execution_stress import evaluate_execution_stress
        from app.validation import set_settings

        set_settings(test_settings)
        df = self._make_df()
        with patch(
            "app.validation.execution_stress.get_params", return_value={}
        ), patch("app.validation.execution_stress._trade_indices", return_value=[]):
            result = evaluate_execution_stress(df, "TEST")
        assert result["status"] == "no_trades"
        assert result["stress_tested"] is False


# ─────────────────────────────────────────────────────────────────────────────
# data_integrity_check supplemental
# ─────────────────────────────────────────────────────────────────────────────


class TestDataIntegrityCheckSupplemental:
    def _make_df(self, rows=30, add_gaps=False):
        if add_gaps:
            # Create index with a gap > 3 days
            idx = pd.date_range("2024-01-01", periods=20, freq="B").tolist()
            idx += pd.date_range("2024-02-15", periods=10, freq="B").tolist()
            idx = pd.DatetimeIndex(idx)
        else:
            idx = pd.date_range("2024-01-01", periods=rows, freq="B")
        n = len(idx)
        return pd.DataFrame(
            {
                "Open": np.linspace(100, 120, n),
                "High": np.linspace(101, 122, n),
                "Low": np.linspace(99, 118, n),
                "Close": np.linspace(100, 120, n),
                "Volume": [1_000_000] * n,
            },
            index=idx,
        )

    def test_gap_detected(self):
        from app.validation.data_integrity_check import run_data_integrity_checks

        df = self._make_df(add_gaps=True)
        result = run_data_integrity_checks(df, lookback=10)
        assert result["gap_ratio"] > 0

    def test_constant_price_streak_detected(self):
        from app.validation.data_integrity_check import run_data_integrity_checks

        idx = pd.date_range("2024-01-01", periods=20, freq="B")
        close = [100.0] * 20  # all same → streak of 20
        df = pd.DataFrame(
            {
                "Open": close,
                "High": close,
                "Low": close,
                "Close": close,
                "Volume": [1e6] * 20,
            },
            index=idx,
        )
        result = run_data_integrity_checks(df, lookback=5)
        assert result["constant_price_streaks"] > 0

    def test_run_no_volume_column(self):
        from app.validation.data_integrity_check import run_data_integrity_checks

        idx = pd.date_range("2024-01-01", periods=20, freq="B")
        df = pd.DataFrame(
            {
                "Open": [100.0] * 20,
                "High": [101.0] * 20,
                "Low": [99.0] * 20,
                "Close": [100.0] * 20,
            },
            index=idx,
        )
        result = run_data_integrity_checks(df, lookback=5)
        assert result["zero_volume_streaks"] == 0
