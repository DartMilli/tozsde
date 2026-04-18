"""S17E: core/analysis coverage boost.

Targets:
- app/core/analysis/go_live_metrics.py (38%)
- app/core/analysis/confidence_calibrator.py (24%)
- app/core/analysis/phase6_validator.py (33%)
- app/core/analysis/safety_stress_tester.py (22%)
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch, call
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# go_live_metrics
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeDrawdownSummary:
    def _make_dm(self, rows):
        dm = MagicMock()
        conn = MagicMock()
        conn.execute.return_value.fetchall.return_value = rows
        dm.connection.return_value.__enter__ = lambda s: conn
        dm.connection.return_value.__exit__ = MagicMock(return_value=False)
        return dm

    def test_no_rows_returns_no_data(self):
        from app.core.analysis.go_live_metrics import compute_drawdown_summary

        dm = self._make_dm([])
        result = compute_drawdown_summary("AAPL", data_manager=dm)
        assert result["status"] == "no_data"
        assert result["rows"] == 0

    def test_single_positive_row(self):
        from app.core.analysis.go_live_metrics import compute_drawdown_summary

        dm = self._make_dm([("2024-01-01", 0.05)])
        result = compute_drawdown_summary("AAPL", data_manager=dm)
        assert result["status"] == "ok"
        assert result["max_drawdown"] == 0.0
        assert result["rows"] == 1

    def test_drawdown_detected(self):
        from app.core.analysis.go_live_metrics import compute_drawdown_summary

        rows = [
            ("2024-01-01", 0.05),
            ("2024-01-02", -0.10),
        ]
        dm = self._make_dm(rows)
        result = compute_drawdown_summary("AAPL", data_manager=dm)
        assert result["max_drawdown"] > 0.0

    def test_returns_roundable_floats(self):
        from app.core.analysis.go_live_metrics import compute_drawdown_summary

        rows = [("2024-01-01", 0.01), ("2024-01-02", 0.02)]
        dm = self._make_dm(rows)
        result = compute_drawdown_summary("AAPL", data_manager=dm)
        assert isinstance(result["max_drawdown"], float)


class TestComputeLossStreak:
    def _make_dm(self, rows):
        dm = MagicMock()
        conn = MagicMock()
        conn.execute.return_value.fetchall.return_value = rows
        dm.connection.return_value.__enter__ = lambda s: conn
        dm.connection.return_value.__exit__ = MagicMock(return_value=False)
        return dm

    def test_no_rows_returns_zero_streak(self):
        from app.core.analysis.go_live_metrics import compute_loss_streak

        dm = self._make_dm([])
        result = compute_loss_streak("AAPL", data_manager=dm)
        assert result["status"] == "no_data"
        assert result["loss_streak"] == 0

    def test_pure_wins_streak_zero(self):
        from app.core.analysis.go_live_metrics import compute_loss_streak

        rows = [("2024-01-01", 1), ("2024-01-02", 1)]
        dm = self._make_dm(rows)
        result = compute_loss_streak("AAPL", data_manager=dm)
        assert result["loss_streak"] == 0

    def test_consecutive_losses(self):
        from app.core.analysis.go_live_metrics import compute_loss_streak

        rows = [("2024-01-05", 0), ("2024-01-04", 0), ("2024-01-03", 1)]
        dm = self._make_dm(rows)
        result = compute_loss_streak("AAPL", data_manager=dm)
        assert result["loss_streak"] == 2

    def test_null_success_breaks_streak(self):
        from app.core.analysis.go_live_metrics import compute_loss_streak

        rows = [("2024-01-01", None)]
        dm = self._make_dm(rows)
        result = compute_loss_streak("AAPL", data_manager=dm)
        assert result["loss_streak"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# ConfidenceCalibrator.apply and load_latest_params
# ─────────────────────────────────────────────────────────────────────────────


class TestConfidenceCalibratorApply:
    def _make_calibrator(self, enable_calibration=True):
        from app.core.analysis.confidence_calibrator import ConfidenceCalibrator

        settings = MagicMock()
        settings.ENABLE_CONFIDENCE_CALIBRATION = enable_calibration
        dm = MagicMock()
        return ConfidenceCalibrator(settings=settings, data_manager=dm)

    def test_apply_disabled_returns_raw(self):
        cal = self._make_calibrator(enable_calibration=False)
        assert cal.apply(0.7, {}) == 0.7

    def test_apply_empty_params_returns_raw(self):
        cal = self._make_calibrator(enable_calibration=True)
        assert cal.apply(0.7, {}) == 0.7

    def test_apply_with_valid_params(self):
        import numpy as np

        cal = self._make_calibrator(enable_calibration=True)
        params = {
            "x_thresholds": [0.0, 0.5, 1.0],
            "y_thresholds": [0.0, 0.6, 1.0],
        }
        result = cal.apply(0.5, params)
        assert abs(result - 0.6) < 0.01

    def test_load_latest_params_no_row(self):
        cal = self._make_calibrator()
        cal.dm.fetch_latest_confidence_calibration.return_value = None
        assert cal.load_latest_params("AAPL") == {}

    def test_load_latest_params_empty_json(self):
        cal = self._make_calibrator()
        cal.dm.fetch_latest_confidence_calibration.return_value = {"params_json": ""}
        assert cal.load_latest_params("AAPL") == {}

    def test_load_latest_params_valid_json(self):
        import json

        cal = self._make_calibrator()
        params = {"x_thresholds": [0.0, 1.0], "y_thresholds": [0.0, 1.0]}
        cal.dm.fetch_latest_confidence_calibration.return_value = {
            "params_json": json.dumps(params)
        }
        result = cal.load_latest_params("AAPL")
        assert result == params

    def test_load_latest_params_bad_json_returns_empty(self):
        cal = self._make_calibrator()
        cal.dm.fetch_latest_confidence_calibration.return_value = {
            "params_json": "{bad json"
        }
        assert cal.load_latest_params("AAPL") == {}

    def test_compute_empty_returns_no_data_result(self):
        import pandas as pd
        from app.core.analysis.confidence_calibrator import ConfidenceCalibrator

        settings = MagicMock()
        settings.ENABLE_CONFIDENCE_CALIBRATION = True
        dm = MagicMock()

        cal = ConfidenceCalibrator(settings=settings, data_manager=dm)

        empty_df = pd.DataFrame(columns=["confidence", "success"])
        with patch.object(cal, "_load_data", return_value=empty_df):
            result = cal.compute(ticker="AAPL")

        assert result.metrics["status"] == "no_data"
        dm.save_confidence_calibration.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# Phase6Validator
# ─────────────────────────────────────────────────────────────────────────────


class TestPhase6Validator:
    def _make_validator(self):
        from app.core.analysis.phase6_validator import Phase6Validator

        dm = MagicMock()
        settings = MagicMock()
        settings.INITIAL_CAPITAL = 10000.0
        return Phase6Validator(dm=dm, settings=settings), dm

    def test_run_check_effectiveness_no_data(self):
        validator, dm = self._make_validator()
        conn = MagicMock()
        conn.execute.return_value.fetchall.return_value = []
        dm.connection.return_value.__enter__ = lambda s: conn
        dm.connection.return_value.__exit__ = MagicMock(return_value=False)
        with patch(
            "app.core.analysis.phase6_validator.DecisionEffectivenessAnalyzer"
        ) as mock_dea:
            mock_dea.return_value.compute_for_range.return_value = None
            result = validator._check_effectiveness("AAPL")
        assert result["status"] == "no_data"

    def test_run_returns_dict_with_expected_keys(self):
        validator, dm = self._make_validator()
        conn = MagicMock()
        conn.execute.return_value.fetchall.return_value = []
        dm.connection.return_value.__enter__ = lambda s: conn
        dm.connection.return_value.__exit__ = MagicMock(return_value=False)
        # Only test _check_effectiveness; skip full run to avoid complex mocking
        with patch(
            "app.core.analysis.phase6_validator.DecisionEffectivenessAnalyzer"
        ) as mock_dea:
            mock_dea.return_value.compute_for_range.return_value = None
            result = validator._check_effectiveness("AAPL")
        assert result["status"] == "no_data"
        # Verify run() returns the right keys structure via partial mock
        with patch.object(
            validator, "_check_effectiveness", return_value={"status": "no_data"}
        ), patch.object(
            validator, "_check_position_sizing", return_value={"status": "ok"}
        ), patch.object(
            validator, "_check_model_trust", return_value={"status": "ok"}
        ), patch.object(
            validator, "_check_reward_shaping", return_value={"status": "ok"}
        ), patch.object(
            validator, "_check_promotion_gate", return_value={"status": "ok"}
        ):
            result = validator.run("AAPL")
        assert "p6_1_effectiveness" in result
        assert "p6_2_position_sizing" in result
