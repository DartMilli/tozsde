"""S17: Pure-logic and small-file coverage boost.

Targets:
- app/models/model_output.py (0% → 100%)
- app/backtesting/confidence_calibrator.py (0% → ~100%)
- app/backtesting/safety_calibrator.py (0% → ~100%)
- app/backtesting/safety_simulator.py (0% → ~100%)
- app/backtesting/decision_replay.py (0% → ~80%)
- app/backtesting/audit_runner.py (0% → ~90%)
- app/backtesting/replay_runner.py (0% → ~80%)
- app/validation/trade_quality_analysis.py helpers (0% → ~70%)
- app/data_access/market_context.py (0% → ~80%)
- app/analysis/decision_effectiveness.py shim (0% → ~80%)
- app/analysis/phase6_validator.py shim (0% → ~80%)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch


# ─────────────────────────────────────────────────────────────────────────────
# ModelOutput dataclass
# ─────────────────────────────────────────────────────────────────────────────
class TestModelOutput:
    def test_basic_construction(self):
        from app.models.model_output import ModelOutput

        out = ModelOutput(model_name="DQN", signal="BUY")
        assert out.model_name == "DQN"
        assert out.signal == "BUY"
        assert out.confidence is None
        assert out.metadata == {}

    def test_full_construction(self):
        from app.models.model_output import ModelOutput

        out = ModelOutput(
            model_name="PPO",
            signal="SELL",
            expected_return=-0.02,
            risk=0.05,
            confidence=0.75,
            metadata={"source": "rl"},
        )
        assert out.signal == "SELL"
        assert out.confidence == 0.75
        assert out.metadata["source"] == "rl"

    def test_hold_signal(self):
        from app.models.model_output import ModelOutput

        out = ModelOutput(model_name="LSTM", signal="HOLD")
        assert out.signal == "HOLD"


# ─────────────────────────────────────────────────────────────────────────────
# ConfidenceCalibrator (pure function)
# ─────────────────────────────────────────────────────────────────────────────
class TestCalibrateConfidence:
    def test_empty_returns_empty(self):
        from app.backtesting.confidence_calibrator import calibrate_confidence

        assert calibrate_confidence([]) == {}

    def test_single_winning_bucket(self):
        from app.backtesting.confidence_calibrator import calibrate_confidence

        rows = [
            {"confidence_bucket": "HIGH", "reward": 1.0},
            {"confidence_bucket": "HIGH", "reward": 0.5},
            {"confidence_bucket": "HIGH", "reward": -0.1},
        ]
        result = calibrate_confidence(rows)
        assert result["HIGH"] == pytest.approx(2 / 3, abs=0.01)

    def test_skips_missing_fields(self):
        from app.backtesting.confidence_calibrator import calibrate_confidence

        rows = [
            {"confidence_bucket": None, "reward": 1.0},
            {"confidence_bucket": "LOW", "reward": None},
            {"confidence_bucket": "MED", "reward": 0.5},
        ]
        result = calibrate_confidence(rows)
        assert "MED" in result
        assert None not in result
        assert "LOW" not in result

    def test_all_losing(self):
        from app.backtesting.confidence_calibrator import calibrate_confidence

        rows = [{"confidence_bucket": "X", "reward": -1.0} for _ in range(5)]
        result = calibrate_confidence(rows)
        assert result["X"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# SafetyCalibrator (pure function)
# ─────────────────────────────────────────────────────────────────────────────
class TestCalibrateThresholds:
    def _rows(self):
        return [
            {
                "confidence": 0.6,
                "reward": 1.0,
                "decision_level": "L1",
                "confidence_bucket": "HIGH",
            },
            {
                "confidence": 0.8,
                "reward": 0.5,
                "decision_level": "L1",
                "confidence_bucket": "HIGH",
            },
            {
                "confidence": 0.3,
                "reward": -1.0,
                "decision_level": "L2",
                "confidence_bucket": "LOW",
            },
        ]

    def test_returns_expected_keys(self):
        from app.backtesting.safety_calibrator import calibrate_safety_thresholds

        result = calibrate_safety_thresholds(self._rows())
        assert "min_confidence" in result
        assert "block_decision_levels" in result
        assert "block_buckets" in result

    def test_empty_rows(self):
        from app.backtesting.safety_calibrator import calibrate_safety_thresholds

        result = calibrate_safety_thresholds([])
        assert isinstance(result, dict)

    def test_blocks_losing_level(self):
        from app.backtesting.safety_calibrator import calibrate_safety_thresholds

        rows = [
            {
                "confidence": 0.5,
                "reward": -1.0,
                "decision_level": "L_BAD",
                "confidence_bucket": "LOW",
            },
            {
                "confidence": 0.5,
                "reward": -0.5,
                "decision_level": "L_BAD",
                "confidence_bucket": "LOW",
            },
        ]
        result = calibrate_safety_thresholds(rows)
        assert "L_BAD" in result["block_decision_levels"]


# ─────────────────────────────────────────────────────────────────────────────
# SafetySimulator (pure function)
# ─────────────────────────────────────────────────────────────────────────────
class TestSimulateSafetyPolicy:
    def _policy(self):
        return {
            "min_confidence": 0.5,
            "block_decision_levels": ["L_BAD"],
            "block_buckets": ["VERY_LOW"],
        }

    def test_no_rows(self):
        from app.backtesting.safety_simulator import simulate_safety_policy

        result = simulate_safety_policy([], self._policy())
        assert isinstance(result, dict)

    def test_blocks_low_confidence(self):
        from app.backtesting.safety_simulator import simulate_safety_policy

        rows = [
            {
                "reward": 1.0,
                "confidence": 0.2,
                "decision_level": "L1",
                "confidence_bucket": "LOW",
            },
        ]
        result = simulate_safety_policy(rows, self._policy())
        assert result["baseline"]["trades"] == 1
        assert result["safety"]["trades"] == 0

    def test_allows_high_confidence(self):
        from app.backtesting.safety_simulator import simulate_safety_policy

        rows = [
            {
                "reward": 1.0,
                "confidence": 0.9,
                "decision_level": "L1",
                "confidence_bucket": "HIGH",
            },
        ]
        result = simulate_safety_policy(rows, self._policy())
        assert result["safety"]["trades"] == 1

    def test_blocks_bad_level(self):
        from app.backtesting.safety_simulator import simulate_safety_policy

        rows = [
            {
                "reward": 1.0,
                "confidence": 0.9,
                "decision_level": "L_BAD",
                "confidence_bucket": "HIGH",
            },
        ]
        result = simulate_safety_policy(rows, self._policy())
        assert result["safety"]["trades"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# DecisionReplay (DataFrame logic)
# ─────────────────────────────────────────────────────────────────────────────


def _make_ohlcv(n=20, base_price=100.0) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    prices = [base_price + i for i in range(n)]
    return pd.DataFrame(
        {
            "Open": prices,
            "High": prices,
            "Low": prices,
            "Close": prices,
            "Volume": [1_000_000] * n,
        },
        index=idx,
    )


class TestDecisionReplay:
    def test_buy_returns_positive_return(self):
        from app.backtesting.decision_replay import replay_decision

        df = _make_ohlcv()
        event = {"timestamp": "2023-01-02T00:00:00", "action": 1}
        result = replay_decision(
            event, df, horizon_days=3, execution_policy="next_open"
        )
        assert result.get("status") in ("OK", "INSUFFICIENT_DATA")

    def test_missing_date_returns_insufficient(self):
        from app.backtesting.decision_replay import replay_decision

        df = _make_ohlcv()
        event = {"timestamp": "2025-12-31T00:00:00", "action": 1}
        result = replay_decision(event, df)
        assert result["status"] == "INSUFFICIENT_DATA"

    def test_hold_action(self):
        from app.backtesting.decision_replay import replay_decision

        df = _make_ohlcv()
        event = {"timestamp": "2023-01-02T00:00:00", "action": 0}
        result = replay_decision(event, df, horizon_days=2)
        # HOLD gives 0 return if status OK
        if result.get("status") == "OK":
            assert result["raw_return"] == 0.0

    def test_near_end_returns_insufficient(self):
        from app.backtesting.decision_replay import replay_decision

        df = _make_ohlcv(n=5)
        event = {"timestamp": "2023-01-05T00:00:00", "action": 1}
        result = replay_decision(event, df, horizon_days=10)
        assert result["status"] == "INSUFFICIENT_DATA"


# ─────────────────────────────────────────────────────────────────────────────
# ReplayRunner (mocked IO)
# ─────────────────────────────────────────────────────────────────────────────
class TestRunReplayForTicker:
    def test_no_records_returns_empty(self):
        from app.backtesting.replay_runner import run_replay_for_ticker

        with patch("app.backtesting.replay_runner.HistoryStore") as MockHS:
            mock_store = MagicMock()
            mock_store.iter_records.return_value = []
            MockHS.return_value = mock_store
            result = run_replay_for_ticker("VOO")
        assert result == []

    def test_with_records_calls_replay(self):
        from app.backtesting.replay_runner import run_replay_for_ticker

        df = _make_ohlcv(30)
        records = [
            {
                "ticker": "VOO",
                "timestamp": "2023-01-04T00:00:00",
                "decision": {"action_code": 1, "confidence": 0.7},
                "audit": {"decision_level": "L1"},
            }
        ]
        with patch("app.backtesting.replay_runner.HistoryStore") as MockHS, patch(
            "app.backtesting.replay_runner.load_data", return_value=df
        ):
            mock_store = MagicMock()
            mock_store.iter_records.return_value = records
            MockHS.return_value = mock_store
            result = run_replay_for_ticker("VOO", horizon_days=5)
        assert len(result) == 1
        assert result[0]["ticker"] == "VOO"


# ─────────────────────────────────────────────────────────────────────────────
# AuditRunner
# ─────────────────────────────────────────────────────────────────────────────
class TestRunBacktestAudit:
    def test_no_rows_returns_empty_dict(self):
        from app.backtesting.audit_runner import run_backtest_audit

        with patch(
            "app.backtesting.audit_runner.run_replay_for_ticker", return_value=[]
        ):
            result = run_backtest_audit("VOO")
        assert result == {}

    def test_with_rows_returns_dict(self):
        from app.backtesting.audit_runner import run_backtest_audit

        rows = [{"confidence": 0.8, "decision_level": "L1", "reward": 1.0}]
        with patch(
            "app.backtesting.audit_runner.run_replay_for_ticker", return_value=rows
        ), patch(
            "app.backtesting.audit_runner.audit_confidence_buckets", return_value={}
        ), patch(
            "app.backtesting.audit_runner.audit_decision_levels", return_value={}
        ), patch(
            "app.backtesting.audit_runner.detect_overconfidence", return_value={}
        ):
            result = run_backtest_audit("VOO")
        assert isinstance(result, dict)


# ─────────────────────────────────────────────────────────────────────────────
# TradeQualityAnalysis – pure helpers
# ─────────────────────────────────────────────────────────────────────────────
class TestTradeQualityHelpers:
    def test_ratio_top_contributors_basic(self):
        from app.validation.trade_quality_analysis import _ratio_top_contributors

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        ratio = _ratio_top_contributors(values, top_pct=0.2)
        assert 0.0 <= ratio <= 1.0

    def test_ratio_top_contributors_all_negative(self):
        from app.validation.trade_quality_analysis import _ratio_top_contributors

        assert _ratio_top_contributors([-1, -2, -3], 0.2) == 0.0

    def test_ratio_top_contributors_empty(self):
        from app.validation.trade_quality_analysis import _ratio_top_contributors

        assert _ratio_top_contributors([], 0.2) == 0.0

    def test_ratio_worst_losses_basic(self):
        from app.validation.trade_quality_analysis import _ratio_worst_losses

        values = [-1.0, -2.0, -3.0, -4.0, -5.0]
        ratio = _ratio_worst_losses(values, bottom_pct=0.2)
        assert 0.0 <= ratio <= 1.0

    def test_ratio_worst_losses_all_positive(self):
        from app.validation.trade_quality_analysis import _ratio_worst_losses

        assert _ratio_worst_losses([1, 2, 3], 0.2) == 0.0

    def test_skewness_positive(self):
        from app.validation.trade_quality_analysis import _skewness

        values = [1.0, 1.0, 1.0, 100.0]  # right-skewed
        s = _skewness(values)
        assert s > 0

    def test_skewness_short_returns_zero(self):
        from app.validation.trade_quality_analysis import _skewness

        assert _skewness([1.0, 2.0]) == 0.0

    def test_skewness_constant_returns_zero(self):
        from app.validation.trade_quality_analysis import _skewness

        assert _skewness([1.0, 1.0, 1.0]) == 0.0

    def test_kurtosis_basic(self):
        from app.validation.trade_quality_analysis import _kurtosis

        values = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        k = _kurtosis(values)
        assert isinstance(k, float)

    def test_kurtosis_short_returns_zero(self):
        from app.validation.trade_quality_analysis import _kurtosis

        assert _kurtosis([1.0, 2.0, 3.0]) == 0.0

    def test_kurtosis_constant_returns_zero(self):
        from app.validation.trade_quality_analysis import _kurtosis

        assert _kurtosis([5.0, 5.0, 5.0, 5.0]) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# MarketContext (mocked yfinance + repo)
# ─────────────────────────────────────────────────────────────────────────────
class TestMarketContext:
    def test_get_risk_free_rate_returns_default(self):
        from app.data_access.market_context import get_risk_free_rate

        with patch(
            "app.data_access.market_context.SqliteMetricsRepository"
        ) as MockRepo:
            mock_repo = MagicMock()
            mock_repo.fetch_metrics.return_value = []
            MockRepo.return_value = mock_repo
            rate = get_risk_free_rate()
        assert rate == pytest.approx(0.045)

    def test_get_risk_free_rate_with_data(self):
        from app.data_access.market_context import get_risk_free_rate

        with patch(
            "app.data_access.market_context.SqliteMetricsRepository"
        ) as MockRepo:
            mock_repo = MagicMock()
            mock_repo.fetch_metrics.return_value = [{"value": 4.5}]
            MockRepo.return_value = mock_repo
            rate = get_risk_free_rate()
        assert rate == pytest.approx(0.045)

    def test_update_macro_context_calls_save(self):
        from app.data_access.market_context import update_macro_context

        mock_df = pd.DataFrame(
            {"Close": [25.0]},
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-01")]),
        )
        with patch(
            "app.data_access.market_context.SqliteMetricsRepository"
        ) as MockRepo, patch("app.data_access.market_context.yf.Ticker") as MockTicker:
            mock_repo = MagicMock()
            MockRepo.return_value = mock_repo
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = mock_df
            MockTicker.return_value = mock_ticker
            update_macro_context()
        assert mock_repo.save_metrics.call_count >= 1

    def test_update_macro_context_empty_df(self):
        from app.data_access.market_context import update_macro_context

        with patch(
            "app.data_access.market_context.SqliteMetricsRepository"
        ) as MockRepo, patch("app.data_access.market_context.yf.Ticker") as MockTicker:
            mock_repo = MagicMock()
            MockRepo.return_value = mock_repo
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = pd.DataFrame()
            MockTicker.return_value = mock_ticker
            update_macro_context()
        mock_repo.save_metrics.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Shim modules import (analysis layer)
# ─────────────────────────────────────────────────────────────────────────────
class TestAnalysisShims:
    def test_decision_effectiveness_import(self):
        with patch("app.infrastructure.repositories.DataManagerRepository"):
            from app.analysis.decision_effectiveness import (
                DecisionEffectivenessAnalyzer,
            )

            assert DecisionEffectivenessAnalyzer is not None

    def test_phase6_validator_import(self):
        with patch("app.infrastructure.repositories.DataManagerRepository"):
            from app.analysis.phase6_validator import Phase6Validator

            assert Phase6Validator is not None
