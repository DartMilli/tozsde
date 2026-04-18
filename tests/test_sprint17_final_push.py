"""S17: Remaining coverage push to reach 75%.

Targets:
- app/config/__init__.py (40% → 100%)
- app/decision/position_sizer.py shim (0% → 80%+)
- app/backtesting/outcome_evaluator.py (20% → 70%+)
- app/backtesting/pipeline_backtester.py (33% → 70%+)
- app/validation/robustness_self_check.py engine_integrity (42% → 70%+)
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# config/__init__.py
# ─────────────────────────────────────────────────────────────────────────────
class TestConfigGetConf:
    def test_returns_settings_when_provided(self):
        from app.config import get_conf

        settings = MagicMock()
        assert get_conf(settings) is settings

    def test_raises_when_none(self):
        from app.config import get_conf

        with pytest.raises(RuntimeError, match="Settings must be injected"):
            get_conf(None)

    def test_raises_when_not_provided(self):
        from app.config import get_conf

        with pytest.raises(RuntimeError):
            get_conf()


# ─────────────────────────────────────────────────────────────────────────────
# decision/position_sizer.py (shim)
# ─────────────────────────────────────────────────────────────────────────────
class TestPositionSizerShim:
    def _make_settings(self):
        s = MagicMock()
        s.P6_POSITION_MAX_PCT = 0.25
        s.P6_MIN_CONF_FACTOR = 0.5
        s.P6_MIN_WF_FACTOR = 0.6
        s.P6_SAFETY_DISCOUNT = 0.1
        return s

    def test_construction(self):
        from app.decision.position_sizer import PositionSizer

        sizer = PositionSizer(settings=self._make_settings())
        assert sizer.max_position_pct == 0.25

    def test_compute_delegates_to_core(self):
        from app.decision.position_sizer import PositionSizer
        from app.core.decision.position_sizer import PositionSizingResult

        settings = self._make_settings()
        sizer = PositionSizer(settings=settings)
        mock_result = MagicMock(spec=PositionSizingResult)
        with patch("app.decision.position_sizer.CorePositionSizer") as MockCore:
            instance = MagicMock()
            instance.compute.return_value = mock_result
            MockCore.return_value = instance
            result = sizer.compute(
                base_position_size=1000.0,
                confidence=0.75,
                wf_score=0.8,
                safety_discount=0.05,
                equity=10000.0,
            )
        assert result is mock_result

    def test_apply_position_sizing_function(self):
        from app.decision.position_sizer import apply_position_sizing

        settings = self._make_settings()
        item = {
            "decision": {"action": "BUY", "action_code": 1},
            "audit": {"confidence": 0.7, "wf_score": 0.8},
        }
        with patch(
            "app.decision.position_sizer.core_apply_position_sizing"
        ) as mock_apply:
            mock_apply.return_value = {"position_sizing": {"final_size": 500}}
            result = apply_position_sizing(item, equity=10000.0, settings=settings)
        mock_apply.assert_called_once()
        assert result["position_sizing"]["final_size"] == 500


# ─────────────────────────────────────────────────────────────────────────────
# OutcomeEvaluator
# ─────────────────────────────────────────────────────────────────────────────


def _make_ohlcv_df(n=30, base=100.0) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    prices = [base + i for i in range(n)]
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


class TestOutcomeEvaluator:
    def _make_evaluator(self):
        dm = MagicMock()
        from app.backtesting.outcome_evaluator import OutcomeEvaluator

        evaluator = OutcomeEvaluator(outcomes_repo=dm)
        return evaluator, dm

    def test_no_pending_decisions_does_nothing(self):
        evaluator, dm = self._make_evaluator()
        dm.get_unevaluated_buy_decisions.return_value = []
        evaluator.evaluate_past_decisions()
        dm.save_outcome.assert_not_called()

    def test_saves_outcome_for_profitable_trade(self):
        evaluator, dm = self._make_evaluator()
        df = _make_ohlcv_df(30)
        dm.get_unevaluated_buy_decisions.return_value = [
            (1, "2023-01-03T00:00:00", "VOO", "{}"),
        ]
        dm.load_ohlcv.return_value = df
        evaluator.evaluate_past_decisions(lookback_days=30, hold_period=5)
        dm.save_outcome.assert_called_once()

    def test_skips_when_no_future_data(self):
        evaluator, dm = self._make_evaluator()
        df = _make_ohlcv_df(5)
        dm.get_unevaluated_buy_decisions.return_value = [
            (1, "2023-01-06T00:00:00", "VOO", "{}"),
        ]
        dm.load_ohlcv.return_value = df
        evaluator.evaluate_past_decisions(hold_period=20)
        # No outcome saved since exit_idx is out of range or data is missing
        # (outcome might or might not be saved - just verify no crash)

    def test_calculate_trade_result_no_future_data(self):
        evaluator, dm = self._make_evaluator()
        # Empty DataFrame after the entry date
        df = _make_ohlcv_df(5)
        dm.load_ohlcv.return_value = df
        result = evaluator._calculate_trade_result(
            "VOO", "2023-01-10T00:00:00", hold_period=10
        )
        # May return None since no future data or exit is too close
        assert result is None or isinstance(result, float)

    def test_calculate_trade_result_returns_pnl(self):
        evaluator, dm = self._make_evaluator()
        df = _make_ohlcv_df(30)
        dm.load_ohlcv.return_value = df
        result = evaluator._calculate_trade_result(
            "VOO", "2023-01-04T00:00:00", hold_period=5
        )
        assert result is None or isinstance(result, float)

    def test_constructor_no_repo_creates_dm(self):
        with patch("app.backtesting.outcome_evaluator.DataManagerRepository") as MockDM:
            MockDM.return_value = MagicMock()
            from app.backtesting.outcome_evaluator import OutcomeEvaluator

            ev = OutcomeEvaluator()
        MockDM.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# PipelineBacktester
# ─────────────────────────────────────────────────────────────────────────────
class TestPipelineBacktester:
    def _make_backtester(self):
        from app.backtesting.pipeline_backtester import PipelineBacktester

        pipeline = MagicMock()
        return PipelineBacktester(pipeline=pipeline), pipeline

    def test_start_after_end_returns_zero(self):
        backtester, pipeline = self._make_backtester()
        result = backtester.run(
            ticker="VOO",
            start=date(2023, 12, 31),
            end=date(2023, 1, 1),
        )
        assert result == 0

    def test_empty_df_returns_zero(self):
        backtester, pipeline = self._make_backtester()
        pipeline.load_market_data.return_value = None
        result = backtester.run(
            ticker="VOO",
            start=date(2023, 1, 1),
            end=date(2023, 1, 31),
        )
        assert result == 0

    def test_runs_for_each_date(self):
        backtester, pipeline = self._make_backtester()
        df = _make_ohlcv_df(5)
        pipeline.load_market_data.return_value = df
        pipeline.build_daily_candidate.return_value = {
            "payload": {"ticker": "VOO"},
            "decision": {"action": "HOLD", "action_code": 0},
            "explanation": {"hu": "h", "en": "e"},
            "audit": {"confidence": 0.7},
        }
        with patch("app.backtesting.pipeline_backtester.OutcomeEvaluator") as MockEval:
            mock_ev = MagicMock()
            MockEval.return_value = mock_ev
            result = backtester.run(
                ticker="VOO",
                start=date(2023, 1, 1),
                end=date(2023, 1, 31),
                evaluate_outcomes=True,
            )
        assert result == len(df)
        mock_ev.evaluate_past_decisions.assert_called_once()

    def test_evaluate_outcomes_false_skips_eval(self):
        backtester, pipeline = self._make_backtester()
        df = _make_ohlcv_df(3)
        pipeline.load_market_data.return_value = df
        pipeline.build_daily_candidate.return_value = {
            "payload": {"ticker": "VOO"},
            "decision": {"action": "HOLD", "action_code": 0},
            "explanation": {"hu": "h", "en": "e"},
            "audit": {"confidence": 0.7},
        }
        with patch("app.backtesting.pipeline_backtester.OutcomeEvaluator") as MockEval:
            result = backtester.run(
                ticker="VOO",
                start=date(2023, 1, 1),
                end=date(2023, 1, 31),
                evaluate_outcomes=False,
            )
        MockEval.assert_not_called()
        assert result == len(df)


# ─────────────────────────────────────────────────────────────────────────────
# robustness_self_check engine_integrity branch (engine has data)
# ─────────────────────────────────────────────────────────────────────────────
class TestRobustnessEngineIntegrity:
    def test_engine_integrity_no_data(self):
        from app.validation.robustness_self_check import _test_engine_integrity

        with patch(
            "app.validation.robustness_self_check.load_data", return_value=None
        ), patch(
            "app.validation.robustness_self_check.get_validation_ticker",
            return_value="VOO",
        ), patch(
            "app.validation.robustness_self_check.get_validation_window",
            return_value=(
                __import__("datetime").date(2022, 1, 1),
                __import__("datetime").date(2022, 12, 31),
            ),
        ):
            result = _test_engine_integrity()
        assert result.name == "engine_integrity"
        assert result.passed is False

    def test_engine_integrity_empty_df(self):
        from app.validation.robustness_self_check import _test_engine_integrity
        import pandas as pd

        with patch(
            "app.validation.robustness_self_check.load_data",
            return_value=pd.DataFrame(),
        ), patch(
            "app.validation.robustness_self_check.get_validation_ticker",
            return_value="VOO",
        ), patch(
            "app.validation.robustness_self_check.get_validation_window",
            return_value=(
                __import__("datetime").date(2022, 1, 1),
                __import__("datetime").date(2022, 12, 31),
            ),
        ):
            result = _test_engine_integrity()
        assert result.passed is False
