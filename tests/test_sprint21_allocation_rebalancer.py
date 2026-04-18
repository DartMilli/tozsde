from dataclasses import replace
from datetime import date
from unittest.mock import MagicMock

import pandas as pd

from app.application.use_cases.daily_pipeline_use_case import DailyPipelineUseCase
from app.backtesting.history_store import HistoryStore
from app.services.trading_pipeline import TradingPipelineService


def _price_frame(close_values: list[float]) -> pd.DataFrame:
    rows = len(close_values)
    return pd.DataFrame(
        {
            "Open": close_values,
            "High": [value + 1 for value in close_values],
            "Low": [value - 1 for value in close_values],
            "Close": close_values,
            "Volume": [1000000] * rows,
        },
        index=pd.date_range("2026-01-01", periods=rows, freq="D"),
    )


def test_trading_pipeline_allocate_capital_risk_parity_uses_current_equity(
    test_settings,
):
    settings = replace(
        test_settings,
        ALLOCATION_MODE="risk_parity",
        ENABLE_CORRELATION_LIMITS=False,
        INITIAL_CAPITAL=10000,
    )

    class Fetcher:
        def load_data(self, ticker, start=None, end=None):
            if ticker == "VOO":
                return _price_frame([100 + i for i in range(70)])
            return _price_frame([100, 103, 97, 104, 96, 105] * 12)

    state_repo = MagicMock()
    state_repo.fetch_latest_portfolio_state.return_value = {"equity": 20000.0}

    pipeline = TradingPipelineService(
        history_store=MagicMock(spec=HistoryStore),
        settings=settings,
        data_fetcher=Fetcher(),
        model_runner=MagicMock(),
        email_notifier=MagicMock(),
        execution_engine=MagicMock(),
        state_repo=state_repo,
    )

    candidates = [
        {
            "ticker": "VOO",
            "decision": {"action_code": 1},
            "payload": {"timestamp": "2026-04-14"},
        },
        {
            "ticker": "QQQ",
            "decision": {"action_code": 1},
            "payload": {"timestamp": "2026-04-14"},
        },
    ]

    allocated = pipeline.allocate_capital(candidates)

    total_allocated = sum(item["allocation_amount"] for item in allocated)
    assert abs(total_allocated - 19000.0) < 5.0
    assert allocated[0]["allocation_amount"] > allocated[1]["allocation_amount"]


def test_daily_pipeline_rebalancer_executes_when_benefit_exceeds_cost(test_settings):
    settings = replace(
        test_settings,
        ENABLE_REBALANCER=True,
        REBALANCE_EXECUTE=True,
        REBALANCE_COST_MULTIPLIER=2.0,
        DRIFT_ANNUAL_IMPACT_FACTOR=0.5,
    )

    pipeline = MagicMock()
    pipeline._get_settings.return_value = settings
    pipeline.state_repo.fetch_latest_portfolio_state.return_value = {
        "equity": 10000.0,
        "positions": {"VOO": {"qty": 10, "entry_price": 100.0}},
    }
    pipeline.check_portfolio_drift.return_value = {
        "should_rebalance": True,
        "trades": [{"ticker": "VOO", "action": "SELL", "qty": 2, "price": 100.0}],
        "estimated_cost": 10.0,
        "drift_info": {"drift_avg": 0.10},
    }
    pipeline.logger = MagicMock()

    use_case = DailyPipelineUseCase(pipeline)
    use_case.execution.execute_finalized = MagicMock()

    use_case._run_rebalancer_check(
        [
            {
                "ticker": "VOO",
                "payload": {"timestamp": "2026-04-14", "latest_price": 100.0},
                "allocation_pct": 0.5,
            }
        ]
    )

    assert use_case.execution.execute_finalized.called
    executed = use_case.execution.execute_finalized.call_args[0][0]
    assert executed[0]["decision"]["action_code"] == 2


def test_daily_pipeline_rebalancer_skips_when_benefit_too_low(test_settings):
    settings = replace(
        test_settings,
        ENABLE_REBALANCER=True,
        REBALANCE_EXECUTE=True,
        REBALANCE_COST_MULTIPLIER=2.0,
        DRIFT_ANNUAL_IMPACT_FACTOR=0.01,
    )

    pipeline = MagicMock()
    pipeline._get_settings.return_value = settings
    pipeline.state_repo.fetch_latest_portfolio_state.return_value = {
        "equity": 10000.0,
        "positions": {"VOO": {"qty": 10, "entry_price": 100.0}},
    }
    pipeline.check_portfolio_drift.return_value = {
        "should_rebalance": True,
        "trades": [{"ticker": "VOO", "action": "SELL", "qty": 2, "price": 100.0}],
        "estimated_cost": 100.0,
        "drift_info": {"drift_avg": 0.10},
    }
    pipeline.logger = MagicMock()

    use_case = DailyPipelineUseCase(pipeline)
    use_case.execution.execute_finalized = MagicMock()

    use_case._run_rebalancer_check(
        [
            {
                "ticker": "VOO",
                "payload": {"timestamp": "2026-04-14", "latest_price": 100.0},
                "allocation_pct": 0.5,
            }
        ]
    )

    assert not use_case.execution.execute_finalized.called
