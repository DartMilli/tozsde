from dataclasses import replace
from datetime import date
from unittest.mock import MagicMock

import pandas as pd

from app.application.use_cases.execution_coordinator import ExecutionCoordinator
from app.core.decision.adaptive_strategy import StrategySelection
from app.core.decision.recommender import generate_daily_recommendation_payload
from app.services.paper_execution import PaperExecutionEngine


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [99.5, 100.5, 101.5, 102.5],
            "High": [100.5, 101.5, 102.5, 103.5],
            "Low": [99.0, 100.0, 101.0, 102.0],
            "Close": [100.0, 101.0, 102.0, 103.0],
            "Volume": [1000000, 1000000, 1000000, 1000000],
            "ADX": [30.0, 31.0, 32.0, 33.0],
        },
        index=pd.date_range("2026-04-10", periods=4, freq="D"),
    )


class DummySafetyRuleEngine:
    def __init__(self, *args, **kwargs):
        pass


class DummyDecisionEngine:
    def __init__(self, *args, **kwargs):
        pass

    def run(self, ticker: str, decision: dict) -> dict:
        return decision


def test_execution_coordinator_cost_gate_blocks_low_edge(test_settings):
    pipeline = MagicMock()
    settings = replace(
        test_settings, ENABLE_POSITION_SIZING=False, ENABLE_COST_GATE=True
    )
    pipeline._get_settings.return_value = settings
    pipeline.settings = settings
    pipeline.logger = MagicMock()
    pipeline.allocate_capital.return_value = [
        {
            "ticker": "VOO",
            "decision": {"action_code": 1, "no_trade": False},
            "payload": {
                "timestamp": "2026-04-14",
                "latest_price": 100.0,
                "expectancy": {"expected_net_pnl": 0.001},
            },
            "allocation_amount": 1000.0,
        }
    ]

    coordinator = ExecutionCoordinator(pipeline)
    no_trade, finalized = coordinator.split_and_finalize(
        [{"decision": {"no_trade": False}, "payload": {}}]
    )

    assert no_trade == []
    assert finalized == []


def test_execution_coordinator_cost_gate_keeps_high_edge(test_settings):
    pipeline = MagicMock()
    settings = replace(
        test_settings, ENABLE_POSITION_SIZING=False, ENABLE_COST_GATE=True
    )
    pipeline._get_settings.return_value = settings
    pipeline.settings = settings
    pipeline.logger = MagicMock()
    pipeline.allocate_capital.return_value = [
        {
            "ticker": "VOO",
            "decision": {"action_code": 1, "no_trade": False},
            "payload": {
                "timestamp": "2026-04-14",
                "latest_price": 100.0,
                "expectancy": {"expected_net_pnl": 0.02},
            },
            "allocation_amount": 1000.0,
        }
    ]

    coordinator = ExecutionCoordinator(pipeline)
    _, finalized = coordinator.split_and_finalize(
        [{"decision": {"no_trade": False}, "payload": {}}]
    )

    assert len(finalized) == 1
    assert finalized[0]["payload"]["implementation_shortfall"]["min_edge_required"] > 0


def test_recommender_adaptive_strategy_adds_strategy_selection(
    monkeypatch, test_settings
):
    settings = replace(
        test_settings,
        ENABLE_ADAPTIVE_STRATEGY=True,
        ENABLE_EXPECTANCY_GATE=False,
        ENABLE_ML_ENSEMBLE=False,
    )

    class DummyRunner:
        def run_ensemble(self, **kwargs):
            return (
                [1],
                [0.9],
                [0.8],
                [
                    {
                        "model_path": "model_a",
                        "model_name": "model_a",
                        "rank": 1,
                        "trained_at": date(2026, 4, 14),
                        "action_label": "BUY",
                        "confidence": 0.9,
                        "wf_score": 0.8,
                    }
                ],
                [],
            )

    class DummySelector:
        def __init__(self, settings=None):
            pass

        def explore_or_exploit(self, market_regime=None):
            return StrategySelection(
                selected_strategies={"MOMENTUM": 0.7, "MACD": 0.3},
                selection_mode="EXPLOIT",
                market_context=market_regime or "UNKNOWN",
                confidence_in_selection=0.8,
            )

    monkeypatch.setattr(
        "app.core.decision.adaptive_strategy_selector.AdaptiveStrategySelector",
        DummySelector,
    )

    history_store = type(
        "H",
        (),
        {
            "save_decision": lambda *a, **k: None,
            "load_range": lambda *a, **k: [],
            "load_recent_outcomes": lambda *a, **k: [],
        },
    )()

    payload = generate_daily_recommendation_payload(
        "TEST",
        history_store=history_store,
        model_runner=DummyRunner(),
        as_of_date=date(2026, 4, 14),
        settings=settings,
        load_data_fn=lambda ticker, start, end: _sample_df().copy(),
        prepare_df_fn=lambda frame, ticker: frame,
        safety_rule_engine_cls=DummySafetyRuleEngine,
        decision_engine_cls=DummyDecisionEngine,
    )

    assert payload["decision"]["strategy_name"] == "MOMENTUM"
    assert payload["adaptive_strategy"]["selected_strategy"] == "MOMENTUM"


def test_paper_execution_updates_strategy_bandit_on_sell(monkeypatch, test_settings):
    settings = replace(
        test_settings,
        EXECUTION_POLICY="close_to_close",
        ENABLE_ADAPTIVE_STRATEGY=True,
    )
    dm = MagicMock()
    dm.fetch_latest_portfolio_state.return_value = {
        "cash": 10000.0,
        "equity": 10000.0,
        "positions": {
            "AAPL": {
                "qty": 1.0,
                "entry_price": 100.0,
                "entry_timestamp": "2026-04-10",
                "decision_id": 1,
                "strategy_name": "MOMENTUM",
            }
        },
    }

    selector_instance = MagicMock()

    class DummySelector:
        def __init__(self, settings=None, data_manager=None):
            pass

        def update_strategy(self, strategy_name: str, success: bool):
            selector_instance.update_strategy(strategy_name, success)

    monkeypatch.setattr(
        "app.core.decision.adaptive_strategy_selector.AdaptiveStrategySelector",
        DummySelector,
    )

    engine = PaperExecutionEngine(dm=dm, logger=MagicMock(), settings=settings)
    engine.execute(
        decisions=[
            {
                "ticker": "AAPL",
                "decision": {"action_code": 2, "strategy_name": "IGNORED"},
                "payload": {"latest_price": 110.0, "timestamp": "2026-04-14"},
            }
        ],
        as_of=date(2026, 4, 14),
    )

    selector_instance.update_strategy.assert_called_once_with("MOMENTUM", True)
    dm.save_outcome.assert_called_once()
