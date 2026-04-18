"""S17D part-2: PaperExecutionEngine coverage boost.

Targets:
- paper_execution.py (was 18%)
"""
from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest


def _make_dm(state=None, ohlcv_df=None):
    dm = MagicMock()
    dm.fetch_latest_portfolio_state.return_value = state
    dm.load_ohlcv.return_value = ohlcv_df
    dm.save_outcome.return_value = None
    dm.save_portfolio_state.return_value = None
    return dm


def _make_logger():
    import logging
    return MagicMock(spec=logging.Logger)


def _make_engine(state=None, ohlcv_df=None, settings=None):
    from app.services.paper_execution import PaperExecutionEngine
    dm = _make_dm(state=state, ohlcv_df=ohlcv_df)
    logger = _make_logger()
    return PaperExecutionEngine(dm=dm, logger=logger, settings=settings), dm, logger


def _decision(ticker, action_code, allocation_amount=0.0, latest_price=100.0, decision_id=None):
    return {
        "ticker": ticker,
        "decision": {"action_code": action_code},
        "payload": {"latest_price": latest_price, "timestamp": "2024-01-01T00:00:00"},
        "allocation_amount": allocation_amount,
        "decision_id": decision_id,
    }


class TestPaperPositionDataclass:
    def test_dataclass_fields(self):
        from app.services.paper_execution import PaperPosition
        pos = PaperPosition(
            ticker="AAPL",
            qty=5.0,
            entry_price=100.0,
            entry_timestamp="2024-01-01T00:00:00",
            decision_id=1,
        )
        assert pos.ticker == "AAPL"
        assert pos.qty == 5.0


class TestDecisionsByTicker:
    def test_returns_execution_price_if_present(self):
        from app.services.paper_execution import decisions_by_ticker
        decisions = [
            {"ticker": "AAPL", "payload": {"execution_price": 150.0, "latest_price": 100.0}}
        ]
        result = decisions_by_ticker(decisions)
        assert result == {"AAPL": 150.0}

    def test_falls_back_to_latest_price(self):
        from app.services.paper_execution import decisions_by_ticker
        decisions = [
            {"ticker": "GOOG", "payload": {"latest_price": 200.0}}
        ]
        result = decisions_by_ticker(decisions)
        assert result == {"GOOG": 200.0}

    def test_empty_decisions(self):
        from app.services.paper_execution import decisions_by_ticker
        assert decisions_by_ticker([]) == {}

    def test_missing_price_excluded(self):
        from app.services.paper_execution import decisions_by_ticker
        decisions = [{"ticker": "X", "payload": {}}]
        result = decisions_by_ticker(decisions)
        assert result == {}


class TestLoadLatestState:
    def test_returns_defaults_on_empty_db(self, test_settings):
        engine, dm, _ = _make_engine(state=None, settings=test_settings)
        state = engine._load_latest_state()
        assert state["cash"] == test_settings.INITIAL_CAPITAL
        assert state["positions"] == {}

    def test_loads_positions_from_db(self, test_settings):
        existing_state = {
            "cash": 5000.0,
            "positions": {
                "AAPL": {
                    "ticker": "AAPL",
                    "qty": 10.0,
                    "entry_price": 150.0,
                    "entry_timestamp": "2024-01-01T00:00:00",
                    "decision_id": 7,
                }
            },
        }
        engine, dm, _ = _make_engine(state=existing_state, settings=test_settings)
        state = engine._load_latest_state()
        assert state["cash"] == 5000.0
        from app.services.paper_execution import PaperPosition
        assert isinstance(state["positions"]["AAPL"], PaperPosition)


class TestResolveExecutionPrice:
    def test_close_to_close_returns_latest_price(self, test_settings):
        engine, _, _ = _make_engine(settings=test_settings)
        payload = {"latest_price": 99.0}
        price, exec_date = engine._resolve_execution_price(
            ticker="AAPL",
            payload=payload,
            as_of=date(2024, 1, 5),
            execution_policy="close_to_close",
        )
        assert price == 99.0
        assert exec_date == date(2024, 1, 5)

    def test_next_open_no_data_returns_none(self, test_settings):
        engine, _, _ = _make_engine(settings=test_settings)
        # dm.load_ohlcv returns None
        price, exec_date = engine._resolve_execution_price(
            ticker="AAPL",
            payload={},
            as_of=date(2024, 1, 5),
            execution_policy="next_open",
        )
        assert price is None

    def test_next_open_with_future_data(self, test_settings):
        import pandas as pd
        as_of = date(2024, 1, 5)
        idx = pd.DatetimeIndex(["2024-01-08"])
        df = pd.DataFrame({"open": [155.0], "close": [156.0]}, index=idx)
        engine, dm, _ = _make_engine(ohlcv_df=df, settings=test_settings)
        price, exec_date = engine._resolve_execution_price(
            ticker="AAPL",
            payload={},
            as_of=as_of,
            execution_policy="next_open",
        )
        assert price == 155.0
        assert exec_date == date(2024, 1, 8)


class TestExecute:
    def test_execute_buy_insufficient_cash(self, test_settings):
        engine, dm, logger = _make_engine(state=None, settings=test_settings)
        # allocation_amount > INITIAL_CAPITAL
        decision = _decision("AAPL", action_code=1, allocation_amount=99999.0, latest_price=100.0, decision_id=1)
        with patch.object(engine, "_resolve_execution_price", return_value=(100.0, date(2024, 1, 8))):
            engine.execute([decision], as_of=date(2024, 1, 5))
        logger.warning.assert_called()

    def test_execute_buy_success(self, test_settings):
        engine, dm, logger = _make_engine(state=None, settings=test_settings)
        decision = _decision("AAPL", action_code=1, allocation_amount=500.0, latest_price=100.0, decision_id=2)
        with patch.object(engine, "_resolve_execution_price", return_value=(100.0, date(2024, 1, 8))):
            engine.execute([decision], as_of=date(2024, 1, 5))
        dm.save_portfolio_state.assert_called_once()
        args = dm.save_portfolio_state.call_args
        positions_json = json.loads(args.kwargs.get("positions_json") or args[1]["positions_json"])
        assert "AAPL" in positions_json

    def test_execute_sell_updates_cash(self, test_settings):
        from app.services.paper_execution import PaperPosition
        pos = PaperPosition(
            ticker="AAPL", qty=5.0, entry_price=100.0,
            entry_timestamp="2024-01-01T00:00:00", decision_id=3
        )
        state = {"cash": 500.0, "positions": {"AAPL": pos}, "equity": 1000.0}
        engine, dm, logger = _make_engine(state=state, settings=test_settings)
        decision = _decision("AAPL", action_code=2, latest_price=120.0, decision_id=3)
        with patch.object(engine, "_resolve_execution_price", return_value=(120.0, date(2024, 1, 8))):
            engine.execute([decision], as_of=date(2024, 1, 5))
        dm.save_outcome.assert_called_once()
        # Position should be removed
        call_kwargs = dm.save_portfolio_state.call_args
        positions_json = json.loads(
            call_kwargs.kwargs.get("positions_json") or call_kwargs[1]["positions_json"]
        )
        assert "AAPL" not in positions_json

    def test_execute_skip_zero_allocation(self, test_settings):
        engine, dm, logger = _make_engine(state=None, settings=test_settings)
        decision = _decision("AAPL", action_code=1, allocation_amount=0.0, latest_price=100.0)
        with patch.object(engine, "_resolve_execution_price", return_value=(100.0, date(2024, 1, 8))):
            engine.execute([decision], as_of=date(2024, 1, 5))
        dm.save_portfolio_state.assert_called_once()
        call_kwargs = dm.save_portfolio_state.call_args
        positions_json = json.loads(
            call_kwargs.kwargs.get("positions_json") or call_kwargs[1]["positions_json"]
        )
        assert "AAPL" not in positions_json

    def test_execute_no_price_skipped(self, test_settings):
        engine, dm, _ = _make_engine(state=None, settings=test_settings)
        decision = _decision("AAPL", action_code=1, allocation_amount=500.0)
        with patch.object(engine, "_resolve_execution_price", return_value=(None, date(2024, 1, 5))):
            engine.execute([decision], as_of=date(2024, 1, 5))
        # save_portfolio_state still called (empty positions)
        dm.save_portfolio_state.assert_called_once()

    def test_execute_sell_missing_decision_id(self, test_settings):
        from app.services.paper_execution import PaperPosition
        pos = PaperPosition(
            ticker="TSLA", qty=2.0, entry_price=200.0,
            entry_timestamp="2024-01-01T00:00:00", decision_id=None
        )
        state = {"cash": 1000.0, "positions": {"TSLA": pos}, "equity": 1400.0}
        engine, dm, logger = _make_engine(state=state, settings=test_settings)
        decision = _decision("TSLA", action_code=2, latest_price=210.0, decision_id=None)
        with patch.object(engine, "_resolve_execution_price", return_value=(210.0, date(2024, 1, 8))):
            engine.execute([decision], as_of=date(2024, 1, 5))
        # save_outcome NOT called — decision_id is None
        dm.save_outcome.assert_not_called()
        logger.warning.assert_called()

    def test_drawdown_circuit_breaker_blocks_buy(self, test_settings):
        """When drawdown >= DRAWDOWN_HALT_PCT, BUY orders are suppressed."""
        engine, dm, logger = _make_engine(state=None, settings=test_settings)
        # Simulate large drawdown: equity = 0 → drawdown = 100%
        state = {"cash": 0.0, "positions": {}, "equity": 0.0}
        with patch.object(engine, "_load_latest_state", return_value=state):
            with patch.object(engine, "_resolve_execution_price", return_value=(100.0, date(2024, 1, 8))):
                decision = _decision("AAPL", action_code=1, allocation_amount=500.0)
                engine.execute([decision], as_of=date(2024, 1, 5))
        logger.warning.assert_called()

    def test_hold_decision_code_ignored(self, test_settings):
        engine, dm, _ = _make_engine(state=None, settings=test_settings)
        decision = _decision("AAPL", action_code=0, latest_price=100.0)
        with patch.object(engine, "_resolve_execution_price", return_value=(100.0, date(2024, 1, 8))):
            engine.execute([decision], as_of=date(2024, 1, 5))
        # Position not created (action_code=0 is HOLD, not BUY)
        call_kwargs = dm.save_portfolio_state.call_args
        positions_json = json.loads(
            call_kwargs.kwargs.get("positions_json") or call_kwargs[1]["positions_json"]
        )
        assert "AAPL" not in positions_json
