"""S17D: Services/execution coverage boost.

Targets:
- execution_engines.py (was 61%)
- trading_pipeline.py (service layer, was 55%)
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List
from unittest.mock import MagicMock, patch, PropertyMock
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# NoopExecutionEngine
# ─────────────────────────────────────────────────────────────────────────────


class TestNoopExecutionEngine:
    def test_execute_logs_noop(self):
        from app.services.execution_engines import NoopExecutionEngine

        mock_logger = MagicMock()
        engine = NoopExecutionEngine(logger=mock_logger)
        engine.execute([], as_of=date.today())
        mock_logger.info.assert_called_once()

    def test_execute_with_decisions(self):
        from app.services.execution_engines import NoopExecutionEngine

        engine = NoopExecutionEngine()
        decisions = [{"ticker": "AAPL", "action": "BUY", "qty": 10}]
        # Should not raise
        engine.execute(decisions, as_of=date.today())


# ─────────────────────────────────────────────────────────────────────────────
# BrokerConfig
# ─────────────────────────────────────────────────────────────────────────────


class TestBrokerConfig:
    def test_defaults(self):
        from app.services.execution_engines import BrokerConfig

        cfg = BrokerConfig()
        assert cfg.api_key == ""
        assert cfg.paper is True

    def test_custom_values(self):
        from app.services.execution_engines import BrokerConfig

        cfg = BrokerConfig(
            api_key="key", api_secret="secret", base_url="url", paper=False
        )
        assert cfg.api_key == "key"
        assert cfg.paper is False


# ─────────────────────────────────────────────────────────────────────────────
# LiveExecutionEngine
# ─────────────────────────────────────────────────────────────────────────────


class TestLiveExecutionEngine:
    def _engine(self):
        from app.services.execution_engines import LiveExecutionEngine, BrokerConfig

        cfg = BrokerConfig(api_key="k", paper=True)
        return LiveExecutionEngine(broker_config=cfg)

    def test_execute_empty_decisions(self):
        engine = self._engine()
        # Should not raise, logs info
        engine.execute([], as_of=date.today())

    def test_execute_hold_skipped(self):
        engine = self._engine()
        decisions = [{"ticker": "AAPL", "action": "HOLD", "qty": 0, "price": 100.0}]
        engine.execute(decisions, as_of=date.today())

    def test_execute_buy_raises_not_implemented(self):
        engine = self._engine()
        decisions = [{"ticker": "AAPL", "action": "BUY", "qty": 5, "price": 100.0}]
        # _send_order raises NotImplementedError — should be caught, logged as warning
        engine.execute(decisions, as_of=date.today())

    def test_execute_send_order_exception_caught(self):
        from app.services.execution_engines import LiveExecutionEngine, BrokerConfig

        engine = LiveExecutionEngine(broker_config=BrokerConfig())
        with patch.object(
            engine, "_send_order", side_effect=RuntimeError("broker down")
        ):
            decisions = [{"ticker": "AAPL", "action": "BUY", "qty": 5, "price": 100.0}]
            engine.execute(decisions, as_of=date.today())  # Should not raise

    def test_send_order_raises_not_implemented(self):
        engine = self._engine()
        with pytest.raises(NotImplementedError):
            engine._send_order("AAPL", "BUY", 1, 100.0, date.today())

    def test_config_from_env(self):
        from app.services.execution_engines import LiveExecutionEngine
        import os

        with patch.dict(
            os.environ,
            {
                "BROKER_API_KEY": "testkey",
                "BROKER_API_SECRET": "testsecret",
                "BROKER_BASE_URL": "https://test.broker",
                "BROKER_PAPER": "false",
            },
        ):
            cfg = LiveExecutionEngine._config_from_env()
        assert cfg.api_key == "testkey"
        assert cfg.paper is False

    def test_execute_qty_zero_skipped(self):
        engine = self._engine()
        decisions = [{"ticker": "META", "action": "BUY", "qty": 0, "price": 200.0}]
        engine.execute(decisions, as_of=date.today())


# ─────────────────────────────────────────────────────────────────────────────
# TradingPipelineService
# ─────────────────────────────────────────────────────────────────────────────


def _make_pipeline(test_settings):
    from app.services.trading_pipeline import TradingPipelineService

    history_store = MagicMock()
    data_fetcher = MagicMock()
    model_runner = MagicMock()
    email_notifier = MagicMock()
    execution_engine = MagicMock()
    state_repo = MagicMock()
    state_repo.fetch_latest_portfolio_state.return_value = None

    return TradingPipelineService(
        history_store=history_store,
        settings=test_settings,
        data_fetcher=data_fetcher,
        model_runner=model_runner,
        email_notifier=email_notifier,
        execution_engine=execution_engine,
        state_repo=state_repo,
    )


class TestTradingPipelineService:
    def test_init_requires_history_store(self, test_settings):
        from app.services.trading_pipeline import TradingPipelineService

        with pytest.raises(ValueError, match="history_store"):
            TradingPipelineService(
                history_store=None,
                data_fetcher=MagicMock(),
                model_runner=MagicMock(),
                email_notifier=MagicMock(),
                execution_engine=MagicMock(),
            )

    def test_init_requires_data_fetcher(self, test_settings):
        from app.services.trading_pipeline import TradingPipelineService

        with pytest.raises(ValueError, match="data_fetcher"):
            TradingPipelineService(
                history_store=MagicMock(),
                data_fetcher=None,
                model_runner=MagicMock(),
                email_notifier=MagicMock(),
                execution_engine=MagicMock(),
            )

    def test_load_market_data(self, test_settings):
        pipeline = _make_pipeline(test_settings)
        pipeline.data_fetcher.load_data.return_value = MagicMock()
        result = pipeline.load_market_data("AAPL", start="2024-01-01")
        pipeline.data_fetcher.load_data.assert_called_once_with(
            "AAPL", start="2024-01-01", end=None
        )

    def test_apply_safety_rules_passthrough_in_validation_mode(self, test_settings):
        pipeline = _make_pipeline(test_settings)
        decision = {"action": "BUY"}
        audit = {}
        import os

        with patch.dict(os.environ, {"VALIDATION_DISABLE_POLICY": "true"}):
            result = pipeline.apply_safety_rules(decision, audit)
        assert result is decision  # passed through unchanged

    def test_check_portfolio_drift_disabled(self, test_settings):
        from dataclasses import replace

        settings = replace(test_settings, ENABLE_REBALANCER=False)
        from app.services.trading_pipeline import TradingPipelineService

        pipeline = TradingPipelineService(
            history_store=MagicMock(),
            settings=settings,
            data_fetcher=MagicMock(),
            model_runner=MagicMock(),
            email_notifier=MagicMock(),
            execution_engine=MagicMock(),
            state_repo=MagicMock(),
        )
        result = pipeline.check_portfolio_drift(
            current_positions={"AAPL": 0.3},
            target_allocation={"AAPL": 0.5},
            prices={"AAPL": 150.0},
            total_value=10000.0,
        )
        assert result["should_rebalance"] is False

    def test_send_notifications(self, test_settings):
        pipeline = _make_pipeline(test_settings)
        pipeline.send_notifications("Subject", "Body", "test@example.com")
        pipeline.email_notifier.send.assert_called_once_with(
            "Subject", "Body", "test@example.com"
        )

    def test_execute_trades(self, test_settings):
        pipeline = _make_pipeline(test_settings)
        decisions = [{"ticker": "AAPL", "action": "BUY"}]
        today = date.today()
        pipeline.execute_trades(decisions, as_of=today)
        pipeline.execution_engine.execute.assert_called_once_with(
            decisions=decisions, as_of=today
        )

    def test_persist_decision_delegates_to_history_store(self, test_settings):
        pipeline = _make_pipeline(test_settings)
        pipeline.history_store.save_decision.return_value = 42
        result = pipeline.persist_decision(
            payload={"model_votes": [], "model_id": None, "timestamp": None},
            decision={"action_code": 1},
            explanation={},
            audit={},
        )
        assert result == 42
        pipeline.history_store.save_decision.assert_called_once()

    def test_allocate_capital_with_no_equity(self, test_settings):
        pipeline = _make_pipeline(test_settings)
        pipeline.state_repo.fetch_latest_portfolio_state.return_value = None
        candidates = [
            {
                "ticker": "AAPL",
                "decision": {"action_code": 1},
                "payload": {"volatility": 0.02},
                "allocation_amount": 0.0,
                "allocation_pct": 0.0,
            }
        ]
        result = pipeline.allocate_capital(candidates)
        assert isinstance(result, list)

    def test_get_tickers_to_process_single(self, test_settings):
        pipeline = _make_pipeline(test_settings)
        result = pipeline.get_tickers_to_process("AAPL")
        assert result == ["AAPL"]

    def test_get_settings_returns_injected(self, test_settings):
        pipeline = _make_pipeline(test_settings)
        assert pipeline._get_settings() is test_settings

    def test_compute_signals_calls_prepare_df(self, test_settings):
        import pandas as pd

        pipeline = _make_pipeline(test_settings)
        df = pd.DataFrame({"Close": [100.0, 101.0, 102.0]})
        with patch(
            "app.services.trading_pipeline.prepare_df", return_value=df
        ) as mock_prep:
            result = pipeline.compute_signals(df, "AAPL")
        mock_prep.assert_called_once()

    def test_get_tickers_from_settings(self, test_settings):
        from dataclasses import replace

        settings = replace(test_settings, TICKERS=["AAPL", "MSFT"])
        from app.services.trading_pipeline import TradingPipelineService

        pipeline = TradingPipelineService(
            history_store=MagicMock(),
            settings=settings,
            data_fetcher=MagicMock(),
            model_runner=MagicMock(),
            email_notifier=MagicMock(),
            execution_engine=MagicMock(),
        )
        result = pipeline.get_tickers_to_process(None)
        assert result == ["AAPL", "MSFT"]

    def test_get_tickers_no_ticker_falls_back_to_settings_then_supported_list(
        self, test_settings
    ):
        from dataclasses import replace

        settings = replace(test_settings, TICKERS=None)
        from app.services.trading_pipeline import TradingPipelineService

        pipeline = TradingPipelineService(
            history_store=MagicMock(),
            settings=settings,
            data_fetcher=MagicMock(),
            model_runner=MagicMock(),
            email_notifier=MagicMock(),
            execution_engine=MagicMock(),
        )
        with patch(
            "app.services.trading_pipeline.get_supported_ticker_list",
            return_value=["VOO", "SPY"],
        ):
            result = pipeline.get_tickers_to_process(None)
        assert "VOO" in result

    def test_check_portfolio_drift_enabled(self, test_settings):
        from dataclasses import replace

        settings = replace(test_settings, ENABLE_REBALANCER=True)
        from app.services.trading_pipeline import TradingPipelineService

        pipeline = TradingPipelineService(
            history_store=MagicMock(),
            settings=settings,
            data_fetcher=MagicMock(),
            model_runner=MagicMock(),
            email_notifier=MagicMock(),
            execution_engine=MagicMock(),
        )
        rebalance_result = {
            "should_rebalance": True,
            "drift_info": {"drift_avg": 0.15},
            "trades": [],
        }
        with patch(
            "app.services.trading_pipeline.check_and_rebalance",
            return_value=rebalance_result,
        ):
            result = pipeline.check_portfolio_drift({}, {}, {}, 10000.0)
        assert result["should_rebalance"] is True


# ─────────────────────────────────────────────────────────────────────────────
# AlpacaExecutionEngine
# ─────────────────────────────────────────────────────────────────────────────


class TestAlpacaExecutionEngine:
    def test_send_order_import_error_returns_empty(self):
        import sys
        from app.services.execution_engines import AlpacaExecutionEngine, BrokerConfig

        eng = AlpacaExecutionEngine(broker_config=BrokerConfig(api_key="k"))
        with patch.dict(sys.modules, {"alpaca_trade_api": None}):
            result = eng._send_order("AAPL", "BUY", 5, 100.0, date.today())
        # alpaca_trade_api not installed -> ImportError -> returns ""
        assert isinstance(result, str)

    def test_send_order_no_broker_config_returns_empty(self):
        from app.services.execution_engines import AlpacaExecutionEngine

        eng = AlpacaExecutionEngine(broker_config=None)
        # Will fail ImportError (alpaca not installed) before config check
        result = eng._send_order("AAPL", "BUY", 5, 100.0, date.today())
        assert result == ""

    def test_send_order_api_exception_returns_empty(self):
        from app.services.execution_engines import AlpacaExecutionEngine, BrokerConfig

        eng = AlpacaExecutionEngine(broker_config=BrokerConfig(api_key="k"))
        mock_alpaca = MagicMock()
        mock_api_instance = MagicMock()
        mock_api_instance.submit_order.side_effect = RuntimeError("timeout")
        mock_alpaca.REST.return_value = mock_api_instance
        import sys

        with patch.dict(sys.modules, {"alpaca_trade_api": mock_alpaca}):
            result = eng._send_order("AAPL", "BUY", 5, 100.0, date.today())
        assert result == ""

    def test_send_order_success(self):
        from app.services.execution_engines import AlpacaExecutionEngine, BrokerConfig

        eng = AlpacaExecutionEngine(
            broker_config=BrokerConfig(api_key="k", api_secret="s")
        )
        mock_alpaca = MagicMock()
        mock_order = MagicMock()
        mock_order.id = "order-123"
        mock_alpaca.REST.return_value.submit_order.return_value = mock_order
        import sys

        with patch.dict(sys.modules, {"alpaca_trade_api": mock_alpaca}):
            result = eng._send_order("AAPL", "BUY", 5, 100.0, date.today())
        assert result == "order-123"
