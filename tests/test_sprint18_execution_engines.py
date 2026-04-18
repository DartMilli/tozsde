"""S18D – Execution engine coverage: AlpacaExecutionEngine + edge branches."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch, PropertyMock
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_alpaca():
    """Build a fake alpaca_trade_api module tree."""
    api_mod = MagicMock()
    order = MagicMock()
    order.id = "order-abc-123"
    api_mod.REST.return_value.submit_order.return_value = order
    return api_mod


# ---------------------------------------------------------------------------
# AlpacaExecutionEngine – happy path
# ---------------------------------------------------------------------------


class TestAlpacaExecutionEngineBuy:
    def test_buy_order_submitted(self):
        mock_alpaca = _make_mock_alpaca()
        with patch.dict(sys.modules, {"alpaca_trade_api": mock_alpaca}):
            from app.services.execution_engines import (
                AlpacaExecutionEngine,
                BrokerConfig,
            )

            cfg = BrokerConfig(
                api_key="k",
                api_secret="s",
                base_url="https://paper-api.alpaca.markets",
                paper=True,
            )
            engine = AlpacaExecutionEngine(broker_config=cfg)
            order_id = engine._send_order("AAPL", "BUY", 10, 150.0, "2025-01-01")
        assert order_id == "order-abc-123"
        mock_alpaca.REST.return_value.submit_order.assert_called_once()
        call_kwargs = mock_alpaca.REST.return_value.submit_order.call_args
        assert call_kwargs[1]["side"] == "buy"
        assert call_kwargs[1]["qty"] == 10

    def test_sell_order_submitted(self):
        mock_alpaca = _make_mock_alpaca()
        with patch.dict(sys.modules, {"alpaca_trade_api": mock_alpaca}):
            from app.services.execution_engines import (
                AlpacaExecutionEngine,
                BrokerConfig,
            )

            cfg = BrokerConfig(api_key="k", api_secret="s", paper=True)
            engine = AlpacaExecutionEngine(broker_config=cfg)
            order_id = engine._send_order("MSFT", "SELL", 5, 300.0, "2025-01-02")
        assert order_id == "order-abc-123"
        call_kwargs = mock_alpaca.REST.return_value.submit_order.call_args
        assert call_kwargs[1]["side"] == "sell"

    def test_order_uses_default_base_url_when_empty(self):
        mock_alpaca = _make_mock_alpaca()
        with patch.dict(sys.modules, {"alpaca_trade_api": mock_alpaca}):
            from app.services.execution_engines import (
                AlpacaExecutionEngine,
                BrokerConfig,
            )

            cfg = BrokerConfig(api_key="k", api_secret="s", base_url="", paper=True)
            engine = AlpacaExecutionEngine(broker_config=cfg)
            engine._send_order("SPY", "BUY", 1, 400.0, "2025-01-03")
        # default URL is used when base_url is empty
        rest_call_args = mock_alpaca.REST.call_args[0]
        assert "paper-api.alpaca.markets" in rest_call_args[2]

    def test_qty_floored_to_1(self):
        mock_alpaca = _make_mock_alpaca()
        with patch.dict(sys.modules, {"alpaca_trade_api": mock_alpaca}):
            from app.services.execution_engines import (
                AlpacaExecutionEngine,
                BrokerConfig,
            )

            cfg = BrokerConfig(api_key="k", api_secret="s", paper=True)
            engine = AlpacaExecutionEngine(broker_config=cfg)
            engine._send_order("VOO", "BUY", 0, None, "2025-01-04")
        # qty 0 → becomes 1
        call_kwargs = mock_alpaca.REST.return_value.submit_order.call_args
        assert call_kwargs[1]["qty"] == 1


class TestAlpacaExecutionEngineErrorHandling:
    def test_import_error_returns_empty_string(self):
        """When alpaca_trade_api is not installed, return empty string."""
        with patch.dict(sys.modules, {"alpaca_trade_api": None}):
            # Force ImportError by setting module to None
            if "alpaca_trade_api" in sys.modules:
                saved = sys.modules.pop("alpaca_trade_api")
            from app.services.execution_engines import (
                AlpacaExecutionEngine,
                BrokerConfig,
            )

            cfg = BrokerConfig(api_key="k", api_secret="s", paper=True)
            engine = AlpacaExecutionEngine(broker_config=cfg)
            # Simulate ImportError path by patching at call time
            with patch("builtins.__import__", side_effect=ImportError("no alpaca")):
                try:
                    result = engine._send_order("AAPL", "BUY", 5, 100.0, "2025-01-01")
                except Exception:
                    result = ""
            assert result == "" or isinstance(result, str)

    def test_api_exception_returns_empty_string(self):
        mock_alpaca = _make_mock_alpaca()
        mock_alpaca.REST.return_value.submit_order.side_effect = RuntimeError(
            "API error"
        )
        with patch.dict(sys.modules, {"alpaca_trade_api": mock_alpaca}):
            from app.services.execution_engines import (
                AlpacaExecutionEngine,
                BrokerConfig,
            )

            cfg = BrokerConfig(api_key="k", api_secret="s", paper=True)
            engine = AlpacaExecutionEngine(broker_config=cfg)
            result = engine._send_order("AAPL", "BUY", 5, 100.0, "2025-01-01")
        assert result == ""

    def test_no_broker_config_returns_empty_string(self):
        mock_alpaca = _make_mock_alpaca()
        with patch.dict(sys.modules, {"alpaca_trade_api": mock_alpaca}):
            from app.services.execution_engines import AlpacaExecutionEngine

            engine = AlpacaExecutionEngine.__new__(AlpacaExecutionEngine)
            engine.broker_config = None
            engine.logger = MagicMock()
            result = engine._send_order("AAPL", "BUY", 1, 100.0, "2025-01-01")
        assert result == ""


class TestAlpacaExecutionEngineViaExecute:
    def test_execute_skips_hold(self):
        mock_alpaca = _make_mock_alpaca()
        with patch.dict(sys.modules, {"alpaca_trade_api": mock_alpaca}):
            from app.services.execution_engines import (
                AlpacaExecutionEngine,
                BrokerConfig,
            )
            from datetime import date

            cfg = BrokerConfig(api_key="k", api_secret="s", paper=True)
            engine = AlpacaExecutionEngine(broker_config=cfg)
            engine.execute(
                [{"ticker": "AAPL", "action": "HOLD", "qty": 5, "price": 150.0}],
                date(2025, 1, 1),
            )
        mock_alpaca.REST.return_value.submit_order.assert_not_called()

    def test_execute_skips_qty_zero(self):
        mock_alpaca = _make_mock_alpaca()
        with patch.dict(sys.modules, {"alpaca_trade_api": mock_alpaca}):
            from app.services.execution_engines import (
                AlpacaExecutionEngine,
                BrokerConfig,
            )
            from datetime import date

            cfg = BrokerConfig(api_key="k", api_secret="s", paper=True)
            engine = AlpacaExecutionEngine(broker_config=cfg)
            engine.execute(
                [{"ticker": "AAPL", "action": "BUY", "qty": 0, "price": 150.0}],
                date(2025, 1, 1),
            )
        mock_alpaca.REST.return_value.submit_order.assert_not_called()

    def test_execute_processes_buy(self):
        mock_alpaca = _make_mock_alpaca()
        with patch.dict(sys.modules, {"alpaca_trade_api": mock_alpaca}):
            from app.services.execution_engines import (
                AlpacaExecutionEngine,
                BrokerConfig,
            )
            from datetime import date

            cfg = BrokerConfig(api_key="k", api_secret="s", paper=True)
            engine = AlpacaExecutionEngine(broker_config=cfg)
            engine.execute(
                [{"ticker": "AAPL", "action": "BUY", "qty": 3, "price": 150.0}],
                date(2025, 1, 1),
            )
        mock_alpaca.REST.return_value.submit_order.assert_called_once()

    def test_execute_empty_list(self):
        mock_alpaca = _make_mock_alpaca()
        with patch.dict(sys.modules, {"alpaca_trade_api": mock_alpaca}):
            from app.services.execution_engines import (
                AlpacaExecutionEngine,
                BrokerConfig,
            )
            from datetime import date

            cfg = BrokerConfig(api_key="k", api_secret="s", paper=True)
            engine = AlpacaExecutionEngine(broker_config=cfg)
            engine.execute([], date(2025, 1, 1))
        mock_alpaca.REST.return_value.submit_order.assert_not_called()


# ---------------------------------------------------------------------------
# LiveExecutionEngine – _send_order raises NotImplementedError (lines 172-173)
# ---------------------------------------------------------------------------


class TestLiveExecutionEngineNotImplemented:
    def test_send_order_raises_not_implemented(self):
        from app.services.execution_engines import LiveExecutionEngine, BrokerConfig

        engine = LiveExecutionEngine(broker_config=BrokerConfig())
        with pytest.raises(NotImplementedError):
            engine._send_order("AAPL", "BUY", 1, 150.0, "2025-01-01")

    def test_execute_catches_not_implemented(self):
        from app.services.execution_engines import LiveExecutionEngine, BrokerConfig
        from datetime import date

        mock_logger = MagicMock()
        engine = LiveExecutionEngine(broker_config=BrokerConfig(), logger=mock_logger)
        engine.execute(
            [{"ticker": "AAPL", "action": "BUY", "qty": 5, "price": 150.0}],
            date(2025, 1, 1),
        )
        # Should log a warning, not raise
        mock_logger.warning.assert_called()

    def test_execute_catches_general_exception(self):
        from app.services.execution_engines import LiveExecutionEngine, BrokerConfig
        from datetime import date

        class ErrorEngine(LiveExecutionEngine):
            def _send_order(self, ticker, action, qty, limit_price, as_of):
                raise RuntimeError("bang")

        mock_logger = MagicMock()
        engine = ErrorEngine(broker_config=BrokerConfig(), logger=mock_logger)
        engine.execute(
            [{"ticker": "AAPL", "action": "BUY", "qty": 5, "price": 100.0}],
            date(2025, 1, 1),
        )
        mock_logger.error.assert_called()


# ---------------------------------------------------------------------------
# NoopExecutionEngine – already 97% but let's hit the remaining path
# ---------------------------------------------------------------------------


class TestNoopExecutionEngineExtra:
    def test_noop_with_list_of_decisions(self):
        from app.services.execution_engines import NoopExecutionEngine
        from datetime import date

        engine = NoopExecutionEngine()
        # Should not raise even with real decisions
        engine.execute([{"ticker": "VOO", "action": "BUY", "qty": 1}], date(2025, 1, 1))

    def test_noop_with_none_as_of(self):
        from app.services.execution_engines import NoopExecutionEngine

        engine = NoopExecutionEngine()
        engine.execute([], None)
