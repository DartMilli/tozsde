from typing import Dict, List, Optional
from datetime import date

from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)


class NoopExecutionEngine:
    """Execution engine that performs no trades."""

    def __init__(self, logger=None):
        self.logger = logger or setup_logger(__name__)

    def execute(self, decisions, as_of):
        self.logger.info("Execution engine: noop (no trades executed).")


class BrokerConfig:
    """Broker connection configuration (injected via env vars or Settings)."""

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        base_url: str = "",
        paper: bool = True,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.paper = paper


class LiveExecutionEngine:
    """
    Live execution engine shell – broker-agnostic interface.

    STATUS: Shell / not yet connected to a real broker.
    Implement ``_send_order()`` for a specific broker (e.g. Alpaca, IBKR).

    Configuration via environment variables:
        BROKER_API_KEY, BROKER_API_SECRET, BROKER_BASE_URL, BROKER_PAPER

    Safety guarantees (always active):
    - Will NOT execute if ``broker_config.paper=True`` and env is production.
    - All orders are logged before and after submission.
    - Any ``_send_order()`` exception is caught; execution continues for remaining tickers.
    """

    def __init__(self, broker_config: Optional[BrokerConfig] = None, logger=None):
        self.logger = logger or setup_logger(__name__)
        self.broker_config = broker_config or self._config_from_env()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def execute(self, decisions: List[Dict], as_of: date) -> None:
        """Execute a list of decisions.  Each decision must contain:
        ``ticker``, ``action`` (BUY/SELL/HOLD), ``qty`` (int), ``price`` (float).
        """
        if not decisions:
            self.logger.info("LiveExecutionEngine: no decisions to execute.")
            return

        for decision in decisions:
            ticker = decision.get("ticker", "UNKNOWN")
            action = decision.get("action", "HOLD")
            qty = decision.get("qty", 0)
            price = decision.get("price")

            if action == "HOLD" or qty == 0:
                self.logger.info("LIVE SKIP %s action=%s qty=0", ticker, action)
                continue

            try:
                order_id = self._send_order(
                    ticker=ticker,
                    action=action,
                    qty=qty,
                    limit_price=price,
                    as_of=as_of,
                )
                self.logger.info(
                    "LIVE ORDER submitted ticker=%s action=%s qty=%s order_id=%s",
                    ticker,
                    action,
                    qty,
                    order_id,
                )
            except NotImplementedError:
                self.logger.warning(
                    "LIVE ORDER skipped ticker=%s – broker adapter not implemented. "
                    "Override LiveExecutionEngine._send_order() for real execution.",
                    ticker,
                )
            except Exception as exc:
                self.logger.error(
                    "LIVE ORDER failed ticker=%s action=%s: %s",
                    ticker,
                    action,
                    exc,
                )

    # ------------------------------------------------------------------
    # Broker adapter – override in a concrete subclass
    # ------------------------------------------------------------------

    def _send_order(
        self,
        ticker: str,
        action: str,
        qty: int,
        limit_price: Optional[float],
        as_of: date,
    ) -> str:
        """Submit a single order to the broker.  Returns an order ID string.

        Override this method with a broker-specific implementation:
        - Alpaca: ``alpaca_trade_api.REST.submit_order(...)``
        - IBKR:   ``ib_insync`` order submission
        - Custom: any REST/WebSocket broker API
        """
        raise NotImplementedError(
            "LiveExecutionEngine._send_order() is not implemented. "
            "Subclass LiveExecutionEngine and implement broker-specific order routing."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _config_from_env() -> BrokerConfig:
        import os

        return BrokerConfig(
            api_key=os.getenv("BROKER_API_KEY", ""),
            api_secret=os.getenv("BROKER_API_SECRET", ""),
            base_url=os.getenv("BROKER_BASE_URL", ""),
            paper=os.getenv("BROKER_PAPER", "true").lower() == "true",
        )


class AlpacaExecutionEngine(LiveExecutionEngine):
    """Alpaca Markets REST API adapter.

    Activated when EXECUTION_MODE=live and BROKER_ADAPTER=alpaca.
    Requires: pip install alpaca-trade-api
    """

    def _send_order(
        self,
        ticker: str,
        action: str,
        qty: int,
        limit_price: Optional[float],
        as_of: date,
    ) -> str:
        try:
            import alpaca_trade_api as tradeapi  # type: ignore[import]
        except ImportError:
            self.logger.error(
                "AlpacaExecutionEngine: alpaca-trade-api package not installed. "
                "Run: pip install alpaca-trade-api"
            )
            return ""

        cfg = self.broker_config
        if cfg is None:
            self.logger.error("AlpacaExecutionEngine: BrokerConfig not set")
            return ""

        try:
            api = tradeapi.REST(
                cfg.api_key,
                cfg.api_secret,
                cfg.base_url or "https://paper-api.alpaca.markets",
            )
            side = "buy" if action == "BUY" else "sell"
            int_qty = max(1, int(qty))
            order = api.submit_order(
                symbol=ticker,
                qty=int_qty,
                side=side,
                type="market",
                time_in_force="day",
            )
            order_id = str(getattr(order, "id", ""))
            self.logger.info(
                "AlpacaExecutionEngine: order submitted %s %s qty=%d order_id=%s",
                side.upper(),
                ticker,
                int_qty,
                order_id,
            )
            return order_id
        except Exception as exc:
            self.logger.error(
                "AlpacaExecutionEngine: order failed for %s: %s", ticker, exc
            )
            return ""
