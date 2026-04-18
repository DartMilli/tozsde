"""DEPRECATED: compatibility shim - use ``app.core.decision`` directly. Will be removed in a future release."""

from app.decision import emit_compat_warning

emit_compat_warning("app.decision.rebalancer")

from app.core.decision.rebalancer import (  # noqa: F401
    PortfolioRebalancer,
    check_and_rebalance,
)
