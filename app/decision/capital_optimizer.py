"""DEPRECATED: compatibility shim - use ``app.core.decision`` directly. Will be removed in a future release."""

from app.decision import emit_compat_warning

emit_compat_warning("app.decision.capital_optimizer")

from app.core.decision.capital_optimizer import (  # noqa: F401
    CapitalAllocation,
    CapitalUtilizationOptimizer,
    PositionSize,
)
