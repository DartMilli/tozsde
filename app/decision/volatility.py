"""DEPRECATED: compatibility shim - use ``app.core.decision`` directly. Will be removed in a future release."""

from app.decision import emit_compat_warning

emit_compat_warning("app.decision.volatility")

from app.core.decision.volatility import (  # noqa: F401
    compute_normalized_volatility,
    scale_confidence_by_volatility,
)
