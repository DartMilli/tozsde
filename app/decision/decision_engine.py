"""DEPRECATED: compatibility shim - use ``app.core.decision`` directly. Will be removed in a future release."""

from app.decision import emit_compat_warning

emit_compat_warning("app.decision.decision_engine")

from app.core.decision.decision_engine import DecisionEngine  # noqa: F401
