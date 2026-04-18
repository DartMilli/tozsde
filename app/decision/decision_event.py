"""DEPRECATED: compatibility shim - use ``app.core.decision`` directly. Will be removed in a future release."""

from app.decision import emit_compat_warning

emit_compat_warning("app.decision.decision_event")

from app.core.decision.decision_event import build_decision_event  # noqa: F401
