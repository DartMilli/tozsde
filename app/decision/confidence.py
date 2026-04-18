"""DEPRECATED: compatibility shim - use ``app.core.decision`` directly. Will be removed in a future release."""

from app.decision import emit_compat_warning

emit_compat_warning("app.decision.confidence")

from app.core.decision.confidence import (  # noqa: F401
    apply_confidence,
    clamp,
    normalize_dqn_confidence,
    normalize_final_confidence,
    normalize_ppo_confidence,
)
