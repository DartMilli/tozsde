"""DEPRECATED: compatibility shim - use ``app.core.decision`` directly. Will be removed in a future release."""

from app.decision import emit_compat_warning

emit_compat_warning("app.decision.volatility_bucket")

from app.core.decision.volatility_bucket import (  # noqa: F401
    VolatilityBucket,
    bucket_volatility,
)
