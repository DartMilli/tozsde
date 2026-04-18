"""DEPRECATED: compatibility shim - use ``app.core.decision`` directly. Will be removed in a future release."""

from app.decision import emit_compat_warning

emit_compat_warning("app.decision.confidence_allocator")

from app.core.decision.confidence_allocator import (  # noqa: F401
    BucketStatistics,
    ConfidenceAllocation,
    ConfidenceBucket,
    ConfidenceBucketAllocator,
)
