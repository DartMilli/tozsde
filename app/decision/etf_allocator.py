"""DEPRECATED: compatibility shim - use ``app.core.decision`` directly. Will be removed in a future release."""

from app.decision import emit_compat_warning

emit_compat_warning("app.decision.etf_allocator")

from app.core.decision.etf_allocator import (  # noqa: F401
    AssetType,
    CostComparison,
    PortfolioMix,
    ETFAllocator,
    get_low_cost_etf,
    estimate_portfolio_cost,
    classify_asset_type,
)
