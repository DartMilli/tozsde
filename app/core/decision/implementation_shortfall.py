from dataclasses import dataclass


@dataclass(frozen=True)
class ShortfallEstimate:
    commission_pct: float
    slippage_pct: float
    spread_pct: float
    total_cost_pct: float
    min_edge_required: float


class ImplementationShortfallEstimator:
    def __init__(self, settings) -> None:
        self.settings = settings

    def estimate(
        self, notional: float, atr_pct: float | None = None
    ) -> ShortfallEstimate:
        commission_pct = float(getattr(self.settings, "TRANSACTION_FEE_PCT", 0.0))
        slippage_pct = float(getattr(self.settings, "MIN_SLIPPAGE_PCT", 0.0))
        spread_pct = float(getattr(self.settings, "SPREAD_PCT", 0.0))

        if atr_pct is not None and atr_pct > 0:
            slippage_pct = max(slippage_pct, min(atr_pct * 0.1, 0.02))

        total_cost_pct = commission_pct + slippage_pct + spread_pct
        min_edge_required = total_cost_pct * float(
            getattr(self.settings, "COST_BUFFER_MULTIPLIER", 1.5)
        )

        return ShortfallEstimate(
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
            spread_pct=spread_pct,
            total_cost_pct=total_cost_pct,
            min_edge_required=min_edge_required,
        )
