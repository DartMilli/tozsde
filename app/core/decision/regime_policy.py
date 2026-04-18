from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class RegimePolicy:
    confidence_floor: float
    max_position_pct: float
    safety_strictness: str
    allow_new_buys: bool
    ensemble_quality_floor: str


REGIME_POLICIES: Dict[str, RegimePolicy] = {
    "BULL": RegimePolicy(0.20, 0.20, "RELAXED", True, "WEAK"),
    "RANGING": RegimePolicy(0.30, 0.15, "NORMAL", True, "NORMAL"),
    "BEAR": RegimePolicy(0.50, 0.10, "STRICT", False, "STRONG"),
    "VOLATILE": RegimePolicy(0.45, 0.08, "STRICT", False, "STRONG"),
    "TREND": RegimePolicy(0.25, 0.18, "NORMAL", True, "NORMAL"),
    "RANGE": RegimePolicy(0.30, 0.15, "NORMAL", True, "NORMAL"),
    "TRANSITION": RegimePolicy(0.35, 0.12, "STRICT", True, "NORMAL"),
    "UNKNOWN": RegimePolicy(0.25, 0.15, "NORMAL", True, "NORMAL"),
}


def get_regime_policy(regime: str, settings=None) -> RegimePolicy:
    normalized_regime = (regime or "UNKNOWN").upper()
    base = REGIME_POLICIES.get(normalized_regime, REGIME_POLICIES["UNKNOWN"])
    overrides = (
        getattr(settings, "REGIME_POLICY_OVERRIDES", {}) if settings is not None else {}
    ) or {}
    override = overrides.get(normalized_regime, {})
    if not override:
        return base

    merged: Dict[str, Any] = asdict(base)
    merged.update(override)
    return RegimePolicy(**merged)
