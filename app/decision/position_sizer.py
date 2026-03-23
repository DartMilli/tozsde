from typing import Dict

from app.bootstrap.build_settings import build_settings
from app.core.decision.position_sizer import (
    PositionSizingResult,
    PositionSizer as CorePositionSizer,
    apply_position_sizing as core_apply_position_sizing,
)


def _get_settings(settings):
    return settings or build_settings()


class PositionSizer:
    """
    Deterministic position sizing based on confidence, WF stability, and safety.
    """

    def __init__(self, max_position_pct: float = None, settings=None):
        self.settings = settings
        cfg = _get_settings(settings)
        self._params = {
            "P6_POSITION_MAX_PCT": getattr(cfg, "P6_POSITION_MAX_PCT"),
            "P6_MIN_CONF_FACTOR": getattr(cfg, "P6_MIN_CONF_FACTOR"),
            "P6_MIN_WF_FACTOR": getattr(cfg, "P6_MIN_WF_FACTOR"),
            "P6_SAFETY_DISCOUNT": getattr(cfg, "P6_SAFETY_DISCOUNT"),
        }
        self.max_position_pct = (
            max_position_pct
            if max_position_pct is not None
            else getattr(cfg, "P6_POSITION_MAX_PCT")
        )

    def compute(
        self,
        base_position_size: float,
        confidence: float,
        wf_score: float,
        safety_discount: float,
        equity: float,
    ) -> PositionSizingResult:
        core = CorePositionSizer(
            max_position_pct=self.max_position_pct,
            params=self._params,
        )
        return core.compute(
            base_position_size=base_position_size,
            confidence=confidence,
            wf_score=wf_score,
            safety_discount=safety_discount,
            equity=equity,
        )


def apply_position_sizing(
    item: Dict,
    equity: float,
    max_position_pct: float = None,
    settings=None,
) -> Dict:
    cfg = _get_settings(settings)
    params = {
        "P6_POSITION_MAX_PCT": getattr(cfg, "P6_POSITION_MAX_PCT"),
        "P6_MIN_CONF_FACTOR": getattr(cfg, "P6_MIN_CONF_FACTOR"),
        "P6_MIN_WF_FACTOR": getattr(cfg, "P6_MIN_WF_FACTOR"),
        "P6_SAFETY_DISCOUNT": getattr(cfg, "P6_SAFETY_DISCOUNT"),
    }
    return core_apply_position_sizing(
        item=item,
        equity=equity,
        max_position_pct=max_position_pct,
        params=params,
    )
