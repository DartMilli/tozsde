from dataclasses import dataclass
from typing import Dict, Mapping, Optional


DEFAULT_POSITIONING = {
    "P6_POSITION_MAX_PCT": 0.1,
    "P6_MIN_CONF_FACTOR": 0.1,
    "P6_MIN_WF_FACTOR": 0.1,
    "P6_SAFETY_DISCOUNT": 0.25,
}


@dataclass
class PositionSizingResult:
    final_size: float
    base_size: float
    confidence_factor: float
    wf_factor: float
    safety_factor: float
    capped: bool


class PositionSizer:
    def __init__(
        self,
        max_position_pct: Optional[float] = None,
        params: Optional[Mapping[str, float]] = None,
    ):
        cfg = dict(DEFAULT_POSITIONING)
        if params:
            cfg.update({k: float(v) for k, v in params.items()})
        self._cfg = cfg
        self.max_position_pct = (
            float(max_position_pct)
            if max_position_pct is not None
            else float(cfg["P6_POSITION_MAX_PCT"])
        )

    def compute(
        self,
        base_position_size: float,
        confidence: float,
        wf_score: float,
        safety_discount: float,
        equity: float,
    ) -> PositionSizingResult:
        confidence = float(confidence or 0.0)
        wf_score = float(wf_score or 0.0)

        conf_min = float(self._cfg["P6_MIN_CONF_FACTOR"])
        wf_min = float(self._cfg["P6_MIN_WF_FACTOR"])

        confidence_factor = max(conf_min, min(1.0, confidence))
        wf_factor = max(wf_min, min(1.0, wf_score))
        safety_factor = max(0.0, min(1.0, 1.0 - safety_discount))

        scaled_size = base_position_size * confidence_factor * wf_factor * safety_factor

        cap_amount = max(0.0, equity * self.max_position_pct)
        capped = False
        if scaled_size > cap_amount:
            scaled_size = cap_amount
            capped = True

        return PositionSizingResult(
            final_size=round(float(scaled_size), 2),
            base_size=round(float(base_position_size), 2),
            confidence_factor=round(confidence_factor, 4),
            wf_factor=round(wf_factor, 4),
            safety_factor=round(safety_factor, 4),
            capped=capped,
        )


def apply_position_sizing(
    item: Dict,
    equity: float,
    max_position_pct: Optional[float] = None,
    params: Optional[Mapping[str, float]] = None,
) -> Dict:
    if item.get("decision", {}).get("action_code") != 1:
        return item

    base_size = float(item.get("allocation_amount", 0.0))
    if base_size <= 0:
        return item

    decision = item.get("decision", {})
    confidence = decision.get("confidence", 0.0)
    wf_score = decision.get("wf_score", 0.0)
    safety_override = decision.get("safety_override")

    cfg = dict(DEFAULT_POSITIONING)
    if params:
        cfg.update({k: float(v) for k, v in params.items()})

    safety_discount = float(cfg["P6_SAFETY_DISCOUNT"]) if safety_override else 0.0

    sizer = PositionSizer(max_position_pct=max_position_pct, params=cfg)
    result = sizer.compute(
        base_position_size=base_size,
        confidence=confidence,
        wf_score=wf_score,
        safety_discount=safety_discount,
        equity=equity,
    )

    item["allocation_amount"] = result.final_size
    item["allocation_pct"] = round(result.final_size / equity, 4) if equity > 0 else 0.0
    item["position_sizing"] = {
        "base_size": result.base_size,
        "final_size": result.final_size,
        "confidence_factor": result.confidence_factor,
        "wf_factor": result.wf_factor,
        "safety_factor": result.safety_factor,
        "capped": result.capped,
    }
    return item
