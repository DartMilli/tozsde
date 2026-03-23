from dataclasses import dataclass
from typing import Mapping, Optional


DEFAULT_THRESHOLDS = {
    "CONFIDENCE_NO_TRADE_THRESHOLD": 0.25,
    "STRONG_CONFIDENCE_THRESHOLD": 0.75,
    "WEAK_CONFIDENCE_THRESHOLD": 0.5,
    "STRONG_WF_THRESHOLD": 0.6,
}


@dataclass
class DecisionReliabilityResult:
    trade_allowed: bool
    confidence_level: str
    final_confidence: float
    wf_score: float


def assess_decision_reliability(
    final_confidence: float,
    wf_score: Optional[float],
    thresholds: Optional[Mapping[str, float]] = None,
) -> DecisionReliabilityResult:
    cfg = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        cfg.update({k: float(v) for k, v in thresholds.items()})

    no_trade_thr = cfg["CONFIDENCE_NO_TRADE_THRESHOLD"]
    strong_conf_thr = cfg["STRONG_CONFIDENCE_THRESHOLD"]
    weak_conf_thr = cfg["WEAK_CONFIDENCE_THRESHOLD"]
    strong_wf_thr = cfg["STRONG_WF_THRESHOLD"]

    if final_confidence < no_trade_thr:
        return DecisionReliabilityResult(
            trade_allowed=False,
            confidence_level="NO_TRADE",
            final_confidence=final_confidence,
            wf_score=wf_score,
        )

    if (
        final_confidence >= strong_conf_thr
        and wf_score is not None
        and wf_score >= strong_wf_thr
    ):
        level = "STRONG"
    elif final_confidence < weak_conf_thr:
        level = "WEAK"
    else:
        level = "NORMAL"

    return DecisionReliabilityResult(
        trade_allowed=True,
        confidence_level=level,
        final_confidence=final_confidence,
        wf_score=wf_score,
    )
