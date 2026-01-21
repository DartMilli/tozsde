# app/decision/decision_reliability.py

from dataclasses import dataclass
from app.config.config import Config


@dataclass
class DecisionReliabilityResult:
    trade_allowed: bool
    confidence_level: str  # STRONG | NORMAL | WEAK | NO_TRADE
    final_confidence: float
    wf_score: float


def assess_decision_reliability(final_confidence: float, wf_score: float | None):
    if final_confidence < Config.CONFIDENCE_NO_TRADE_THRESHOLD:
        return DecisionReliabilityResult(
            trade_allowed=False,
            confidence_level="NO_TRADE",
            final_confidence=final_confidence,
            wf_score=wf_score,
        )

    if (
        final_confidence >= Config.STRONG_CONFIDENCE_THRESHOLD
        and wf_score is not None
        and wf_score >= Config.STRONG_WF_THRESHOLD
    ):
        level = "STRONG"
    elif final_confidence < Config.WEAK_CONFIDENCE_THRESHOLD:
        level = "WEAK"
    else:
        level = "NORMAL"

    return DecisionReliabilityResult(
        trade_allowed=True,
        confidence_level=level,
        final_confidence=final_confidence,
        wf_score=wf_score,
    )
