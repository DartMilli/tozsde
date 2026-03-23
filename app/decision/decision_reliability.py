from app.bootstrap.build_settings import build_settings
from app.core.decision.decision_reliability import (
    DecisionReliabilityResult,
    assess_decision_reliability as core_assess_decision_reliability,
)


def _get_settings(settings):
    return settings or build_settings()


def assess_decision_reliability(final_confidence, wf_score, settings=None):
    # type: (float, float) -> DecisionReliabilityResult
    cfg = _get_settings(settings)
    thresholds = {
        "CONFIDENCE_NO_TRADE_THRESHOLD": getattr(cfg, "CONFIDENCE_NO_TRADE_THRESHOLD"),
        "STRONG_CONFIDENCE_THRESHOLD": getattr(cfg, "STRONG_CONFIDENCE_THRESHOLD"),
        "WEAK_CONFIDENCE_THRESHOLD": getattr(cfg, "WEAK_CONFIDENCE_THRESHOLD"),
        "STRONG_WF_THRESHOLD": getattr(cfg, "STRONG_WF_THRESHOLD"),
    }
    return core_assess_decision_reliability(
        final_confidence=final_confidence,
        wf_score=wf_score,
        thresholds=thresholds,
    )
