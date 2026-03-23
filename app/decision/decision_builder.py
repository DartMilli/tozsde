from app.decision import get_settings
from app.core.decision.decision_builder import (
    compute_decision_quality,
    weighted_ensemble_decision as core_weighted_ensemble_decision,
)


def weighted_ensemble_decision(payload: dict, reliability_scores: dict, settings=None):
    """
    payload: decision_payload (P3.6)
    reliability_scores: model_path -> reliability_score
    """

    cfg = settings or get_settings()
    action_labels = getattr(cfg, "ACTION_LABELS")[getattr(cfg, "LANG")]
    return core_weighted_ensemble_decision(
        payload=payload,
        reliability_scores=reliability_scores,
        action_labels=action_labels,
    )
