"""
P7 – Decision weighting (SKELETON)

This module will handle:
- model reliability
- ensemble dominance
- confidence normalization

P7.0: no active logic yet.
"""

from app.models.model_reliability import ModelReliabilityStore

ENSEMBLE_QUALITY_PENALTY = {
    "STABLE": 1.0,
    "MIXED": 0.75,
    "CHAOTIC": 0.4,
}


def compute_decision_weight(decision: dict) -> float:
    """
    P7.1 – Compute decision weight in [0..1]

    Components:
    - avg confidence
    - average model reliability
    - ensemble quality penalty
    """

    confidence = decision.get("confidence", 0.0)
    wf_score = decision.get("avg_wf_score", 1.0)

    model_votes = decision.get("model_votes", [])
    if not model_votes:
        return 0.0

    reliability_store = ModelReliabilityStore()
    reliabilities = []

    for mv in model_votes:
        model_path = mv.get("model_path")
        if not model_path:
            continue

        r = reliability_store.get(model_path)
        if r is not None:
            reliabilities.append(r)

    avg_reliability = sum(reliabilities) / len(reliabilities) if reliabilities else 0.5

    ensemble_quality = decision.get("ensemble_quality", "CHAOTIC")
    quality_penalty = ENSEMBLE_QUALITY_PENALTY.get(ensemble_quality, 0.4)

    weight = confidence * avg_reliability * wf_score * quality_penalty

    return round(min(max(weight, 0.0), 1.0), 3)
