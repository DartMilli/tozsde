import datetime
from typing import Dict, List, Optional, Tuple


def aggregate_weighted_ensemble(
    votes,
    confidences,
    wf_scores: List[Optional[float]],
    model_votes: List[Dict],
    settings=None,
) -> Tuple[int, float, float]:
    assert len(votes) == len(confidences) == len(wf_scores)

    action_scores = {0: 0.0, 1: 0.0, 2: 0.0}
    total_weight = 0.0

    for action, conf, wf, model_vote in zip(votes, confidences, wf_scores, model_votes):
        if conf is None:
            continue

        rank = model_vote.get("rank")
        trained_at = model_vote.get("trained_at")

        weight = conf

        if wf is not None:
            min_wf = (
                getattr(settings, "MIN_WF_WEIGHT", 0.0) if settings is not None else 0.0
            )
            weight *= max(wf, min_wf)

        alpha = getattr(settings, "RANK_ALPHA", 0.7) if settings is not None else 0.7
        weight *= compute_rank_weight(rank, alpha=alpha)
        half_life = (
            getattr(settings, "RECENCY_HALF_LIFE_DAYS", 90)
            if settings is not None
            else 90
        )
        weight *= compute_recency_weight(
            trained_at, today=datetime.date.today(), half_life_days=half_life
        )

        action_scores[action] += weight
        total_weight += weight

    if total_weight == 0.0:
        return 0, 0.0, 0.0

    final_action = max(action_scores, key=action_scores.get)
    final_confidence = action_scores[final_action] / total_weight

    sorted_scores = sorted(action_scores.values(), reverse=True)
    ensemble_quality = sorted_scores[0] - sorted_scores[-1]

    return final_action, final_confidence, ensemble_quality


def compute_rank_weight(rank: int, alpha: float = 0.7) -> float:
    if rank is None or rank < 1:
        return 1.0
    return alpha ** (rank - 1)


def compute_recency_weight(
    trained_at,
    today,
    half_life_days=90,
) -> float:
    if trained_at is None:
        return 1.0
    age_days = (today - trained_at).days
    return 0.5 ** (age_days / half_life_days)
