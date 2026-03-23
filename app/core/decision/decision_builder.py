from collections import defaultdict
from typing import Dict, Mapping, Optional


DEFAULT_ACTION_LABELS = {
    "hu": {0: "TART", 1: "VESZ", 2: "ELAD"},
    "en": {0: "HOLD", 1: "BUY", 2: "SELL"},
}


def weighted_ensemble_decision(
    payload: Dict,
    reliability_scores: Mapping[str, float],
    action_labels: Optional[Mapping[int, str]] = None,
) -> Dict:
    labels = dict(action_labels or DEFAULT_ACTION_LABELS["en"])

    action_weights = defaultdict(float)
    vote_debug = []

    for mv in payload["model_votes"]:
        model_path = mv["model_path"]

        confidence = mv["confidence"]
        wf_score = mv.get("wf_score", 1.0)
        reliability = reliability_scores.get(model_path, 0.5)

        weight = confidence * wf_score * reliability

        action = mv["action"]
        action_weights[action] += weight

        vote_debug.append(
            {
                "model_path": model_path,
                "action": labels.get(action, str(action)),
                "confidence": confidence,
                "wf_score": wf_score,
                "reliability": reliability,
                "weight": round(weight, 4),
            }
        )

    final_action = max(action_weights, key=action_weights.get)
    total_weight = sum(action_weights.values())

    normalized_confidence = (
        action_weights[final_action] / total_weight if total_weight > 0 else 0.0
    )

    dominance = action_weights[final_action] / total_weight if total_weight > 0 else 0.0

    return {
        "action": labels.get(final_action, str(final_action)),
        "action_code": final_action,
        "weighted_confidence": round(normalized_confidence, 3),
        "dominance": round(dominance, 3),
        "action_weights": dict(action_weights),
        "vote_debug": vote_debug,
    }


def compute_decision_quality(payload: Dict) -> float:
    avg_conf = payload.get("avg_confidence", 0.0)
    avg_wf = payload.get("avg_wf_score", 0.0)
    ensemble_quality = payload.get("ensemble_quality", "CHAOTIC")

    if avg_conf is None:
        avg_conf = 0.0
    if avg_wf is None:
        avg_wf = 0.0

    ensemble_score_map = {
        "STABLE": 1.0,
        "MIXED": 0.7,
        "CHAOTIC": 0.4,
    }

    ensemble_score = ensemble_score_map.get(ensemble_quality, 0.4)

    score = 0.5 * avg_conf + 0.3 * avg_wf + 0.2 * ensemble_score

    return round(float(score), 4)
