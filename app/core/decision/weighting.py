try:
    from app.models.model_reliability import ModelReliabilityStore
except Exception:

    class ModelReliabilityStore:  # type: ignore[no-redef]
        def get(self, model_path):
            return None


ENSEMBLE_QUALITY_PENALTY = {
    "STABLE": 1.0,
    "MIXED": 0.75,
    "CHAOTIC": 0.4,
}


def compute_decision_weight(decision: dict) -> float:
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

        reliability = reliability_store.get(model_path)
        if reliability is not None:
            reliabilities.append(reliability)

    avg_reliability = sum(reliabilities) / len(reliabilities) if reliabilities else 0.5

    ensemble_quality = decision.get("ensemble_quality", "CHAOTIC")
    quality_penalty = ENSEMBLE_QUALITY_PENALTY.get(ensemble_quality, 0.4)

    weight = confidence * avg_reliability * wf_score * quality_penalty

    return round(min(max(weight, 0.0), 1.0), 3)
