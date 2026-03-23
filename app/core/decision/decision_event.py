def build_decision_event(
    payload: dict, decision: dict, audit: dict, settings=None
) -> dict:
    return {
        "ticker": payload.get("ticker"),
        "timestamp": payload.get("timestamp"),
        "action": decision["action"],
        "confidence": decision.get("confidence"),
        "no_trade": decision.get("no_trade"),
        "decision_reason": decision.get("decision_reason"),
        "reward_hint": decision.get("reward_hint"),
        "decision_level": audit.get("decision_level"),
        "trade_allowed": audit.get("trade_allowed"),
        "confidence_bucket": audit.get("confidence_bucket"),
        "quality_score": audit.get("quality_score"),
        "ensemble_quality": audit.get("ensemble_quality"),
        "model_count": len(payload.get("model_votes", [])),
    }
