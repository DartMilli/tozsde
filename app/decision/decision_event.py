def build_decision_event(payload: dict, decision: dict, audit: dict) -> dict:
    decision_event = {
        # --- identity ---
        "ticker": payload.get("ticker"),
        "timestamp": payload.get("timestamp"),
        # --- decision ---
        "action": decision["action"],
        "confidence": decision.get("confidence"),
        "no_trade": decision.get("no_trade"),
        "decision_reason": decision.get("decision_reason"),
        "reward_hint": decision.get("reward_hint"),
        # --- audit ---
        "decision_level": audit.get("decision_level"),
        "trade_allowed": audit.get("trade_allowed"),
        "confidence_bucket": audit.get("confidence_bucket"),
        "quality_score": audit.get("quality_score"),
        "ensemble_quality": audit.get("ensemble_quality"),
        # --- ensemble context ---
        "model_count": len(payload.get("model_votes", [])),
    }
    return decision_event
