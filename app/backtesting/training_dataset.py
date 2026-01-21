def build_training_row(
    replay_row: dict,
    record: dict,
    overconfidence_cases: list[dict],
) -> dict:
    """
    P8.3 – Build one training row from replay + history record
    """

    ts = record["timestamp"]

    overconfident = any(c["timestamp"] == ts for c in overconfidence_cases)

    decision = record.get("decision", {})
    audit = record.get("audit", {})

    return {
        "ticker": record.get("ticker"),
        "timestamp": ts,
        # state
        "confidence": decision.get("confidence"),
        "wf_score": decision.get("wf_score"),
        "confidence_bucket": audit.get("confidence_bucket"),
        "decision_level": audit.get("decision_level"),
        "ensemble_quality": decision.get("ensemble_quality"),
        "quality_score": decision.get("quality_score"),
        # action
        "action_code": decision.get("action_code"),
        # outcome
        "raw_return": replay_row.get("raw_return"),
        "reward": replay_row.get("reward"),
        # flags
        "no_trade": decision.get("no_trade", False),
        "overconfident": overconfident,
    }
