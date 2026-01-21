def realize_reward(
    raw_return: float,
    decision: dict,
    audit: dict,
) -> float:
    """
    P8.1 – Reward realization

    Converts raw_return into learning reward.
    """

    action = decision.get("action_code", 0)
    confidence = decision.get("confidence")
    decision_level = audit.get("decision_level", "UNKNOWN")

    # HOLD → no reward
    if action == 0:
        return 0.0

    # base
    reward = raw_return

    # confidence scaling
    if confidence is not None:
        confidence = max(0.3, min(confidence, 1.0))
        reward *= confidence

    # decision level modifier
    level_multiplier = {
        "STRONG": 1.2,
        "NORMAL": 1.0,
        "WEAK": 0.5,
        "NO_TRADE": 0.0,
        "UNKNOWN": 0.3,
    }.get(decision_level, 0.3)

    reward *= level_multiplier

    return round(reward, 6)
