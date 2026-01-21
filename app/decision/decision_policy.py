from app.config.config import Config


def apply_decision_policy(decision: dict, audit: dict) -> dict:
    """
    P7.1.2 – P7.1.4
    Applies reliability-based policy, reason normalization and reward hint.
    Mutates and returns decision.
    """

    DECISION_REASON_MAP = {
        "STRONG": "CONFIRMED_SIGNAL",
        "NORMAL": "ACCEPTABLE_SIGNAL",
        "WEAK": "LOW_CONFIDENCE",
        "NO_TRADE": "BLOCKED_BY_RELIABILITY",
        "UNKNOWN": "INSUFFICIENT_DATA",
    }

    DECISION_REWARD_MAP = {
        "CONFIRMED_SIGNAL": +1.0,
        "ACCEPTABLE_SIGNAL": +0.5,
        "LOW_CONFIDENCE": -0.2,
        "BLOCKED_BY_RELIABILITY": -0.5,
        "INSUFFICIENT_DATA": -0.3,
        "NO_TRADE_ENFORCED": -0.1,
        "UNSPECIFIED_REASON": 0.0,
    }

    # --- P7.1.2 NO-TRADE ENFORCEMENT ---
    if not audit.get("trade_allowed", True):
        decision["action_code"] = 0
        decision["action"] = Config.ACTION_LABELS[Config.LANG][0]
        decision["no_trade"] = True
    else:
        decision["no_trade"] = False

    # --- P7.1.3 DECISION REASON ---
    decision_level = audit.get("decision_level", "UNKNOWN")

    decision["decision_reason"] = DECISION_REASON_MAP.get(
        decision_level,
        "UNSPECIFIED_REASON",
    )

    if decision.get("no_trade"):
        decision["decision_reason"] = "NO_TRADE_ENFORCED"

    # --- P7.1.4 REWARD HINT ---
    decision["reward_hint"] = DECISION_REWARD_MAP.get(
        decision["decision_reason"],
        0.0,
    )

    return decision
