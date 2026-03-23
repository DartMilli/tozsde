from typing import Dict, Mapping, Optional


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


def apply_decision_policy(
    decision: Dict,
    audit: Dict,
    hold_action_label: Optional[str] = None,
    reason_map: Optional[Mapping[str, str]] = None,
    reward_map: Optional[Mapping[str, float]] = None,
) -> Dict:
    local_reason_map = dict(DECISION_REASON_MAP)
    if reason_map:
        local_reason_map.update(reason_map)

    local_reward_map = dict(DECISION_REWARD_MAP)
    if reward_map:
        local_reward_map.update({k: float(v) for k, v in reward_map.items()})

    if not audit.get("trade_allowed", True):
        decision["action_code"] = 0
        if hold_action_label is not None:
            decision["action"] = hold_action_label
        decision["no_trade"] = True
    else:
        decision["no_trade"] = False

    decision_level = audit.get("decision_level", "UNKNOWN")

    decision["decision_reason"] = local_reason_map.get(
        decision_level,
        "UNSPECIFIED_REASON",
    )

    if decision.get("no_trade"):
        decision["decision_reason"] = "NO_TRADE_ENFORCED"

    decision["reward_hint"] = local_reward_map.get(
        decision["decision_reason"],
        0.0,
    )

    return decision
