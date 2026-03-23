"""Tests for decision policy application."""

from app.decision.decision_policy import apply_decision_policy


def test_policy_enforces_no_trade_when_blocked(test_settings):
    decision = {"action_code": 1, "action": "BUY"}
    audit = {"trade_allowed": False, "decision_level": "STRONG"}

    result = apply_decision_policy(decision, audit, settings=test_settings)

    assert result["no_trade"] is True
    assert result["action_code"] == 0
    assert result["action"] == test_settings.ACTION_LABELS[test_settings.LANG][0]
    assert result["decision_reason"] == "NO_TRADE_ENFORCED"
    assert result["reward_hint"] == -0.1


def test_policy_maps_decision_levels():
    decision = {"action_code": 1, "action": "BUY"}

    result = apply_decision_policy(
        decision, {"trade_allowed": True, "decision_level": "STRONG"}
    )
    assert result["decision_reason"] == "CONFIRMED_SIGNAL"
    assert result["reward_hint"] == 1.0

    result = apply_decision_policy(
        decision, {"trade_allowed": True, "decision_level": "NORMAL"}
    )
    assert result["decision_reason"] == "ACCEPTABLE_SIGNAL"
    assert result["reward_hint"] == 0.5

    result = apply_decision_policy(
        decision, {"trade_allowed": True, "decision_level": "WEAK"}
    )
    assert result["decision_reason"] == "LOW_CONFIDENCE"
    assert result["reward_hint"] == -0.2

    result = apply_decision_policy(
        decision, {"trade_allowed": True, "decision_level": "NO_TRADE"}
    )
    assert result["decision_reason"] == "BLOCKED_BY_RELIABILITY"
    assert result["reward_hint"] == -0.5

    result = apply_decision_policy(
        decision, {"trade_allowed": True, "decision_level": "UNKNOWN"}
    )
    assert result["decision_reason"] == "INSUFFICIENT_DATA"
    assert result["reward_hint"] == -0.3


def test_policy_defaults_to_unspecified_reason():
    decision = {"action_code": 1, "action": "BUY"}
    audit = {"trade_allowed": True, "decision_level": "UNSUPPORTED"}

    result = apply_decision_policy(decision, audit)

    assert result["decision_reason"] == "UNSPECIFIED_REASON"
    assert result["reward_hint"] == 0.0
