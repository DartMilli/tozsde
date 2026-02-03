"""
Tests for DecisionEngine behavior.
"""

from unittest.mock import MagicMock
from datetime import date

from app.decision.decision_engine import DecisionEngine


def test_run_returns_decision_without_safety():
    engine = DecisionEngine(enable_safety=False)
    decision = {"action": "BUY"}

    result = engine.run("AAPL", decision)

    assert result == decision


def test_run_applies_safety_engine_when_enabled():
    safety = MagicMock()
    safety.apply.return_value = {"action": "HOLD"}

    engine = DecisionEngine(safety_engine=safety, enable_safety=True, today=date(2025, 1, 1))
    decision = {"action": "BUY"}

    result = engine.run("AAPL", decision)

    assert result["action"] == "HOLD"
    safety.apply.assert_called_once()


def test_run_ignores_safety_engine_when_disabled():
    safety = MagicMock()
    safety.apply.return_value = {"action": "HOLD"}

    engine = DecisionEngine(safety_engine=safety, enable_safety=False)
    decision = {"action": "BUY"}

    result = engine.run("AAPL", decision)

    assert result == decision
    safety.apply.assert_not_called()
