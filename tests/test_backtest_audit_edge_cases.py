"""
Edge case tests for backtest_audit utilities.
"""

from app.backtesting.backtest_audit import (
    audit_confidence_buckets,
    audit_decision_levels,
    detect_overconfidence,
)


def test_audit_confidence_buckets_basic():
    rows = [
        {"confidence_bucket": "HIGH", "reward": 1.0},
        {"confidence_bucket": "HIGH", "reward": -0.5},
        {"confidence_bucket": "LOW", "reward": 0.2},
        {"confidence_bucket": "LOW", "reward": 0.0},
        {"confidence_bucket": "LOW", "reward": None},
        {"reward": 0.3},  # UNKNOWN bucket
    ]

    summary = audit_confidence_buckets(rows)

    assert summary["HIGH"]["count"] == 2
    assert summary["HIGH"]["avg_reward"] == 0.25
    assert summary["HIGH"]["win_rate"] == 0.5

    assert summary["LOW"]["count"] == 2
    assert summary["LOW"]["avg_reward"] == 0.1
    assert summary["LOW"]["win_rate"] == 0.5

    assert summary["UNKNOWN"]["count"] == 1


def test_audit_confidence_buckets_filters_none_rewards():
    rows = [
        {"confidence_bucket": "MID", "reward": None},
        {"confidence_bucket": "MID"},
    ]

    summary = audit_confidence_buckets(rows)

    assert summary == {}


def test_audit_decision_levels_basic():
    rows = [
        {"decision_level": "L1", "reward": 1.0},
        {"decision_level": "L1", "reward": -1.0},
        {"decision_level": "L2", "reward": 0.5},
        {"decision_level": "L2", "reward": 0.0},
        {"reward": 0.2},  # UNKNOWN level
    ]

    summary = audit_decision_levels(rows)

    assert summary["L1"]["count"] == 2
    assert summary["L1"]["avg_reward"] == 0.0
    assert summary["L1"]["win_rate"] == 0.5

    assert summary["L2"]["count"] == 2
    assert summary["L2"]["avg_reward"] == 0.25
    assert summary["L2"]["win_rate"] == 0.5

    assert summary["UNKNOWN"]["count"] == 1


def test_detect_overconfidence_flags_only_negative_high_confidence():
    rows = [
        {
            "timestamp": "2025-01-01",
            "confidence": 0.8,
            "reward": -0.1,
            "decision_level": "L1",
        },
        {
            "timestamp": "2025-01-02",
            "confidence": 0.9,
            "reward": 0.2,
            "decision_level": "L2",
        },
        {"timestamp": "2025-01-03", "confidence": 0.6, "reward": -0.5},
        {"timestamp": "2025-01-04", "confidence": None, "reward": -0.5},
        {"timestamp": "2025-01-05", "confidence": 0.8, "reward": None},
    ]

    flags = detect_overconfidence(rows)

    assert len(flags) == 1
    assert flags[0]["timestamp"] == "2025-01-01"
    assert flags[0]["confidence"] == 0.8
    assert flags[0]["reward"] == -0.1
