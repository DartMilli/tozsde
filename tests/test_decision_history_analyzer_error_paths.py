"""Error-path and edge-case tests for DecisionHistoryAnalyzer."""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from dataclasses import replace

import pandas as pd
import pytest

import app.decision.decision_history_analyzer as dha
from app.decision import set_settings as set_decision_settings
from app.decision.decision_history_analyzer import DecisionHistoryAnalyzer


@pytest.fixture
def temp_db(tmp_path, test_settings):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE decision_history (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            ticker TEXT,
            action_code INTEGER,
            audit_data TEXT
        )
        """
    )
    conn.commit()
    conn.close()

    settings = replace(test_settings, DB_PATH=db_path)
    set_decision_settings(settings)

    return settings


def _insert_decision(settings, timestamp, ticker, audit_data):
    conn = sqlite3.connect(str(settings.DB_PATH))
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO decision_history (timestamp, ticker, action_code, audit_data) VALUES (?, ?, ?, ?)",
        (timestamp, ticker, 1, audit_data),
    )
    conn.commit()
    conn.close()


def test_load_strategy_decisions_db_error(temp_db):
    analyzer = DecisionHistoryAnalyzer(settings=temp_db)

    with patch(
        "app.decision.decision_history_analyzer.sqlite3.connect",
        side_effect=Exception("boom"),
    ):
        decisions = analyzer._load_strategy_decisions("MA_CROSS", days=30)

    assert decisions == []


def test_load_ticker_decisions_skips_bad_json(temp_db):
    now = datetime.now().isoformat()
    _insert_decision(temp_db, now, "AAPL", "{bad json")
    _insert_decision(
        temp_db,
        now,
        "AAPL",
        json.dumps({"outcome": {"pnl_pct": 0.02}, "confidence": 0.8}),
    )

    analyzer = DecisionHistoryAnalyzer(settings=temp_db)
    decisions = analyzer._load_ticker_decisions("AAPL", days=30)

    assert len(decisions) == 1
    assert decisions[0]["outcome"]["pnl_pct"] == 0.02


def test_load_all_decisions_skips_bad_json(temp_db):
    now = datetime.now().isoformat()
    _insert_decision(temp_db, now, "AAPL", "{bad json")
    _insert_decision(
        temp_db,
        now,
        "AAPL",
        json.dumps({"outcome": {"pnl_pct": -0.01}}),
    )

    analyzer = DecisionHistoryAnalyzer(settings=temp_db)
    decisions = analyzer._load_all_decisions(days=30)

    assert len(decisions) == 1
    assert decisions[0]["outcome"]["pnl_pct"] == -0.01


def test_calculate_confidence_calibration_insufficient_data():
    analyzer = DecisionHistoryAnalyzer()
    decisions = [
        {"confidence": 0.9, "outcome": {"pnl_pct": 0.01}},
        {"confidence": 0.8, "outcome": {"pnl_pct": -0.01}},
        {"confidence": 0.7, "outcome": {"pnl_pct": 0.02}},
    ]

    result = analyzer._calculate_confidence_calibration(decisions)

    assert result == 0.5


def test_calculate_confidence_calibration_missing_buckets():
    analyzer = DecisionHistoryAnalyzer()
    decisions = [
        {"confidence": 0.9, "outcome": {"pnl_pct": 0.01}},
        {"confidence": 0.95, "outcome": {"pnl_pct": 0.02}},
        {"confidence": 0.8, "outcome": {"pnl_pct": -0.01}},
        {"confidence": 0.75, "outcome": {"pnl_pct": 0.01}},
        {"confidence": 0.85, "outcome": {"pnl_pct": 0.01}},
    ]

    result = analyzer._calculate_confidence_calibration(decisions)

    assert result == 0.5


def test_calculate_confidence_calibration_full_buckets():
    analyzer = DecisionHistoryAnalyzer()
    decisions = [
        {"confidence": 0.8, "outcome": {"pnl_pct": 0.02}},
        {"confidence": 0.9, "outcome": {"pnl_pct": 0.01}},
        {"confidence": 0.35, "outcome": {"pnl_pct": -0.02}},
        {"confidence": 0.3, "outcome": {"pnl_pct": -0.01}},
        {"confidence": 0.2, "outcome": {"pnl_pct": 0.01}},
    ]

    result = analyzer._calculate_confidence_calibration(decisions)

    assert 0.0 <= result <= 1.0


def test_is_recent_handles_invalid_timestamp():
    analyzer = DecisionHistoryAnalyzer()
    decision = {"timestamp": "not-a-date"}

    assert analyzer._is_recent(decision, days=30) is False


def test_compute_rolling_metrics_no_decisions(monkeypatch):
    analyzer = DecisionHistoryAnalyzer()

    monkeypatch.setattr(analyzer, "_load_all_decisions", lambda days: [])
    df = analyzer.compute_rolling_metrics(window_days=30)

    assert isinstance(df, pd.DataFrame)
    assert df.empty
