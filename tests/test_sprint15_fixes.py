"""
Sprint 15 regression tests.

Covers:
- S15C: dh.confidence column-agnostic query in DecisionHistoryAnalyzer
- S15A: fetch_recent_outcomes real implementation
- S15B: LiveExecutionEngine shell
- S15B: PortfolioRebalancer wired into TradingPipelineService
- S15B: enforce_correlation_limits wired into TradingPipelineService
- S15B: ML ensemble vote (_try_add_ml_vote graceful skip)
"""

import json
import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# S15C – dh.confidence column agnostic
# ---------------------------------------------------------------------------


def test_load_ticker_decisions_no_confidence_column(test_settings, tmp_path):
    """Analyzer must work even if decision_history has no 'confidence' column."""
    from app.decision import set_settings as set_decision_settings
    from app.core.decision.decision_history_analyzer import DecisionHistoryAnalyzer

    db_path = tmp_path / "no_conf.db"
    settings = replace(test_settings, DB_PATH=db_path)

    conn = sqlite3.connect(str(db_path))
    conn.execute(
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
    now = datetime.now()
    for i in range(3):
        ts = (now - timedelta(days=i)).isoformat()
        audit = json.dumps(
            {"strategy": "MA", "confidence": 0.7, "outcome": {"pnl_pct": 0.02}}
        )
        conn.execute(
            "INSERT INTO decision_history (timestamp, ticker, action_code, audit_data) VALUES (?, ?, ?, ?)",
            (ts, "VOO", 1, audit),
        )
    conn.commit()
    conn.close()

    set_decision_settings(settings)
    try:
        analyzer = DecisionHistoryAnalyzer(settings=settings)
        decisions = analyzer._load_ticker_decisions("VOO", days=30)
        assert len(decisions) == 3, f"Expected 3 decisions, got {len(decisions)}"
        reliability = analyzer.analyze_ticker_reliability("VOO", days=30)
        assert reliability.total_decisions >= 3
    finally:
        set_decision_settings(test_settings)


def test_load_ticker_decisions_with_confidence_column(test_settings, tmp_path):
    """Analyzer must also work when decision_history DOES have 'confidence' column."""
    from app.decision import set_settings as set_decision_settings
    from app.core.decision.decision_history_analyzer import DecisionHistoryAnalyzer

    db_path = tmp_path / "with_conf.db"
    settings = replace(test_settings, DB_PATH=db_path)

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE decision_history (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            ticker TEXT,
            action_code INTEGER,
            confidence REAL,
            audit_data TEXT
        )
        """
    )
    now = datetime.now()
    for i in range(4):
        ts = (now - timedelta(days=i)).isoformat()
        audit = json.dumps({"strategy": "BB", "outcome": {"pnl_pct": 0.01}})
        conn.execute(
            "INSERT INTO decision_history (timestamp, ticker, action_code, confidence, audit_data) VALUES (?, ?, ?, ?, ?)",
            (ts, "SPY", 1, 0.8, audit),
        )
    conn.commit()
    conn.close()

    set_decision_settings(settings)
    try:
        analyzer = DecisionHistoryAnalyzer(settings=settings)
        decisions = analyzer._load_ticker_decisions("SPY", days=30)
        assert len(decisions) == 4
        # confidence should come from the DB column (0.8)
        assert all(abs(d["confidence"] - 0.8) < 0.01 for d in decisions)
    finally:
        set_decision_settings(test_settings)


# ---------------------------------------------------------------------------
# S15A – fetch_recent_outcomes
# ---------------------------------------------------------------------------


def test_load_recent_outcomes_delegates_to_dao(test_settings, tmp_path):
    """load_recent_outcomes returns [] gracefully when outcomes table is empty."""
    from app.backtesting.history_store import HistoryStore
    from app.infrastructure.repositories.sqlite_decision_repository import (
        SqliteDecisionRepository,
    )

    db_path = tmp_path / "outcomes.db"
    settings = replace(test_settings, DB_PATH=db_path)
    repo = SqliteDecisionRepository(settings)
    hs = HistoryStore(history_repo=repo)

    result = hs.load_recent_outcomes("VOO", n=3)
    assert isinstance(result, list)
    assert result == []


# ---------------------------------------------------------------------------
# S15B – LiveExecutionEngine
# ---------------------------------------------------------------------------


def test_live_execution_engine_noop_on_hold():
    """HOLD decisions are skipped without calling _send_order."""
    from app.services.execution_engines import LiveExecutionEngine

    engine = LiveExecutionEngine()
    send_calls = []

    def _patched_send(ticker, action, qty, limit_price, as_of):
        send_calls.append(ticker)
        return "order-123"

    engine._send_order = _patched_send

    from datetime import date

    engine.execute(
        [{"ticker": "VOO", "action": "HOLD", "qty": 0, "price": 400.0}],
        as_of=date.today(),
    )
    assert send_calls == []


def test_live_execution_engine_sends_buy():
    """BUY decision triggers _send_order."""
    from app.services.execution_engines import LiveExecutionEngine
    from datetime import date

    engine = LiveExecutionEngine()
    order_ids = []

    def _patched_send(ticker, action, qty, limit_price, as_of):
        order_ids.append(f"{ticker}:{action}:{qty}")
        return "order-001"

    engine._send_order = _patched_send

    engine.execute(
        [{"ticker": "AAPL", "action": "BUY", "qty": 10, "price": 200.0}],
        as_of=date.today(),
    )
    assert order_ids == ["AAPL:BUY:10"]


def test_live_execution_engine_not_implemented_by_default():
    """Default _send_order raises NotImplementedError; engine catches and logs."""
    from app.services.execution_engines import LiveExecutionEngine
    from datetime import date
    import logging

    mock_logger = MagicMock()
    engine = LiveExecutionEngine(logger=mock_logger)

    # Should NOT raise – exception is caught internally
    engine.execute(
        [{"ticker": "VOO", "action": "BUY", "qty": 5, "price": 400.0}],
        as_of=date.today(),
    )
    assert mock_logger.warning.called


def test_live_execution_engine_config_from_env(monkeypatch):
    """_config_from_env reads BROKER_* env vars."""
    from app.services.execution_engines import LiveExecutionEngine

    monkeypatch.setenv("BROKER_API_KEY", "testkey")
    monkeypatch.setenv("BROKER_PAPER", "false")

    cfg = LiveExecutionEngine._config_from_env()
    assert cfg.api_key == "testkey"
    assert cfg.paper is False


# ---------------------------------------------------------------------------
# S15B – PortfolioRebalancer + correlation limits wired into pipeline
# ---------------------------------------------------------------------------


def test_check_portfolio_drift_disabled_by_default(test_settings):
    """check_portfolio_drift returns should_rebalance=False when ENABLE_REBALANCER=false."""
    from app.services.trading_pipeline import TradingPipelineService
    from app.backtesting.history_store import HistoryStore

    hs = MagicMock(spec=HistoryStore)
    pipeline = TradingPipelineService(
        history_store=hs,
        settings=test_settings,
        data_fetcher=MagicMock(),
        model_runner=MagicMock(),
        email_notifier=MagicMock(),
        execution_engine=MagicMock(),
    )

    result = pipeline.check_portfolio_drift(
        current_positions={"VOO": 0.9, "SPY": 0.1},
        target_allocation={"VOO": 0.5, "SPY": 0.5},
        prices={"VOO": 400.0, "SPY": 450.0},
        total_value=10000.0,
    )
    assert result["should_rebalance"] is False
    assert result["trades"] == []


def test_check_portfolio_drift_enabled(test_settings):
    """check_portfolio_drift calls rebalancer when ENABLE_REBALANCER=true."""
    from app.services.trading_pipeline import TradingPipelineService
    from app.backtesting.history_store import HistoryStore

    settings = replace(test_settings, ENABLE_REBALANCER=True, REBALANCE_THRESHOLD=0.01)
    hs = MagicMock(spec=HistoryStore)
    pipeline = TradingPipelineService(
        history_store=hs,
        settings=settings,
        data_fetcher=MagicMock(),
        model_runner=MagicMock(),
        email_notifier=MagicMock(),
        execution_engine=MagicMock(),
    )

    # Extreme drift: 90% in VOO vs 50% target
    result = pipeline.check_portfolio_drift(
        current_positions={"VOO": 0.9, "SPY": 0.1},
        target_allocation={"VOO": 0.5, "SPY": 0.5},
        prices={"VOO": 400.0, "SPY": 450.0},
        total_value=10000.0,
    )
    assert result["should_rebalance"]


def test_allocate_capital_with_correlation_limits_disabled(test_settings):
    """When ENABLE_CORRELATION_LIMITS=false, allocate_capital still works."""
    from app.services.trading_pipeline import TradingPipelineService
    from app.backtesting.history_store import HistoryStore

    hs = MagicMock(spec=HistoryStore)
    pipeline = TradingPipelineService(
        history_store=hs,
        settings=test_settings,
        data_fetcher=MagicMock(),
        model_runner=MagicMock(),
        email_notifier=MagicMock(),
        execution_engine=MagicMock(),
    )

    candidates = [
        {
            "ticker": "VOO",
            "date": "2024-01-01",
            "decision": {"action_code": 1, "confidence": 0.8},
            "payload": {"volatility": 0.02},
        }
    ]
    result = pipeline.allocate_capital(candidates)
    assert len(result) == 1
    assert result[0].get("allocation_amount", 0) > 0


# ---------------------------------------------------------------------------
# S15B – ML ensemble _try_add_ml_vote
# ---------------------------------------------------------------------------


def test_try_add_ml_vote_skips_gracefully_when_no_model(test_settings, tmp_path):
    """_try_add_ml_vote does not raise when no trained model exists."""
    from app.core.decision.recommender import _try_add_ml_vote

    import pandas as pd
    import numpy as np

    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "Open": rng.random(100),
            "High": rng.random(100),
            "Low": rng.random(100),
            "Close": rng.random(100),
            "Volume": rng.random(100),
        }
    )

    votes: list = []
    confidences: list = []
    wf_scores: list = []
    model_votes: list = []

    model_dir = str(tmp_path)  # empty dir – no models

    _try_add_ml_vote(df, "VOO", model_dir, votes, confidences, wf_scores, model_votes)

    # Should remain unchanged
    assert votes == []
    assert model_votes == []


def test_enable_ml_ensemble_setting_exists(test_settings):
    """ENABLE_ML_ENSEMBLE setting is present and boolean typed."""
    assert hasattr(test_settings, "ENABLE_ML_ENSEMBLE")
    assert isinstance(test_settings.ENABLE_ML_ENSEMBLE, bool)


# ---------------------------------------------------------------------------
# S15B – new settings present in build_settings
# ---------------------------------------------------------------------------


def test_sprint15_settings_present(tmp_path, monkeypatch):
    """All Sprint 15 settings are present and keep defaults in isolated env."""
    from app.config.build_settings import build_settings

    empty_env = tmp_path / "empty.env"
    empty_env.write_text("", encoding="utf-8")
    # Ensure ambient shell env does not override the expected default check.
    monkeypatch.setenv("ENABLE_ML_ENSEMBLE", "false")
    s = build_settings(env_file=empty_env)
    assert hasattr(s, "ENABLE_REBALANCER")
    assert s.ENABLE_REBALANCER is False

    assert hasattr(s, "REBALANCE_THRESHOLD")
    assert abs(s.REBALANCE_THRESHOLD - 0.20) < 1e-6

    assert hasattr(s, "ENABLE_CORRELATION_LIMITS")
    assert s.ENABLE_CORRELATION_LIMITS is True

    assert hasattr(s, "MAX_CORRELATION")
    assert abs(s.MAX_CORRELATION - 0.70) < 1e-6

    assert hasattr(s, "ENABLE_ML_ENSEMBLE")
    assert s.ENABLE_ML_ENSEMBLE is False
