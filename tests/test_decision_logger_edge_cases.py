"""
Edge case tests for NoTradeDecisionLogger.
"""

import os
import json
import tempfile
from datetime import datetime, timedelta

from app.infrastructure.decision_logger import (
    NoTradeDecisionLogger,
    NoTradeDecision,
    NoTradeReason,
)


def _create_logger_with_db():
    tmpdir = tempfile.TemporaryDirectory()
    db_path = os.path.join(tmpdir.name, "test.db")
    logger = NoTradeDecisionLogger(db_path=db_path)
    return logger, tmpdir


def test_log_no_trade_decision_persists():
    logger, tmpdir = _create_logger_with_db()
    try:
        decision = NoTradeDecision(
            timestamp=datetime.now(),
            ticker="AAPL",
            strategy="momentum",
            reason=NoTradeReason.LOW_CONFIDENCE,
            confidence_score=0.2,
            market_regime="BEAR",
            correlation_value=0.9,
            available_capital=1000.0,
            portfolio_risk=0.3,
            details="test",
        )

        assert logger.log_no_trade_decision(decision) is True

        records = logger.get_no_trade_decisions(days_back=365)
        assert len(records) == 1
        assert records[0]["ticker"] == "AAPL"
        assert records[0]["reason"] == NoTradeReason.LOW_CONFIDENCE.value
    finally:
        tmpdir.cleanup()


def test_log_no_trade_simple():
    logger, tmpdir = _create_logger_with_db()
    try:
        result = logger.log_no_trade_simple(
            ticker="MSFT",
            strategy="mean_reversion",
            reason=NoTradeReason.HIGH_CORRELATION,
            confidence_score=0.4,
        )

        assert result is True
        records = logger.get_no_trade_decisions(days_back=7)
        assert len(records) == 1
        assert records[0]["reason"] == NoTradeReason.HIGH_CORRELATION.value
    finally:
        tmpdir.cleanup()


def test_get_no_trade_decisions_filters():
    logger, tmpdir = _create_logger_with_db()
    try:
        now = datetime.now()
        decisions = [
            NoTradeDecision(
                timestamp=now,
                ticker="AAPL",
                strategy="s1",
                reason=NoTradeReason.LOW_CONFIDENCE,
            ),
            NoTradeDecision(
                timestamp=now,
                ticker="MSFT",
                strategy="s2",
                reason=NoTradeReason.HIGH_CORRELATION,
            ),
        ]

        for d in decisions:
            logger.log_no_trade_decision(d)

        by_ticker = logger.get_no_trade_decisions(days_back=7, ticker="AAPL")
        assert len(by_ticker) == 1
        assert by_ticker[0]["ticker"] == "AAPL"

        by_reason = logger.get_no_trade_decisions(
            days_back=7, reason=NoTradeReason.HIGH_CORRELATION
        )
        assert len(by_reason) == 1
        assert by_reason[0]["reason"] == NoTradeReason.HIGH_CORRELATION.value
    finally:
        tmpdir.cleanup()


def test_get_no_trade_analysis_aggregates():
    logger, tmpdir = _create_logger_with_db()
    try:
        now = datetime.now()
        decisions = [
            NoTradeDecision(
                timestamp=now,
                ticker="AAPL",
                strategy="s1",
                reason=NoTradeReason.LOW_CONFIDENCE,
                confidence_score=0.2,
                portfolio_risk=0.5,
            ),
            NoTradeDecision(
                timestamp=now,
                ticker="AAPL",
                strategy="s1",
                reason=NoTradeReason.LOW_CONFIDENCE,
                confidence_score=0.3,
                portfolio_risk=0.4,
            ),
            NoTradeDecision(
                timestamp=now,
                ticker="MSFT",
                strategy="s2",
                reason=NoTradeReason.HIGH_CORRELATION,
                confidence_score=0.6,
                portfolio_risk=0.7,
            ),
        ]
        for d in decisions:
            logger.log_no_trade_decision(d)

        analysis = logger.get_no_trade_analysis(days_back=7)

        assert analysis["total_no_trades"] == 3
        assert analysis["by_reason"][NoTradeReason.LOW_CONFIDENCE.value] == 2
        assert analysis["by_ticker"]["AAPL"] == 2
        assert analysis["by_strategy"]["s1"] == 2
        assert analysis["avg_confidence_when_skipped"] > 0
        assert NoTradeReason.LOW_CONFIDENCE.value in analysis["reason_details"]
    finally:
        tmpdir.cleanup()


def test_export_no_trade_decisions_json():
    logger, tmpdir = _create_logger_with_db()
    try:
        decision = NoTradeDecision(
            timestamp=datetime.now(),
            ticker="AAPL",
            strategy="s1",
            reason=NoTradeReason.INSUFFICIENT_CAPITAL,
        )
        logger.log_no_trade_decision(decision)

        output_file = os.path.join(tmpdir.name, "export.json")
        assert logger.export_no_trade_decisions(output_file, days_back=7) is True

        with open(output_file, "r") as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["ticker"] == "AAPL"
    finally:
        tmpdir.cleanup()


def test_clear_old_decisions_deletes():
    logger, tmpdir = _create_logger_with_db()
    try:
        old_date = datetime.now() - timedelta(days=120)
        recent_date = datetime.now() - timedelta(days=2)

        old_decision = NoTradeDecision(
            timestamp=old_date,
            ticker="OLD",
            strategy="s1",
            reason=NoTradeReason.OTHER,
        )
        recent_decision = NoTradeDecision(
            timestamp=recent_date,
            ticker="NEW",
            strategy="s1",
            reason=NoTradeReason.OTHER,
        )

        logger.log_no_trade_decision(old_decision)
        logger.log_no_trade_decision(recent_decision)

        deleted = logger.clear_old_decisions(days_to_keep=30)
        assert deleted == 1

        remaining = logger.get_no_trade_decisions(days_back=365)
        assert len(remaining) == 1
        assert remaining[0]["ticker"] == "NEW"
    finally:
        tmpdir.cleanup()


def test_no_db_path_returns_defaults():
    logger = NoTradeDecisionLogger(db_path=None)

    assert logger.log_no_trade_simple(
        ticker="AAPL",
        strategy="s1",
        reason=NoTradeReason.LOW_CONFIDENCE,
    ) is False

    assert logger.get_no_trade_decisions() == []
    assert logger.get_no_trade_analysis() == {}
    assert logger.clear_old_decisions() == 0
