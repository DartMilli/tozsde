"""
Tests for Decision History Analyzer (P8 — Learning System)

Tests strategy performance analysis, ticker reliability,
rolling metrics, and best/worst strategy identification.
"""

import pytest
import tempfile
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path

from app.decision.decision_history_analyzer import (
    DecisionHistoryAnalyzer,
    StrategyStats,
    TickerReliability
)
from app.config.config import Config


@pytest.fixture
def test_db():
    """Create temporary test database with decision history."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        Config.DB_PATH = db_path
        
        # Create decision_history table
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        
        cur.execute("""
            CREATE TABLE decision_history (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                ticker TEXT,
                action_code INTEGER,
                audit_data TEXT
            )
        """)
        
        # Insert sample decisions with outcomes
        now = datetime.now()
        
        # MA_CROSS strategy - good performance
        for i in range(10):
            date = (now - timedelta(days=i*3)).isoformat()
            pnl = 0.03 if i % 3 != 0 else -0.01  # 66% win rate
            audit = json.dumps({
                "strategy": "MA_CROSS",
                "confidence": 0.7,
                "outcome": {"pnl_pct": pnl, "evaluated_at": date}
            })
            cur.execute(
                "INSERT INTO decision_history (timestamp, ticker, action_code, audit_data) VALUES (?, ?, ?, ?)",
                (date, "VOO", 1, audit)
            )
        
        # RSI_MEAN strategy - poor performance
        for i in range(8):
            date = (now - timedelta(days=i*4)).isoformat()
            pnl = -0.02 if i % 3 == 0 else 0.01  # 37.5% win rate
            audit = json.dumps({
                "strategy": "RSI_MEAN",
                "confidence": 0.6,
                "outcome": {"pnl_pct": pnl, "evaluated_at": date}
            })
            cur.execute(
                "INSERT INTO decision_history (timestamp, ticker, action_code, audit_data) VALUES (?, ?, ?, ?)",
                (date, "MSFT", 1, audit)
            )
        
        # Decisions without outcomes (pending)
        for i in range(3):
            date = (now - timedelta(days=i)).isoformat()
            audit = json.dumps({"strategy": "MACD", "confidence": 0.5})
            cur.execute(
                "INSERT INTO decision_history (timestamp, ticker, action_code, audit_data) VALUES (?, ?, ?, ?)",
                (date, "AAPL", 1, audit)
            )
        
        conn.commit()
        conn.close()
        
        yield db_path


def test_strategy_performance_calculation(test_db):
    """Test strategy performance metrics calculation."""
    analyzer = DecisionHistoryAnalyzer()
    
    stats = analyzer.analyze_strategy_performance("MA_CROSS", days=90)
    
    assert stats.strategy_name == "MA_CROSS"
    assert stats.trades_analyzed == 10
    assert stats.win_rate >= 0.6  # Should be ~66%
    assert stats.avg_pnl > 0  # Positive average
    assert stats.status in ["EXCELLENT", "GOOD"]


def test_strategy_performance_no_data(test_db):
    """Test strategy with no decisions."""
    analyzer = DecisionHistoryAnalyzer()
    
    stats = analyzer.analyze_strategy_performance("NONEXISTENT", days=90)
    
    assert stats.strategy_name == "NONEXISTENT"
    assert stats.trades_analyzed == 0
    assert stats.win_rate == 0.0
    assert stats.status == "NO_DATA"


def test_ticker_reliability_analysis(test_db):
    """Test ticker-specific reliability analysis."""
    analyzer = DecisionHistoryAnalyzer()
    
    reliability = analyzer.analyze_ticker_reliability("VOO", days=60)
    
    assert reliability.ticker == "VOO"
    assert reliability.total_decisions >= 10
    assert reliability.success_rate >= 0.6  # MA_CROSS performance
    assert reliability.status in ["RELIABLE", "MODERATE"]


def test_ticker_reliability_no_data(test_db):
    """Test ticker with no decisions."""
    analyzer = DecisionHistoryAnalyzer()
    
    reliability = analyzer.analyze_ticker_reliability("UNKNOWN", days=60)
    
    assert reliability.ticker == "UNKNOWN"
    assert reliability.total_decisions == 0
    assert reliability.status == "NO_DATA"


def test_best_strategy_identification(test_db):
    """Test identification of best performing strategy."""
    analyzer = DecisionHistoryAnalyzer()
    
    best = analyzer.get_best_strategy(days=90, min_trades=5)
    
    # MA_CROSS should be best (66% win rate)
    assert best == "MA_CROSS"


def test_worst_strategy_identification(test_db):
    """Test identification of worst performing strategy."""
    analyzer = DecisionHistoryAnalyzer()
    
    worst = analyzer.get_worst_strategy(days=90, min_trades=5)
    
    # RSI_MEAN should be worst (37.5% win rate)
    assert worst == "RSI_MEAN"


def test_empty_history_handling(test_db):
    """Test handling of empty decision history."""
    # Clear the database
    conn = sqlite3.connect(str(test_db))
    cur = conn.cursor()
    cur.execute("DELETE FROM decision_history")
    conn.commit()
    conn.close()
    
    analyzer = DecisionHistoryAnalyzer()
    
    stats = analyzer.analyze_strategy_performance("ANY", days=90)
    assert stats.trades_analyzed == 0
    
    best = analyzer.get_best_strategy(days=90)
    assert best is None


def test_sharpe_ratio_calculation(test_db):
    """Test Sharpe ratio calculation."""
    analyzer = DecisionHistoryAnalyzer()
    
    stats = analyzer.analyze_strategy_performance("MA_CROSS", days=90)
    
    # With positive returns and reasonable variance, Sharpe should be positive
    assert stats.sharpe_ratio != 0.0  # Should calculate something


def test_max_drawdown_calculation(test_db):
    """Test maximum drawdown calculation."""
    analyzer = DecisionHistoryAnalyzer()
    
    stats = analyzer.analyze_strategy_performance("MA_CROSS", days=90)
    
    # Should have some drawdown from negative trades
    assert stats.max_drawdown >= 0.0
    assert stats.max_drawdown <= 1.0  # Can't be more than 100%


def test_confidence_calibration(test_db):
    """Test confidence score calibration analysis."""
    analyzer = DecisionHistoryAnalyzer()
    
    reliability = analyzer.analyze_ticker_reliability("VOO", days=60)
    
    # Calibration should be between 0 and 1
    assert 0.0 <= reliability.confidence_calibration <= 1.0
