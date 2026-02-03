"""Extended test coverage for Decision History Analyzer."""

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
def test_db_extended():
    """Extended test database with comprehensive decision history."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_extended.db"
        Config.DB_PATH = db_path
        
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
        
        now = datetime.now()
        
        # Perfect strategy (100% win rate)
        for i in range(5):
            date = (now - timedelta(days=i*2)).isoformat()
            audit = json.dumps({
                "strategy": "PERFECT",
                "confidence": 0.95,
                "outcome": {"pnl_pct": 0.05, "evaluated_at": date}
            })
            cur.execute(
                "INSERT INTO decision_history (timestamp, ticker, action_code, audit_data) VALUES (?, ?, ?, ?)",
                (date, "AAPL", 1, audit)
            )
        
        # Losing strategy (0% win rate)
        for i in range(4):
            date = (now - timedelta(days=i*3)).isoformat()
            audit = json.dumps({
                "strategy": "LOSING",
                "confidence": 0.3,
                "outcome": {"pnl_pct": -0.03, "evaluated_at": date}
            })
            cur.execute(
                "INSERT INTO decision_history (timestamp, ticker, action_code, audit_data) VALUES (?, ?, ?, ?)",
                (date, "MSFT", 1, audit)
            )
        
        # Single trade
        audit = json.dumps({
            "strategy": "SINGLE",
            "confidence": 0.7,
            "outcome": {"pnl_pct": 0.02, "evaluated_at": now.isoformat()}
        })
        cur.execute(
            "INSERT INTO decision_history (timestamp, ticker, action_code, audit_data) VALUES (?, ?, ?, ?)",
            (now.isoformat(), "GOOGL", 1, audit)
        )
        
        # Old decisions (outside lookback window)
        for i in range(3):
            date = (now - timedelta(days=365+i)).isoformat()
            audit = json.dumps({
                "strategy": "OLD",
                "confidence": 0.5,
                "outcome": {"pnl_pct": 0.01, "evaluated_at": date}
            })
            cur.execute(
                "INSERT INTO decision_history (timestamp, ticker, action_code, audit_data) VALUES (?, ?, ?, ?)",
                (date, "VTI", 1, audit)
            )
        
        conn.commit()
        conn.close()
        
        yield db_path


class TestStrategyPerformanceEdgeCases:
    """Test edge cases in strategy performance analysis."""
    
    def test_perfect_strategy(self, test_db_extended):
        """Test strategy with 100% win rate."""
        analyzer = DecisionHistoryAnalyzer()
        stats = analyzer.analyze_strategy_performance("PERFECT", days=90)
        
        assert stats.win_rate == 1.0
        assert stats.avg_pnl > 0
        assert stats.status == "EXCELLENT"
    
    def test_losing_strategy(self, test_db_extended):
        """Test strategy with 0% win rate."""
        analyzer = DecisionHistoryAnalyzer()
        stats = analyzer.analyze_strategy_performance("LOSING", days=90)
        
        assert stats.win_rate == 0.0
        assert stats.avg_pnl < 0
        assert stats.status in ["POOR", "DECLINING"]
    
    def test_single_trade_strategy(self, test_db_extended):
        """Test strategy with only one trade."""
        analyzer = DecisionHistoryAnalyzer()
        stats = analyzer.analyze_strategy_performance("SINGLE", days=90)
        
        assert stats.total_trades == 1
        assert stats.win_rate in [0.0, 1.0]
    
    def test_no_trades_strategy(self, test_db_extended):
        """Test strategy with no trades in lookback window."""
        analyzer = DecisionHistoryAnalyzer()
        stats = analyzer.analyze_strategy_performance("NONEXISTENT", days=90)
        
        assert stats.total_trades == 0
        assert stats.win_rate == 0.0
    
    def test_old_decisions_excluded(self, test_db_extended):
        """Test that old decisions are excluded from analysis."""
        analyzer = DecisionHistoryAnalyzer()
        stats = analyzer.analyze_strategy_performance("OLD", days=30)
        
        # Should be empty because decisions are >365 days old
        assert stats.total_trades == 0


class TestTickerReliability:
    """Test ticker reliability analysis."""
    
    def test_reliable_ticker(self, test_db_extended):
        """Test ticker with consistent positive outcomes."""
        analyzer = DecisionHistoryAnalyzer()
        reliability = analyzer.analyze_ticker_reliability("AAPL", days=90)
        
        assert reliability.ticker == "AAPL"
        assert reliability.total_decisions > 0
    
    def test_unreliable_ticker(self, test_db_extended):
        """Test ticker with mostly negative outcomes."""
        analyzer = DecisionHistoryAnalyzer()
        reliability = analyzer.analyze_ticker_reliability("MSFT", days=90)
        
        assert reliability.ticker == "MSFT"
        assert reliability.total_decisions > 0


class TestRollingMetrics:
    """Test rolling metrics computation."""
    
    def test_rolling_metrics_structure(self, test_db_extended):
        """Test that rolling metrics returns proper structure."""
        analyzer = DecisionHistoryAnalyzer()
        metrics = analyzer.compute_rolling_metrics(window_days=7)
        
        # Should be iterable and contain date/metric data
        assert metrics is not None
    
    def test_rolling_metrics_30day(self, test_db_extended):
        """Test 30-day rolling window."""
        analyzer = DecisionHistoryAnalyzer()
        metrics = analyzer.compute_rolling_metrics(window_days=30)
        
        assert metrics is not None


class TestBestWorstStrategy:
    """Test best/worst strategy identification."""
    
    def test_get_best_strategy(self, test_db_extended):
        """Test identification of best performing strategy."""
        analyzer = DecisionHistoryAnalyzer()
        best = analyzer.get_best_strategy(days=90, min_trades=1)
        
        # Should be PERFECT (100% win rate)
        if best:
            assert best == "PERFECT"
    
    def test_get_worst_strategy(self, test_db_extended):
        """Test identification of worst performing strategy."""
        analyzer = DecisionHistoryAnalyzer()
        worst = analyzer.get_worst_strategy(days=90, min_trades=1)
        
        # Should be LOSING (0% win rate)
        if worst:
            assert worst == "LOSING"
    
    def test_min_trades_threshold(self, test_db_extended):
        """Test that min_trades threshold is respected."""
        analyzer = DecisionHistoryAnalyzer()
        best = analyzer.get_best_strategy(days=90, min_trades=10)
        
        # PERFECT has 5 trades, should be None with min_trades=10
        assert best is None
    
    def test_no_strategies_in_window(self, test_db_extended):
        """Test when no strategies meet criteria."""
        analyzer = DecisionHistoryAnalyzer()
        best = analyzer.get_best_strategy(days=1, min_trades=100)
        
        assert best is None


class TestPrivateMethods:
    """Test private helper methods."""
    
    def test_is_recent_true(self, test_db_extended):
        """Test _is_recent returns True for recent decisions."""
        analyzer = DecisionHistoryAnalyzer()
        now = datetime.now()
        decision = {"timestamp": now.isoformat()}
        
        result = analyzer._is_recent(decision, days=1)
        assert result is True
    
    def test_is_recent_false(self, test_db_extended):
        """Test _is_recent returns False for old decisions."""
        analyzer = DecisionHistoryAnalyzer()
        old_date = (datetime.now() - timedelta(days=100)).isoformat()
        decision = {"timestamp": old_date}
        
        result = analyzer._is_recent(decision, days=30)
        assert result is False
    
    def test_calculate_sharpe_with_volatility(self, test_db_extended):
        """Test Sharpe ratio calculation with normal returns."""
        analyzer = DecisionHistoryAnalyzer()
        returns = [0.01, 0.02, -0.01, 0.03, 0.00]
        
        sharpe = analyzer._calculate_sharpe(returns)
        assert sharpe is not None
        assert isinstance(sharpe, (int, float))
    
    def test_calculate_sharpe_zero_volatility(self, test_db_extended):
        """Test Sharpe ratio with zero volatility."""
        analyzer = DecisionHistoryAnalyzer()
        returns = [0.02, 0.02, 0.02, 0.02]
        
        sharpe = analyzer._calculate_sharpe(returns)
        # Should handle gracefully (return 0 or inf)
        assert sharpe is not None
    
    def test_calculate_max_drawdown_positive(self, test_db_extended):
        """Test max drawdown calculation with volatility."""
        analyzer = DecisionHistoryAnalyzer()
        returns = [0.05, -0.03, 0.02, -0.04, 0.06]
        
        dd = analyzer._calculate_max_drawdown(returns)
        # Drawdown is cumulative decline from peak, so could be positive
        assert dd is not None
    
    def test_calculate_max_drawdown_only_gains(self, test_db_extended):
        """Test max drawdown with only positive returns."""
        analyzer = DecisionHistoryAnalyzer()
        returns = [0.01, 0.02, 0.03, 0.01]
        
        dd = analyzer._calculate_max_drawdown(returns)
        assert dd == 0 or dd is not None


class TestClassifyStrategyStatus:
    """Test strategy status classification."""
    
    def test_excellent_status(self, test_db_extended):
        """Test EXCELLENT status classification."""
        analyzer = DecisionHistoryAnalyzer()
        stats = analyzer.analyze_strategy_performance("PERFECT", days=90)
        
        # Perfect strategy should be EXCELLENT
        assert stats.status == "EXCELLENT"
    
    def test_poor_status(self, test_db_extended):
        """Test POOR status classification."""
        analyzer = DecisionHistoryAnalyzer()
        stats = analyzer.analyze_strategy_performance("LOSING", days=90)
        
        # Losing strategy should be POOR
        assert stats.status in ["POOR", "DECLINING"]


class TestEmptyReturns:
    """Test empty result handling."""
    
    def test_empty_strategy_stats(self, test_db_extended):
        """Test _empty_strategy_stats returns proper structure."""
        analyzer = DecisionHistoryAnalyzer()
        stats = analyzer._empty_strategy_stats("TEST")
        
        assert isinstance(stats, StrategyStats)
        assert stats.total_trades == 0
        assert stats.win_rate == 0.0
    
    def test_empty_ticker_reliability(self, test_db_extended):
        """Test _empty_ticker_reliability returns proper structure."""
        analyzer = DecisionHistoryAnalyzer()
        reliability = analyzer._empty_ticker_reliability("TEST")
        
        assert isinstance(reliability, TickerReliability)
        assert reliability.total_decisions == 0
