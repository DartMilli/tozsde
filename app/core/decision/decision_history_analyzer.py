"""
Decision History Analyzer (P8 - Learning System)

Responsibility:
    - Analyze past trading decisions and their outcomes
    - Compute strategy-level and ticker-level performance metrics
    - Identify best/worst performing strategies
    - Provide rolling performance analytics
    - Support adaptive strategy selection

Features:
    - Strategy performance aggregation (win rate, avg PnL, Sharpe)
    - Ticker reliability analysis
    - Rolling metrics computation (30-day windows)
    - Time-series performance tracking
    - Best/worst strategy identification

Usage:
    analyzer = DecisionHistoryAnalyzer()
    stats = analyzer.analyze_strategy_performance("MA_CROSS", days=90)
    print(f"Win rate: {stats.win_rate:.1%}, Sharpe: {stats.sharpe_ratio:.2f}")

    # Get best strategy for current period
    best = analyzer.get_best_strategy(days=30)
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import sqlite3
import json
import pandas as pd

from app.bootstrap.build_settings import build_settings
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class StrategyStats:
    """Performance statistics for a trading strategy."""

    strategy_name: str
    total_trades: int
    win_rate: float  # 0.0 to 1.0
    avg_pnl: float  # Average profit/loss percentage
    sharpe_ratio: float
    max_drawdown: float
    last_30d_performance: float  # Recent performance (win rate)
    status: str  # "EXCELLENT" | "GOOD" | "DECLINING" | "POOR"
    trades_analyzed: int  # Number of completed trades with outcomes


@dataclass
class TickerReliability:
    """Reliability metrics for a specific ticker."""

    ticker: str
    total_decisions: int
    successful_trades: int
    failed_trades: int
    success_rate: float
    avg_return: float
    confidence_calibration: float  # How well confidence predicts success
    trend_accuracy: float  # How often trend predictions were correct
    status: str  # "RELIABLE" | "MODERATE" | "UNRELIABLE"


class DecisionHistoryAnalyzer:
    """
    Analyzes historical trading decisions and outcomes.

    Provides strategy-level and ticker-level performance analytics
    to support adaptive decision-making and strategy selection.
    """

    def __init__(self, settings=None):
        """Initialize the analyzer."""
        self.settings = settings
        cfg = settings or build_settings()

        self.db_path = None
        if cfg is not None:
            self.db_path = getattr(cfg, "DB_PATH", None)

    def analyze_strategy_performance(
        self, strategy_name: str, days: int = 90
    ) -> StrategyStats:
        """
        Compute comprehensive performance statistics for a strategy.

        Args:
            strategy_name: Name of the strategy to analyze
            days: Lookback period in days

        Returns:
            StrategyStats object with complete performance metrics
        """
        # Load historical decisions for this strategy
        decisions = self._load_strategy_decisions(strategy_name, days)

        if not decisions:
            logger.info(f"No decisions found for strategy {strategy_name}")
            return self._empty_strategy_stats(strategy_name)

        # Filter only decisions with outcome data
        completed = [d for d in decisions if d.get("outcome") is not None]

        if not completed:
            logger.info(f"No completed trades for strategy {strategy_name}")
            return self._empty_strategy_stats(strategy_name)

        # Extract PnL values
        pnl_values = [d["outcome"]["pnl_pct"] for d in completed]

        # Compute metrics
        total_trades = len(completed)
        wins = sum(1 for pnl in pnl_values if pnl > 0)
        win_rate = wins / total_trades if total_trades > 0 else 0.0
        avg_pnl = sum(pnl_values) / total_trades if total_trades > 0 else 0.0

        # Sharpe ratio (simplified: mean / std of returns)
        sharpe_ratio = self._calculate_sharpe(pnl_values)

        # Max drawdown
        max_drawdown = self._calculate_max_drawdown(pnl_values)

        # Recent performance (last 30 days)
        recent_decisions = [d for d in completed if self._is_recent(d, days=30)]
        recent_wins = sum(1 for d in recent_decisions if d["outcome"]["pnl_pct"] > 0)
        last_30d_performance = (
            recent_wins / len(recent_decisions) if recent_decisions else 0.0
        )

        # Classify status
        status = self._classify_strategy_status(
            win_rate, sharpe_ratio, last_30d_performance
        )

        return StrategyStats(
            strategy_name=strategy_name,
            total_trades=len(decisions),
            win_rate=win_rate,
            avg_pnl=avg_pnl,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            last_30d_performance=last_30d_performance,
            status=status,
            trades_analyzed=total_trades,
        )

    def analyze_ticker_reliability(
        self, ticker: str, days: int = 60
    ) -> TickerReliability:
        """
        Analyze how reliable predictions are for a specific ticker.

        Args:
            ticker: Ticker symbol to analyze
            days: Lookback period

        Returns:
            TickerReliability object with ticker-specific metrics
        """
        decisions = self._load_ticker_decisions(ticker, days)

        if not decisions:
            return self._empty_ticker_reliability(ticker)

        # Filter completed trades
        completed = [d for d in decisions if d.get("outcome") is not None]

        if not completed:
            return self._empty_ticker_reliability(ticker)

        # Success metrics
        successful = sum(1 for d in completed if d["outcome"]["pnl_pct"] > 0)
        failed = len(completed) - successful
        success_rate = successful / len(completed) if completed else 0.0

        # Average return
        avg_return = sum(d["outcome"]["pnl_pct"] for d in completed) / len(completed)

        # Confidence calibration: correlation between confidence and success
        confidence_calibration = self._calculate_confidence_calibration(completed)

        # Trend accuracy: placeholder (would need trend prediction data)
        trend_accuracy = 0.5  # Default neutral value

        # Status classification
        status = (
            "RELIABLE"
            if success_rate > 0.55
            else "MODERATE" if success_rate > 0.45 else "UNRELIABLE"
        )

        return TickerReliability(
            ticker=ticker,
            total_decisions=len(decisions),
            successful_trades=successful,
            failed_trades=failed,
            success_rate=success_rate,
            avg_return=avg_return,
            confidence_calibration=confidence_calibration,
            trend_accuracy=trend_accuracy,
            status=status,
        )

    def compute_rolling_metrics(
        self, window_days: int = 30, total_days: int = 180
    ) -> pd.DataFrame:
        """
        Compute rolling performance metrics over time.

        Args:
            window_days: Size of rolling window
            total_days: Total lookback period

        Returns:
            DataFrame with columns: date, rolling_sharpe, rolling_win_rate, rolling_avg_pnl
        """
        decisions = self._load_all_decisions(days=total_days)

        if not decisions:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(decisions)
        df["date"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("date")

        # Filter only completed trades
        df = df[df["outcome"].notna()]
        df["pnl"] = df["outcome"].apply(lambda x: x.get("pnl_pct", 0) if x else 0)
        df["win"] = df["pnl"] > 0

        # Compute rolling metrics
        df["rolling_win_rate"] = (
            df["win"].rolling(window=window_days, min_periods=5).mean()
        )
        df["rolling_avg_pnl"] = (
            df["pnl"].rolling(window=window_days, min_periods=5).mean()
        )
        df["rolling_sharpe"] = (
            df["pnl"]
            .rolling(window=window_days, min_periods=5)
            .apply(lambda x: x.mean() / x.std() if x.std() > 0 else 0)
        )

        # Resample to daily
        result = df.set_index("date").resample("D").last()
        result = result[
            ["rolling_sharpe", "rolling_win_rate", "rolling_avg_pnl"]
        ].dropna()

        return result

    def get_best_strategy(self, days: int = 30, min_trades: int = 5) -> Optional[str]:
        """
        Identify the best performing strategy in recent period.

        Args:
            days: Recent period to analyze
            min_trades: Minimum trades required for consideration

        Returns:
            Strategy name or None if insufficient data
        """
        strategies = self._get_all_strategies()

        best_strategy = None
        best_score = -float("inf")

        for strategy in strategies:
            if strategy == "ALL":
                continue
            stats = self.analyze_strategy_performance(strategy, days=days)

            if stats.trades_analyzed < min_trades or stats.trades_analyzed == 0:
                continue

            # Composite score: win_rate * sharpe_ratio
            score = stats.win_rate * max(stats.sharpe_ratio, 0)

            if score > best_score:
                best_score = score
                best_strategy = strategy

        return best_strategy

    def get_worst_strategy(self, days: int = 30, min_trades: int = 5) -> Optional[str]:
        """
        Identify the worst performing strategy in recent period.

        Args:
            days: Recent period to analyze
            min_trades: Minimum trades required

        Returns:
            Strategy name or None
        """
        strategies = self._get_all_strategies()

        worst_strategy = None
        worst_score = float("inf")

        for strategy in strategies:
            if strategy == "ALL":
                continue
            stats = self.analyze_strategy_performance(strategy, days=days)

            if stats.trades_analyzed < min_trades or stats.trades_analyzed == 0:
                continue

            score = stats.win_rate * max(stats.sharpe_ratio, 0)

            if score < worst_score:
                worst_score = score
                worst_strategy = strategy

        return worst_strategy

    # Helper methods

    def _load_strategy_decisions(self, strategy_name: str, days: int) -> List[Dict]:
        """Load decisions for a specific strategy."""
        try:
            db_path = self._resolve_db_path()
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()

            cutoff_date = datetime.now() - timedelta(days=days)

            # Detect schema: whether audit stored in `audit_blob` or `audit_data`
            cur.execute("PRAGMA table_info(decision_history)")
            cols = [r[1] for r in cur.fetchall()]
            audit_col = (
                "audit_blob"
                if "audit_blob" in cols
                else ("audit_data" if "audit_data" in cols else "audit_blob")
            )

            # Check if outcomes table exists
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='outcomes'"
            )
            outcomes_exists = cur.fetchone() is not None

            if outcomes_exists:
                query = f"""
                    SELECT dh.timestamp, dh.ticker, dh.{audit_col}, o.pnl_pct
                    FROM decision_history dh
                    LEFT JOIN outcomes o ON o.decision_id = dh.id
                    WHERE dh.timestamp >= ? AND dh.action_code = 1
                """
            else:
                query = f"""
                    SELECT dh.timestamp, dh.ticker, dh.{audit_col} as audit_blob, NULL as pnl_pct
                    FROM decision_history dh
                    WHERE dh.timestamp >= ? AND dh.action_code = 1
                """

            try:
                cur.execute(query, (cutoff_date.isoformat(),))
                rows = cur.fetchall()
            except sqlite3.OperationalError:
                conn.close()
                raise

            conn.close()

            decisions = []
            for timestamp, ticker, audit_blob, pnl_pct in rows:
                try:
                    audit = json.loads(audit_blob) if audit_blob else {}
                    # Filter by strategy name (if stored in audit data)
                    if audit.get("strategy") == strategy_name or strategy_name == "ALL":
                        outcome = None
                        if pnl_pct is not None:
                            outcome = {"pnl_pct": pnl_pct}
                        else:
                            audit_outcome = audit.get("outcome")
                            if isinstance(audit_outcome, dict):
                                pnl_val = audit_outcome.get("pnl_pct")
                                if pnl_val is not None:
                                    outcome = {"pnl_pct": pnl_val}
                        decisions.append(
                            {
                                "timestamp": timestamp,
                                "ticker": ticker,
                                "strategy": audit.get("strategy", "UNKNOWN"),
                                "outcome": outcome,
                            }
                        )
                except json.JSONDecodeError:
                    continue

            return decisions

        except Exception as e:
            logger.error(f"Failed to load strategy decisions: {e}")
            return []

    def _load_ticker_decisions(self, ticker: str, days: int) -> List[Dict]:
        """Load decisions for a specific ticker."""
        try:
            db_path = self._resolve_db_path()
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()

            cutoff_date = datetime.now() - timedelta(days=days)

            cur.execute("PRAGMA table_info(decision_history)")
            cols = [r[1] for r in cur.fetchall()]
            audit_col = (
                "audit_blob"
                if "audit_blob" in cols
                else ("audit_data" if "audit_data" in cols else "audit_blob")
            )

            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='outcomes'"
            )
            outcomes_exists = cur.fetchone() is not None

            if outcomes_exists:
                query = f"""
                    SELECT dh.timestamp, dh.{audit_col}, dh.confidence, o.pnl_pct
                    FROM decision_history dh
                    LEFT JOIN outcomes o ON o.decision_id = dh.id
                    WHERE dh.ticker = ? AND dh.timestamp >= ? AND dh.action_code = 1
                """
            else:
                query = f"""
                    SELECT dh.timestamp, dh.{audit_col} as audit_blob, NULL as confidence, NULL as pnl_pct
                    FROM decision_history dh
                    WHERE dh.ticker = ? AND dh.timestamp >= ? AND dh.action_code = 1
                """

            try:
                cur.execute(query, (ticker, cutoff_date.isoformat()))
                rows = cur.fetchall()
            except sqlite3.OperationalError:
                conn.close()
                raise

            conn.close()

            decisions = []
            for timestamp, audit_blob, confidence, pnl_pct in rows:
                try:
                    audit = json.loads(audit_blob) if audit_blob else {}
                    outcome = None
                    if pnl_pct is not None:
                        outcome = {"pnl_pct": pnl_pct}
                    else:
                        audit_outcome = audit.get("outcome")
                        if isinstance(audit_outcome, dict):
                            pnl_val = audit_outcome.get("pnl_pct")
                            if pnl_val is not None:
                                outcome = {"pnl_pct": pnl_val}
                    if confidence is None:
                        confidence = audit.get("confidence", 0.5)
                    decisions.append(
                        {
                            "timestamp": timestamp,
                            "ticker": ticker,
                            "outcome": outcome,
                            "confidence": confidence if confidence is not None else 0.5,
                        }
                    )
                except json.JSONDecodeError:
                    continue

            return decisions

        except Exception as e:
            logger.error(f"Failed to load ticker decisions: {e}")
            return []

    def _load_all_decisions(self, days: int) -> List[Dict]:
        """Load all decisions in lookback period."""
        try:
            db_path = self._resolve_db_path()
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()

            cutoff_date = datetime.now() - timedelta(days=days)

            cur.execute("PRAGMA table_info(decision_history)")
            cols = [r[1] for r in cur.fetchall()]
            audit_col = (
                "audit_blob"
                if "audit_blob" in cols
                else ("audit_data" if "audit_data" in cols else "audit_blob")
            )

            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='outcomes'"
            )
            outcomes_exists = cur.fetchone() is not None

            if outcomes_exists:
                query = f"""
                    SELECT dh.timestamp, dh.ticker, dh.{audit_col}, o.pnl_pct
                    FROM decision_history dh
                    LEFT JOIN outcomes o ON o.decision_id = dh.id
                    WHERE dh.timestamp >= ? AND dh.action_code = 1
                """
            else:
                query = f"""
                    SELECT dh.timestamp, dh.ticker, dh.{audit_col} as audit_blob, NULL as pnl_pct
                    FROM decision_history dh
                    WHERE dh.timestamp >= ? AND dh.action_code = 1
                """

            try:
                cur.execute(query, (cutoff_date.isoformat(),))
                rows = cur.fetchall()
            except sqlite3.OperationalError:
                conn.close()
                raise

            conn.close()

            decisions = []
            for timestamp, ticker, audit_blob, pnl_pct in rows:
                try:
                    audit = json.loads(audit_blob) if audit_blob else {}
                    outcome = None
                    if pnl_pct is not None:
                        outcome = {"pnl_pct": pnl_pct}
                    else:
                        audit_outcome = audit.get("outcome")
                        if isinstance(audit_outcome, dict):
                            pnl_val = audit_outcome.get("pnl_pct")
                            if pnl_val is not None:
                                outcome = {"pnl_pct": pnl_val}
                    decisions.append(
                        {
                            "timestamp": timestamp,
                            "ticker": ticker,
                            "outcome": outcome,
                        }
                    )
                except json.JSONDecodeError:
                    continue

            return decisions

        except Exception as e:
            logger.error(f"Failed to load all decisions: {e}")
            return []

    def _resolve_db_path(self):
        if self.settings is not None and self.db_path:
            return self.db_path
        cfg = self.settings
        if cfg is None:
            try:
                cfg = get_settings()
            except Exception:
                cfg = build_settings()
        db_path = getattr(cfg, "DB_PATH", None) if cfg is not None else None
        if db_path is not None:
            return db_path
        return self.db_path

    def _get_all_strategies(self) -> List[str]:
        """Get list of all strategies used."""
        # Placeholder: return common strategy names
        # In production, this would query the database
        return ["MA_CROSS", "RSI_MEAN", "MACD", "BB_MEAN", "ALL"]

    def _calculate_sharpe(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio from returns."""
        if not returns or len(returns) < 2:
            return 0.0

        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std = variance**0.5

        if std < 1e-6:
            return 0.0

        # Annualized Sharpe (assuming ~252 trading days)
        sharpe = (mean_return / std) * (252**0.5)
        return sharpe

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns."""
        if not returns:
            return 0.0

        cumulative = [1.0]
        for ret in returns:
            cumulative.append(cumulative[-1] * (1 + ret))

        max_dd = 0.0
        peak = cumulative[0]

        for value in cumulative:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def _is_recent(self, decision: Dict, days: int) -> bool:
        """Check if decision is within recent days."""
        try:
            decision_date = pd.to_datetime(decision["timestamp"])
            cutoff = datetime.now() - timedelta(days=days)
            return decision_date >= cutoff
        except:
            return False

    def _classify_strategy_status(
        self, win_rate: float, sharpe: float, recent_performance: float
    ) -> str:
        """Classify strategy status based on metrics."""
        if win_rate >= 0.9 and recent_performance >= 0.8:
            return "EXCELLENT"
        if win_rate > 0.6 and sharpe > 1.0 and recent_performance > 0.55:
            return "EXCELLENT"
        elif win_rate > 0.5 and sharpe > 0.5:
            return "GOOD"
        elif recent_performance < win_rate * 0.8:
            return "DECLINING"
        else:
            return "POOR"

    def _calculate_confidence_calibration(self, decisions: List[Dict]) -> float:
        """
        Calculate how well confidence scores predict actual outcomes.

        Returns value between 0 (no correlation) and 1 (perfect correlation).
        """
        if len(decisions) < 5:
            return 0.5  # Not enough data

        # Group by confidence buckets
        high_conf = [d for d in decisions if d.get("confidence", 0.5) > 0.7]
        low_conf = [d for d in decisions if d.get("confidence", 0.5) < 0.4]

        if not high_conf or not low_conf:
            return 0.5

        high_success = sum(1 for d in high_conf if d["outcome"]["pnl_pct"] > 0) / len(
            high_conf
        )
        low_success = sum(1 for d in low_conf if d["outcome"]["pnl_pct"] > 0) / len(
            low_conf
        )

        # Calibration: high confidence should lead to higher success
        calibration = (high_success - low_success + 1) / 2  # Normalize to 0-1
        return max(0.0, min(1.0, calibration))

    def _empty_strategy_stats(self, strategy_name: str) -> StrategyStats:
        """Return empty stats for strategy with no data."""
        return StrategyStats(
            strategy_name=strategy_name,
            total_trades=0,
            win_rate=0.0,
            avg_pnl=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            last_30d_performance=0.0,
            status="NO_DATA",
            trades_analyzed=0,
        )

    def _empty_ticker_reliability(self, ticker: str) -> TickerReliability:
        """Return empty reliability for ticker with no data."""
        return TickerReliability(
            ticker=ticker,
            total_decisions=0,
            successful_trades=0,
            failed_trades=0,
            success_rate=0.0,
            avg_return=0.0,
            confidence_calibration=0.5,
            trend_accuracy=0.5,
            status="NO_DATA",
        )
