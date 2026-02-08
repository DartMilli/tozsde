"""
No-Trade Decision Logger - Logs all no-trade decisions for audit and analysis.

This module tracks all decisions where the system chose not to trade, logging
the reasons for these decisions (low confidence, high correlation, etc.) to
enable post-trade analysis and decision quality assessment.

Classes:
    NoTradeReason: Enum for no-trade decision reasons
    NoTradeDecision: Dataclass for logged decision
    NoTradeDecisionLogger: Main logger class
"""

from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
import sqlite3
import logging

logger = logging.getLogger(__name__)


class NoTradeReason(Enum):
    """Enumeration of reasons for not trading."""

    LOW_CONFIDENCE = "LOW_CONFIDENCE"  # confidence_score < threshold
    HIGH_CORRELATION = (
        "HIGH_CORRELATION"  # correlation with existing positions too high
    )
    INSUFFICIENT_CAPITAL = "INSUFFICIENT_CAPITAL"  # not enough available capital
    POSITION_LIMIT_REACHED = "POSITION_LIMIT_REACHED"  # max positions already open
    MARKET_REGIMEN = "MARKET_REGIME"  # current market regime not favorable
    DIVERSIFICATION_CONSTRAINT = (
        "DIVERSIFICATION_CONSTRAINT"  # would violate diversification limits
    )
    RISK_THRESHOLD_EXCEEDED = (
        "RISK_THRESHOLD_EXCEEDED"  # portfolio risk already too high
    )
    DECISION_DISABLED = "DECISION_DISABLED"  # trading disabled in config
    OTHER = "OTHER"  # unspecified reason


@dataclass
class NoTradeDecision:
    """Record of a no-trade decision."""

    timestamp: datetime
    ticker: str
    strategy: str
    reason: NoTradeReason
    confidence_score: Optional[float] = None
    market_regime: Optional[str] = None
    correlation_value: Optional[float] = None
    available_capital: Optional[float] = None
    portfolio_risk: Optional[float] = None
    details: Optional[str] = None


class NoTradeDecisionLogger:
    """
    Logs all no-trade decisions for audit trail and analytics.

    Tracks:
    - Why a trade wasn't executed
    - Market conditions at time of decision
    - Risk metrics and constraints
    - Frequency analysis by reason
    """

    def __init__(self, db_path: str = None):
        """
        Initialize the no-trade decision logger.

        Args:
            db_path: Path to SQLite database for persistence
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database for no-trade decision tracking."""
        if not self.db_path:
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS no_trade_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    ticker TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    confidence_score REAL,
                    market_regime TEXT,
                    correlation_value REAL,
                    available_capital REAL,
                    portfolio_risk REAL,
                    details TEXT,
                    indexed_date DATE NOT NULL
                )
            """
            )

            # Create index for efficient date-based queries
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_no_trade_date 
                ON no_trade_decisions(indexed_date)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_no_trade_ticker 
                ON no_trade_decisions(ticker)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_no_trade_reason 
                ON no_trade_decisions(reason)
            """
            )

            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")

    def log_no_trade_decision(self, decision: NoTradeDecision) -> bool:
        """
        Log a no-trade decision.

        Args:
            decision: NoTradeDecision to log

        Returns:
            bool: True if successfully logged, False otherwise
        """
        if not self.db_path:
            logger.warning("No database path configured, decision not persisted")
            return False

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            indexed_date = decision.timestamp.date()

            cursor.execute(
                """
                INSERT INTO no_trade_decisions
                (timestamp, ticker, strategy, reason, confidence_score, market_regime,
                 correlation_value, available_capital, portfolio_risk, details, indexed_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    decision.timestamp,
                    decision.ticker,
                    decision.strategy,
                    decision.reason.value,
                    decision.confidence_score,
                    decision.market_regime,
                    decision.correlation_value,
                    decision.available_capital,
                    decision.portfolio_risk,
                    decision.details,
                    indexed_date,
                ),
            )

            conn.commit()
            conn.close()

            logger.info(
                f"Logged no-trade decision for {decision.ticker}: {decision.reason.value}"
            )
            return True
        except sqlite3.Error as e:
            logger.error(f"Error logging no-trade decision: {e}")
            return False

    def log_no_trade_simple(
        self, ticker: str, strategy: str, reason: NoTradeReason, **kwargs
    ) -> bool:
        """
        Log a no-trade decision with simplified interface.

        Args:
            ticker: Ticker symbol
            strategy: Strategy name
            reason: NoTradeReason enum
            **kwargs: Additional fields (confidence_score, market_regime, etc.)

        Returns:
            bool: True if successfully logged
        """
        decision = NoTradeDecision(
            timestamp=datetime.now(),
            ticker=ticker,
            strategy=strategy,
            reason=reason,
            confidence_score=kwargs.get("confidence_score"),
            market_regime=kwargs.get("market_regime"),
            correlation_value=kwargs.get("correlation_value"),
            available_capital=kwargs.get("available_capital"),
            portfolio_risk=kwargs.get("portfolio_risk"),
            details=kwargs.get("details"),
        )

        return self.log_no_trade_decision(decision)

    def get_no_trade_decisions(
        self,
        days_back: int = 7,
        ticker: Optional[str] = None,
        reason: Optional[NoTradeReason] = None,
    ) -> List[Dict]:
        """
        Retrieve no-trade decisions from database.

        Args:
            days_back: How many days back to retrieve
            ticker: Optional ticker to filter by
            reason: Optional reason to filter by

        Returns:
            List of no-trade decision records
        """
        if not self.db_path:
            return []

        records = []

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cutoff_date = (datetime.now() - timedelta(days=days_back)).date()

            query = """
                SELECT timestamp, ticker, strategy, reason, confidence_score,
                       market_regime, correlation_value, available_capital, portfolio_risk, details
                FROM no_trade_decisions
                WHERE indexed_date >= ?
            """
            params = [cutoff_date]

            if ticker:
                query += " AND ticker = ?"
                params.append(ticker)

            if reason:
                query += " AND reason = ?"
                params.append(reason.value)

            query += " ORDER BY timestamp DESC"

            cursor.execute(query, params)

            for row in cursor.fetchall():
                records.append(
                    {
                        "timestamp": row[0],
                        "ticker": row[1],
                        "strategy": row[2],
                        "reason": row[3],
                        "confidence_score": row[4],
                        "market_regime": row[5],
                        "correlation_value": row[6],
                        "available_capital": row[7],
                        "portfolio_risk": row[8],
                        "details": row[9],
                    }
                )

            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Error retrieving no-trade decisions: {e}")

        return records

    def get_no_trade_analysis(self, days_back: int = 30) -> Dict:
        """
        Get aggregated no-trade decision analysis.

        Args:
            days_back: Days to include in analysis

        Returns:
            Dict with aggregated statistics
        """
        if not self.db_path:
            return {}

        analysis = {
            "total_no_trades": 0,
            "by_reason": {},
            "by_ticker": {},
            "by_strategy": {},
            "avg_confidence_when_skipped": 0.0,
            "reason_details": {},
        }

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cutoff_date = (datetime.now() - timedelta(days=days_back)).date()

            # Total count
            cursor.execute(
                """
                SELECT COUNT(*) FROM no_trade_decisions
                WHERE indexed_date >= ?
            """,
                (cutoff_date,),
            )

            analysis["total_no_trades"] = cursor.fetchone()[0]

            # By reason
            cursor.execute(
                """
                SELECT reason, COUNT(*) FROM no_trade_decisions
                WHERE indexed_date >= ?
                GROUP BY reason
            """,
                (cutoff_date,),
            )

            for reason, count in cursor.fetchall():
                analysis["by_reason"][reason] = count

            # By ticker
            cursor.execute(
                """
                SELECT ticker, COUNT(*) FROM no_trade_decisions
                WHERE indexed_date >= ?
                GROUP BY ticker
                ORDER BY COUNT(*) DESC
                LIMIT 10
            """,
                (cutoff_date,),
            )

            for ticker, count in cursor.fetchall():
                analysis["by_ticker"][ticker] = count

            # By strategy
            cursor.execute(
                """
                SELECT strategy, COUNT(*) FROM no_trade_decisions
                WHERE indexed_date >= ?
                GROUP BY strategy
            """,
                (cutoff_date,),
            )

            for strategy, count in cursor.fetchall():
                analysis["by_strategy"][strategy] = count

            # Average confidence when skipped
            cursor.execute(
                """
                SELECT AVG(confidence_score) FROM no_trade_decisions
                WHERE indexed_date >= ? AND confidence_score IS NOT NULL
            """,
                (cutoff_date,),
            )

            avg_conf = cursor.fetchone()[0]
            if avg_conf is not None:
                analysis["avg_confidence_when_skipped"] = avg_conf

            # Detailed reason analysis
            for reason in NoTradeReason:
                cursor.execute(
                    """
                    SELECT COUNT(*), AVG(confidence_score), AVG(portfolio_risk)
                    FROM no_trade_decisions
                    WHERE indexed_date >= ? AND reason = ?
                """,
                    (cutoff_date, reason.value),
                )

                result = cursor.fetchone()
                if result:
                    analysis["reason_details"][reason.value] = {
                        "count": result[0],
                        "avg_confidence": result[1],
                        "avg_portfolio_risk": result[2],
                    }

            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Error performing no-trade analysis: {e}")

        return analysis

    def export_no_trade_decisions(self, output_file: str, days_back: int = 30) -> bool:
        """
        Args:
            output_file: Path to output JSON file
            days_back: Days to include

        Returns:
            bool: True if successfully exported
        """
        if not self.db_path:
            return False

        try:
            rows = self.get_no_trade_decisions(days_back=days_back)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2, default=str)
            logger.info(f"Exported {len(rows)} no-trade decisions to {output_file}")
            return True
        except (IOError, OSError, sqlite3.Error) as e:
            logger.error(f"Error exporting no-trade decisions: {e}")
            return False

    def clear_old_decisions(self, days_to_keep: int = 90) -> int:
        """
        Clear no-trade decisions older than specified days.

        Args:
            days_to_keep: Number of days to retain

        Returns:
            int: Number of records deleted
        """
        if not self.db_path:
            return 0

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).date()

            cursor.execute(
                """
                DELETE FROM no_trade_decisions
                WHERE indexed_date < ?
            """,
                (cutoff_date,),
            )

            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()

            logger.info(
                f"Deleted {deleted_count} no-trade decisions older than {days_to_keep} days"
            )
            return deleted_count
        except sqlite3.Error as e:
            logger.error(f"Error deleting old decisions: {e}")
            return 0
