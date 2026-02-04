"""
System Metrics & Monitoring Module (P9 — Engineering Hardening)

Responsibility:
    - Track system health metrics (execution time, success rate, errors)
    - Log metrics for dashboard and alerting
    - Enable operational monitoring and SLA tracking

Features:
    - Pipeline execution logging (via DataManager)
    - Metrics aggregation (success rate, avg duration)
    - Recent metrics retrieval for admin dashboard
    - Performance trending

Usage:
    metrics = get_metrics()
    metrics.log_pipeline_execution("VOO", "success", 45.2)
    recent = metrics.get_recent_metrics(hours=24)
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from app.config.config import Config
from app.data_access.data_manager import DataManager
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)


class SystemMetrics:
    """Tracks and reports system health metrics via DataManager."""

    def __init__(self):
        """Initialize metrics tracking."""
        self.dm = DataManager()
        # Ensure table exists during initialization
        self.dm.initialize_tables()

    def log_pipeline_execution(
        self,
        ticker: str,
        status: str,
        duration_sec: float,
        error_message: Optional[str] = None,
    ) -> bool:
        """
        Log daily pipeline execution to database.

        Args:
            ticker: Asset ticker
            status: "success", "error", "timeout", "skipped"
            duration_sec: Execution time in seconds
            error_message: Error details if status is "error"

        Returns:
            True if logged successfully, False otherwise
        """
        return self.dm.log_pipeline_execution(
            ticker, status, duration_sec, error_message
        )

    def log_backtest_execution(
        self,
        ticker: str,
        wf_score: float,
        trades_count: int,
        profit_factor: float,
        error_message: Optional[str] = None,
    ) -> bool:
        """
        Log backtest/walk-forward execution.

        Args:
            ticker: Asset ticker
            wf_score: Walk-forward fitness score (0-1)
            trades_count: Number of trades in backtest
            profit_factor: Gross profit / Gross loss
            error_message: Error if occurred

        Returns:
            True if logged successfully
        """
        return self.dm.log_backtest_execution(
            ticker, wf_score, trades_count, profit_factor, error_message
        )

    def get_recent_metrics(self, hours: int = 24) -> Dict:
        """
        Aggregate recent metrics for dashboard display.

        Args:
            hours: Number of hours to look back

        Returns:
            {
                "success_rate": float (0-1),
                "avg_duration_sec": float,
                "errors_count": int,
                "total_executions": int,
                "last_success": datetime or None,
                "last_error": datetime or None
            }
        """
        return self.dm.get_recent_metrics(hours)

    def get_daily_summary(self, date: str) -> Dict:
        """
        Get summary metrics for a specific date (YYYY-MM-DD format).

        Returns:
            {
                "date": str,
                "executions": int,
                "successes": int,
                "failures": int,
                "avg_duration_sec": float,
                "tickers_processed": list
            }
        """
        return self.dm.get_daily_summary(date)

    def get_health_status(self) -> Dict:
        """
        Get overall system health snapshot.

        Returns:
            {
                "status": "healthy" | "degraded" | "critical",
                "uptime_pct": float,
                "error_rate": float,
                "avg_response_time_sec": float,
                "last_check": datetime
            }
        """
        recent = self.get_recent_metrics(hours=24)
        error_rate = 1.0 - recent["success_rate"]

        if error_rate > 0.1:  # >10% errors = critical
            status = "critical"
        elif error_rate > 0.05:  # >5% errors = degraded
            status = "degraded"
        else:
            status = "healthy"

        return {
            "status": status,
            "uptime_pct": recent["success_rate"],
            "error_rate": round(error_rate, 3),
            "avg_response_time_sec": recent["avg_duration_sec"],
            "last_check": datetime.now(timezone.utc).isoformat(),
        }


# Singleton instance
_metrics_instance = None


def get_metrics() -> SystemMetrics:
    """Get or create SystemMetrics singleton."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = SystemMetrics()
    return _metrics_instance
