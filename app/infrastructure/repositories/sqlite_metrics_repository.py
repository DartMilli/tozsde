from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from app.config.build_settings import build_settings
from app.core.ports.imetrics_repository import IMetricsRepository
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)


class SqliteMetricsRepository(IMetricsRepository):
    """SQLite-backed metrics repository.

    This repository owns SQL for the pipeline metrics domain and no longer
    delegates to DataManager for these operations.
    """

    def __init__(self, settings=None):
        self.settings = settings
        cfg = settings or build_settings()
        db_path = getattr(cfg, "DB_PATH", None)
        if not db_path:
            raise RuntimeError("DB_PATH is not configured")
        self.db_path = str(db_path)

    def _get_conn(self):
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return sqlite3.connect(self.db_path)

    def initialize_tables(self) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pipeline_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    ticker TEXT NOT NULL,
                    status TEXT NOT NULL,
                    duration_sec REAL,
                    error_message TEXT,
                    execution_date DATE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_metrics_date ON pipeline_metrics(execution_date)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_metrics_ticker ON pipeline_metrics(ticker)"
            )
            conn.commit()

    def log_pipeline_execution(
        self, ticker: str, status: str, duration_sec: float, error_message: str = None
    ) -> bool:
        try:
            with self._get_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO pipeline_metrics
                    (ticker, status, duration_sec, error_message, execution_date)
                    VALUES (?, ?, ?, ?, DATE('now'))
                    """,
                    (ticker, status, duration_sec, error_message),
                )
                conn.commit()
            return True
        except Exception as exc:
            logger.error(f"Failed to log pipeline execution: {exc}")
            return False

    def log_backtest_execution(
        self,
        ticker: str,
        wf_score: float,
        trades_count: int,
        profit_factor: float,
        error_message: str = None,
    ) -> bool:
        try:
            with self._get_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO pipeline_metrics
                    (ticker, status, duration_sec, error_message, execution_date)
                    VALUES (?, ?, ?, ?, DATE('now'))
                    """,
                    (
                        ticker,
                        "success" if error_message is None else "error",
                        0.0,
                        error_message
                        or f"WF={wf_score:.3f} PF={profit_factor:.2f} trades={trades_count}",
                    ),
                )
                conn.commit()
            return True
        except Exception as exc:
            logger.error(f"Failed to log backtest execution: {exc}")
            return False

    def get_recent_metrics(self, hours: int = 24) -> dict:
        try:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    """
                    SELECT status, COUNT(*) as count, AVG(duration_sec) as avg_duration
                    FROM pipeline_metrics
                    WHERE timestamp >= datetime('now', '-' || ? || ' hours')
                    GROUP BY status
                    """,
                    (hours,),
                )

                results = cursor.fetchall()
                total = sum(r[1] for r in results)
                success_count = next((r[1] for r in results if r[0] == "success"), 0)
                error_count = next((r[1] for r in results if r[0] == "error"), 0)
                avg_duration = (
                    next((r[2] for r in results if r[0] == "success"), 0) or 0
                )

                success_rate = success_count / total if total > 0 else 0

                cursor = conn.execute(
                    """
                    SELECT timestamp FROM pipeline_metrics
                    WHERE status = 'success'
                    ORDER BY timestamp DESC LIMIT 1
                    """
                )
                last_success = cursor.fetchone()

                cursor = conn.execute(
                    """
                    SELECT timestamp FROM pipeline_metrics
                    WHERE status = 'error'
                    ORDER BY timestamp DESC LIMIT 1
                    """
                )
                last_error = cursor.fetchone()

                return {
                    "success_rate": round(success_rate, 3),
                    "avg_duration_sec": round(avg_duration, 2),
                    "errors_count": error_count,
                    "total_executions": total,
                    "last_success": last_success[0] if last_success else None,
                    "last_error": last_error[0] if last_error else None,
                }
        except Exception as exc:
            logger.error(f"Failed to get recent metrics: {exc}")
            return {
                "success_rate": 0.0,
                "avg_duration_sec": 0.0,
                "errors_count": 0,
                "total_executions": 0,
                "last_success": None,
                "last_error": None,
            }

    def get_daily_summary(self, date: str) -> dict:
        try:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    """
                    SELECT COUNT(*), SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END),
                           SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END),
                           AVG(duration_sec)
                    FROM pipeline_metrics
                    WHERE execution_date = ?
                    """,
                    (date,),
                )

                row = cursor.fetchone()
                total, successes, failures, avg_duration = (
                    row[0] or 0,
                    row[1] or 0,
                    row[2] or 0,
                    row[3] or 0.0,
                )

                cursor = conn.execute(
                    """
                    SELECT DISTINCT ticker FROM pipeline_metrics
                    WHERE execution_date = ?
                    ORDER BY ticker
                    """,
                    (date,),
                )
                tickers = [result_row[0] for result_row in cursor.fetchall()]

                return {
                    "date": date,
                    "executions": total,
                    "successes": successes,
                    "failures": failures,
                    "avg_duration_sec": round(avg_duration, 2),
                    "tickers_processed": tickers,
                }
        except Exception as exc:
            logger.error(f"Failed to get daily summary for {date}: {exc}")
            return {
                "date": date,
                "executions": 0,
                "successes": 0,
                "failures": 0,
                "avg_duration_sec": 0.0,
                "tickers_processed": [],
            }

    # Protocol compatibility methods retained during migration window.
    def save_metrics(self, metrics: dict) -> None:
        self.log_pipeline_execution(
            ticker=metrics.get("ticker", "UNKNOWN"),
            status=metrics.get("status", "success"),
            duration_sec=float(metrics.get("duration_sec", 0.0) or 0.0),
            error_message=metrics.get("error_message"),
        )

    def fetch_metrics(self, ticker: str, start_date: str, end_date: str) -> list[dict]:
        query = """
            SELECT ticker, status, duration_sec, error_message, execution_date, timestamp
            FROM pipeline_metrics
            WHERE ticker = ? AND execution_date >= ? AND execution_date <= ?
            ORDER BY timestamp ASC
        """
        try:
            with self._get_conn() as conn:
                frame = pd.read_sql(query, conn, params=[ticker, start_date, end_date])
            return frame.to_dict("records") if not frame.empty else []
        except Exception:
            return []
