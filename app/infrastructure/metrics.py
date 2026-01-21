"""
System Metrics & Monitoring Module (P9 — Engineering Hardening)

Responsibility:
    - Track system health metrics (execution time, success rate, errors)
    - Log metrics for dashboard and alerting
    - Enable operational monitoring and SLA tracking

Features:
    - Pipeline execution logging (JSONL format)
    - Metrics aggregation (success rate, avg duration)
    - Recent metrics retrieval for admin dashboard
    - Performance trending

Usage:
    metrics = SystemMetrics()
    metrics.log_pipeline_execution("VOO", "success", 45.2)
    recent = metrics.get_recent_metrics(hours=24)
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import os

from app.config.config import Config
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)


class SystemMetrics:
    """
    Tracks and reports system health metrics.

    Stores metrics as JSONL (one JSON object per line) for easy parsing.
    """

    def __init__(self):
        """Initialize metrics tracking."""
        self.metrics_dir = Config.LOG_DIR / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def log_pipeline_execution(
        self, ticker: str, status: str, duration_sec: float, details: Dict = None
    ):
        """
        Log daily pipeline execution metrics.

        Args:
            ticker: Asset ticker
            status: "success", "error", "timeout", "skipped"
            duration_sec: Execution time in seconds
            details: Additional context (error message, trades count, etc.)
        """
        # TODO: Implement
        metric = {
            "timestamp": datetime.utcnow().isoformat(),
            "ticker": ticker,
            "status": status,
            "duration_sec": duration_sec,
            "details": details or {},
        }

        # Append to monthly JSONL file
        path = (
            self.metrics_dir / f"pipeline_{datetime.utcnow().strftime('%Y_%m')}.jsonl"
        )

        try:
            with open(path, "a") as f:
                f.write(json.dumps(metric) + "\n")
            logger.debug(f"Logged metric: {ticker} {status}")
        except Exception as e:
            logger.error(f"Failed to log metric: {e}")

    def log_backtest_execution(
        self,
        ticker: str,
        wf_score: float,
        trades_count: int,
        profit_factor: float,
        details: Dict = None,
    ):
        """
        Log backtest/walk-forward execution.

        Args:
            ticker: Asset ticker
            wf_score: Walk-forward fitness score (0-1)
            trades_count: Number of trades in backtest
            profit_factor: Gross profit / Gross loss
            details: Additional metrics
        """
        # TODO: Implement
        metric = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "backtest",
            "ticker": ticker,
            "wf_score": wf_score,
            "trades_count": trades_count,
            "profit_factor": profit_factor,
            "details": details or {},
        }

        path = (
            self.metrics_dir / f"backtest_{datetime.utcnow().strftime('%Y_%m')}.jsonl"
        )

        try:
            with open(path, "a") as f:
                f.write(json.dumps(metric) + "\n")
        except Exception as e:
            logger.error(f"Failed to log backtest metric: {e}")

    def get_recent_metrics(self, hours: int = 24) -> Dict:
        """
        Aggregate recent metrics for dashboard display.

        Args:
            hours: Number of hours to look back

        Returns:
            {
                "success_rate": float,        # 0-1
                "avg_duration_sec": float,
                "errors_count": int,
                "total_executions": int,
                "last_success": datetime or None,
                "last_error": datetime or None,
                "error_message": str or None
            }
        """
        # TODO: Implement
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        executions = []

        # Read metrics files
        for metrics_file in self.metrics_dir.glob("pipeline_*.jsonl"):
            try:
                with open(metrics_file, "r") as f:
                    for line in f:
                        try:
                            metric = json.loads(line)
                            ts = datetime.fromisoformat(metric["timestamp"])
                            if ts > cutoff_time:
                                executions.append(metric)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.warning(f"Error reading metrics file {metrics_file}: {e}")

        if not executions:
            return {
                "success_rate": 0.0,
                "avg_duration_sec": 0.0,
                "errors_count": 0,
                "total_executions": 0,
                "last_success": None,
                "last_error": None,
                "error_message": "No metrics available",
            }

        # Compute aggregates
        successes = [e for e in executions if e["status"] == "success"]
        errors = [e for e in executions if e["status"] == "error"]

        success_rate = len(successes) / len(executions) if executions else 0.0
        avg_duration = (
            np.mean([e["duration_sec"] for e in executions]) if executions else 0.0
        )

        last_success = max([e["timestamp"] for e in successes]) if successes else None
        last_error = max([e["timestamp"] for e in errors]) if errors else None
        last_error_msg = errors[-1].get("details", {}).get("error") if errors else None

        return {
            "success_rate": success_rate,
            "avg_duration_sec": round(avg_duration, 2),
            "errors_count": len(errors),
            "total_executions": len(executions),
            "last_success": last_success,
            "last_error": last_error,
            "error_message": last_error_msg,
        }

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
        # TODO: Implement
        return {
            "date": date,
            "executions": 0,
            "successes": 0,
            "failures": 0,
            "avg_duration_sec": 0.0,
            "tickers_processed": [],
        }


# Utility functions


def get_system_health() -> Dict[str, any]:
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
    # TODO: Implement
    metrics = SystemMetrics()
    recent = metrics.get_recent_metrics(hours=24)

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
        "error_rate": error_rate,
        "avg_response_time_sec": recent["avg_duration_sec"],
        "last_check": datetime.utcnow().isoformat(),
    }


import numpy as np
