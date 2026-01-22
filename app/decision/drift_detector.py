"""
Performance Drift Detection Module (P8 — Learning System)

Responsibility:
    - Track model performance across rolling windows
    - Detect when performance degrades significantly
    - Generate alerts and audit trail entries
    - Enable proactive risk management

Features:
    - Historical score tracking (last N days)
    - Drift percentage calculation
    - Alert level classification (NONE, WARNING, CRITICAL)
    - Integration with audit builder

Usage:
    detector = PerformanceDriftDetector(lookback_days=30)
    drift_info = detector.check_drift("VOO", current_wf_score=0.65)
    if drift_info["drifting"]:
        # Alert operator or take corrective action
        logger.warning(f"Drift detected: {drift_info}")
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import sqlite3

from app.config.config import Config
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)


class PerformanceDriftDetector:
    """
    Monitors model performance degradation across rolling windows.

    Attributes:
        lookback_days: Number of historical days to consider
        drift_threshold: Percentage drop to trigger warning (default 15%)
        critical_threshold: Percentage drop to trigger critical (default 30%)
    """

    def __init__(
        self,
        lookback_days: int = 30,
        drift_threshold: float = 0.15,
        critical_threshold: float = 0.30,
    ):
        """
        Initialize drift detector.

        Args:
            lookback_days: Days of history to analyze
            drift_threshold: Warning if performance drops >15%
            critical_threshold: Critical if performance drops >30%
        """
        self.lookback_days = lookback_days
        self.drift_threshold = drift_threshold
        self.critical_threshold = critical_threshold

    def check_drift(self, ticker: str, current_score: float) -> Dict:
        """
        Compare current performance vs. historical average.

        Args:
            ticker: Asset ticker symbol
            current_score: Current walk-forward score (0-1)

        Returns:
            {
                "drifting": bool,
                "drift_pct": float,  # Negative if degrading
                "alert_level": "NONE" | "WARNING" | "CRITICAL",
                "message": str,
                "historical_avg": float,
                "days_tracked": int
            }
        """
        historical_scores = self._load_historical_scores(ticker)

        if not historical_scores:
            return {
                "drifting": False,
                "drift_pct": 0.0,
                "alert_level": "NONE",
                "message": "Insufficient historical data",
                "historical_avg": 0.0,
                "days_tracked": 0,
            }

        # Compute metrics
        metrics = self._compute_drift_metrics(historical_scores, current_score)
        historical_avg = metrics.get("avg", 0.0)

        # Calculate drift percentage
        if historical_avg > 0.01:
            drift_pct = (historical_avg - current_score) / historical_avg
        else:
            drift_pct = 0.0

        # Classify alert level
        alert_level = "NONE"
        if drift_pct >= self.critical_threshold:
            alert_level = "CRITICAL"
        elif drift_pct >= self.drift_threshold:
            alert_level = "WARNING"

        drifting = alert_level != "NONE"

        # Log if drifting
        if drifting:
            logger.warning(
                f"{ticker}: Performance drift detected! "
                f"Avg={historical_avg:.3f}, Current={current_score:.3f}, "
                f"Drift={drift_pct:.1%} ({alert_level})"
            )

        return {
            "drifting": drifting,
            "drift_pct": drift_pct,
            "alert_level": alert_level,
            "message": (
                f"{alert_level}: {drift_pct:.1%} performance drop"
                if drifting
                else "No drift"
            ),
            "historical_avg": historical_avg,
            "days_tracked": len(historical_scores),
            "current_score": current_score,
        }

    def _load_historical_scores(self, ticker: str) -> List[float]:
        """
        Load historical performance scores for ticker from database.

        Returns:
            List of scores from last lookback_days, most recent first
        """
        scores = []
        try:
            conn = sqlite3.connect(str(Config.DB_PATH))
            cur = conn.cursor()

            # Query decision_history table for wf_score values
            query = """
                SELECT wf_score FROM decision_history 
                WHERE ticker = ? AND wf_score IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT ?
            """
            cur.execute(query, (ticker, self.lookback_days))
            rows = cur.fetchall()

            # Extract scores from query results
            scores = [float(row[0]) for row in rows if row[0] is not None]

            conn.close()
        except (sqlite3.Error, IOError, ValueError) as e:
            logger.warning(f"Failed to load history for {ticker}: {e}")

        return scores

    def _compute_drift_metrics(
        self, historical_scores: List[float], current_score: float
    ) -> Dict:
        """
        Compute drift statistics.

        Returns:
            {
                "avg": float,
                "std": float,
                "min": float,
                "max": float,
                "trend": "improving" | "degrading" | "stable"
            }
        """
        if not historical_scores:
            return {"avg": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "trend": "unknown"}

        # Calculate basic metrics
        avg = sum(historical_scores) / len(historical_scores)

        # Standard deviation
        variance = sum((x - avg) ** 2 for x in historical_scores) / len(
            historical_scores
        )
        std = variance**0.5

        min_score = min(historical_scores)
        max_score = max(historical_scores)

        # Determine trend: compare recent performance vs older performance
        midpoint = len(historical_scores) // 2
        recent_avg = (
            sum(historical_scores[:midpoint]) / midpoint if midpoint > 0 else avg
        )
        older_avg = (
            sum(historical_scores[midpoint:]) / (len(historical_scores) - midpoint)
            if len(historical_scores) > midpoint
            else avg
        )

        trend = "stable"
        if recent_avg > older_avg * 1.05:
            trend = "improving"
        elif recent_avg < older_avg * 0.95:
            trend = "degrading"

        return {
            "avg": avg,
            "std": std,
            "min": min_score,
            "max": max_score,
            "trend": trend,
        }

    def generate_alert(self, ticker: str, drift_info: Dict) -> Optional[str]:
        """
        Generate alert message if drift detected.

        Args:
            ticker: Asset ticker
            drift_info: Output from check_drift()

        Returns:
            Alert message string or None
        """
        if not drift_info.get("drifting", False):
            return None

        alert_level = drift_info.get("alert_level", "UNKNOWN")
        current_score = drift_info.get("current_score", 0)
        historical_avg = drift_info.get("historical_avg", 0)
        drift_pct = drift_info.get("drift_pct", 0)

        alert_template = (
            f"{alert_level}: {ticker} performance drift detected\n"
            f"Current score: {current_score:.3f}\n"
            f"Historical avg: {historical_avg:.3f}\n"
            f"Drift: {drift_pct:.1%}\n"
            f"Recommendation: Review parameters, consider retraining"
        )

        return alert_template


# Utility functions


def batch_check_drift(scores_dict: Dict[str, float]) -> Dict[str, Dict]:
    """
    Check drift for multiple tickers at once.

    Args:
        scores_dict: {ticker: current_score} mapping

    Returns:
        {ticker: drift_info_dict} with drifting status for each ticker
    """
    detector = PerformanceDriftDetector()
    results = {}

    for ticker, current_score in scores_dict.items():
        try:
            drift_info = detector.check_drift(ticker, current_score)
            results[ticker] = drift_info
        except Exception as e:
            logger.error(f"Drift check failed for {ticker}: {e}")
            results[ticker] = {"drifting": False, "alert_level": "ERROR"}

    return results


def get_drifting_tickers(
    scores_dict: Dict[str, float], alert_level: str = "WARNING"
) -> List[str]:
    """
    Get list of tickers with drift at or above alert_level.

    Args:
        scores_dict: {ticker: current_score} mapping
        alert_level: "WARNING" or "CRITICAL"

    Returns:
        List of ticker symbols showing drift
    """
    drift_results = batch_check_drift(scores_dict)

    alert_levels = ["CRITICAL", "WARNING"]  # Order of severity
    try:
        min_level_idx = alert_levels.index(alert_level)
    except ValueError:
        min_level_idx = len(alert_levels)  # Default to highest

    drifting = [
        ticker
        for ticker, info in drift_results.items()
        if info.get("drifting", False)
        and alert_levels.index(info.get("alert_level", "NONE")) <= min_level_idx
    ]

    return drifting
