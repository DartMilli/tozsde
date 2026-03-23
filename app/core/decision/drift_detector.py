"""Performance drift detection core logic."""

import sqlite3
from typing import Dict, List, Optional

from app.bootstrap.build_settings import build_settings
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)


class PerformanceDriftDetector:
    """Monitors model performance degradation across rolling windows."""

    def __init__(
        self,
        lookback_days: int = 30,
        drift_threshold: float = 0.15,
        critical_threshold: float = 0.30,
        settings=None,
    ):
        self.lookback_days = lookback_days
        cfg = settings or build_settings()
        self.drift_threshold = getattr(cfg, "DRIFT_THRESHOLD", drift_threshold)
        self.critical_threshold = getattr(
            cfg, "CRITICAL_DRIFT_THRESHOLD", critical_threshold
        )
        self.settings = settings

    def check_drift(self, ticker: str, current_score: float) -> Dict:
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

        metrics = self._compute_drift_metrics(historical_scores, current_score)
        historical_avg = metrics.get("avg", 0.0)

        if historical_avg > 0.01:
            drift_pct = (historical_avg - current_score) / historical_avg
        else:
            drift_pct = 0.0

        alert_level = "NONE"
        if drift_pct >= self.critical_threshold:
            alert_level = "CRITICAL"
        elif drift_pct >= self.drift_threshold:
            alert_level = "WARNING"

        drifting = alert_level != "NONE"

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
        scores: List[float] = []
        try:
            cfg = self.settings or build_settings()
            db_path = getattr(cfg, "DB_PATH", None)

            if db_path is None:
                raise ValueError("DB_PATH not configured")

            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()

            query = """
                SELECT wf_score FROM decision_history
                WHERE ticker = ? AND wf_score IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT ?
            """
            cur.execute(query, (ticker, self.lookback_days))
            rows = cur.fetchall()

            scores = [float(row[0]) for row in rows if row[0] is not None]
            conn.close()
        except (sqlite3.Error, IOError, ValueError) as e:
            logger.warning(f"Failed to load history for {ticker}: {e}")

        return scores

    def _compute_drift_metrics(
        self, historical_scores: List[float], current_score: float
    ) -> Dict:
        if not historical_scores:
            return {"avg": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "trend": "unknown"}

        avg = sum(historical_scores) / len(historical_scores)
        variance = sum((x - avg) ** 2 for x in historical_scores) / len(
            historical_scores
        )
        std = variance**0.5

        min_score = min(historical_scores)
        max_score = max(historical_scores)

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
        if not drift_info.get("drifting", False):
            return None

        alert_level = drift_info.get("alert_level", "UNKNOWN")
        current_score = drift_info.get("current_score", 0)
        historical_avg = drift_info.get("historical_avg", 0)
        drift_pct = drift_info.get("drift_pct", 0)

        return (
            f"{alert_level}: {ticker} performance drift detected\n"
            f"Current score: {current_score:.3f}\n"
            f"Historical avg: {historical_avg:.3f}\n"
            f"Drift: {drift_pct:.1%}\n"
            f"Recommendation: Review parameters, consider retraining"
        )


def batch_check_drift(scores_dict: Dict[str, float]) -> Dict[str, Dict]:
    detector = PerformanceDriftDetector()
    results: Dict[str, Dict] = {}

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
    drift_results = batch_check_drift(scores_dict)

    alert_levels = ["CRITICAL", "WARNING"]
    try:
        min_level_idx = alert_levels.index(alert_level)
    except ValueError:
        min_level_idx = len(alert_levels)

    return [
        ticker
        for ticker, info in drift_results.items()
        if info.get("drifting", False)
        and alert_levels.index(info.get("alert_level", "NONE")) <= min_level_idx
    ]
