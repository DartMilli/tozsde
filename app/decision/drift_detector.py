from typing import Dict, List

from app.core.decision.drift_detector import PerformanceDriftDetector  # noqa: F401
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)


def batch_check_drift(scores_dict: Dict[str, float]) -> Dict[str, Dict]:
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
    drift_results = batch_check_drift(scores_dict)

    alert_levels = ["CRITICAL", "WARNING"]
    try:
        min_level_idx = alert_levels.index(alert_level)
    except ValueError:
        min_level_idx = len(alert_levels)

    drifting = [
        ticker
        for ticker, info in drift_results.items()
        if info.get("drifting", False)
        and alert_levels.index(info.get("alert_level", "NONE")) <= min_level_idx
    ]

    return drifting
