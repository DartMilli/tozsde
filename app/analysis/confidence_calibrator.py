import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from app.config.config import Config
from app.data_access.data_manager import DataManager
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class CalibrationResult:
    method: str
    params: Dict
    metrics: Dict


class ConfidenceCalibrator:
    """
    Computes and applies confidence calibration using historical decisions.
    """

    def __init__(self):
        self.dm = DataManager()

    def compute(
        self,
        ticker: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        as_of_date: Optional[str] = None,
        method: str = "isotonic",
        n_bins: int = 10,
    ) -> CalibrationResult:
        df = self._load_data(ticker, start_date, end_date, as_of_date)
        if df.empty:
            metrics = {"status": "no_data"}
            self.dm.save_confidence_calibration(
                ticker=ticker or "ALL",
                start_date=start_date or "",
                end_date=end_date or "",
                method=method,
                params_json=json.dumps({}),
                metrics_json=json.dumps(metrics),
            )
            return CalibrationResult(method=method, params={}, metrics=metrics)

        x = df["confidence"].clip(0, 1).values
        y = df["success"].values.astype(int)

        if method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(x, y)
            calibrated = iso.transform(x)
            params = {
                "x_thresholds": iso.X_thresholds_.tolist(),
                "y_thresholds": iso.y_thresholds_.tolist(),
            }
        else:
            raise ValueError("Unsupported method")

        metrics = self._compute_metrics(x, y, calibrated, n_bins=n_bins)

        self.dm.save_confidence_calibration(
            ticker=ticker or "ALL",
            start_date=start_date or "",
            end_date=end_date or "",
            method=method,
            params_json=json.dumps(params),
            metrics_json=json.dumps(metrics),
        )

        return CalibrationResult(method=method, params=params, metrics=metrics)

    def apply(self, raw_confidence: float, params: Dict) -> float:
        if not Config.ENABLE_CONFIDENCE_CALIBRATION:
            return raw_confidence

        x_thresh = params.get("x_thresholds")
        y_thresh = params.get("y_thresholds")
        if not x_thresh or not y_thresh:
            return raw_confidence

        return float(np.interp(raw_confidence, x_thresh, y_thresh))

    def load_latest_params(
        self,
        ticker: str,
        as_of_date: Optional[str] = None,
    ) -> Dict:
        row = self.dm.fetch_latest_confidence_calibration(
            ticker=ticker,
            as_of_date=as_of_date,
        )
        if not row:
            return {}
        raw = row.get("params_json")
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def _load_data(
        self,
        ticker: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        as_of_date: Optional[str],
    ) -> pd.DataFrame:
        query = """
            SELECT dh.confidence, o.success
            FROM decision_history dh
            JOIN outcomes o ON o.decision_id = dh.id
            WHERE 1=1
        """
        params = []
        if ticker:
            query += " AND dh.ticker = ?"
            params.append(ticker)
        if start_date:
            query += " AND date(dh.timestamp) >= date(?)"
            params.append(start_date)
        if end_date:
            query += " AND date(dh.timestamp) <= date(?)"
            params.append(end_date)
        if as_of_date:
            query += " AND date(dh.timestamp) <= date(?)"
            params.append(as_of_date)

        with self.dm.connection() as conn:
            return pd.read_sql(query, conn, params=params)

    def _compute_metrics(
        self,
        raw: np.ndarray,
        y_true: np.ndarray,
        calibrated: np.ndarray,
        n_bins: int,
    ) -> Dict:
        # Expected Calibration Error (ECE)
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (raw >= bins[i]) & (raw < bins[i + 1])
            if not mask.any():
                continue
            acc = y_true[mask].mean()
            conf = raw[mask].mean()
            ece += np.abs(acc - conf) * (mask.sum() / len(raw))

        # Brier score
        brier = float(np.mean((calibrated - y_true) ** 2))

        return {
            "ece": round(float(ece), 5),
            "brier": round(brier, 5),
            "count": int(len(raw)),
        }
