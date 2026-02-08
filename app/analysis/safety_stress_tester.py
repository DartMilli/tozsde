import json
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from app.data_access.data_manager import DataManager
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)


class SafetyStressTester:
    """
    Stress-tests safety rules by replaying historical decisions with
    synthetic stress scenarios. This module does not alter trading logic.
    """

    def __init__(self):
        self.dm = DataManager()

    def run(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        scenario: str = "elevated_volatility",
    ) -> Dict:
        decisions = self._load_decisions(ticker, start_date, end_date)
        if decisions.empty:
            result = {"status": "no_data", "scenario": scenario}
            self.dm.save_safety_stress_results(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                scenario=scenario,
                results_json=json.dumps(result),
            )
            return result

        ohlcv = self.dm.load_ohlcv(ticker)
        if ohlcv.empty:
            result = {"status": "no_market_data", "scenario": scenario}
            self.dm.save_safety_stress_results(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                scenario=scenario,
                results_json=json.dumps(result),
            )
            return result

        result = self._run_scenario(decisions, ohlcv, scenario)
        self.dm.save_safety_stress_results(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            scenario=scenario,
            results_json=json.dumps(result),
        )
        return result

    def _run_scenario(
        self, decisions: pd.DataFrame, ohlcv: pd.DataFrame, scenario: str
    ) -> Dict:
        decisions = decisions.copy()
        decisions["date"] = pd.to_datetime(decisions["timestamp"]).dt.date

        ohlcv = ohlcv.copy()
        ohlcv.index = pd.to_datetime(ohlcv.index)
        ohlcv["ret"] = ohlcv["Close"].pct_change()
        ohlcv["vol_20"] = ohlcv["ret"].rolling(20).std()

        merged = decisions.merge(
            ohlcv[["ret", "vol_20"]],
            left_on="date",
            right_index=True,
            how="left",
        )

        if scenario == "elevated_volatility":
            threshold = merged["vol_20"].median() * 2
            interventions = merged["vol_20"] > threshold
        elif scenario == "gap_days":
            interventions = merged["ret"].abs() > 0.05
        elif scenario == "drawdown_injection":
            # Artificially mark every 3rd decision as failed streak
            idx = np.arange(len(merged))
            interventions = (idx % 3) == 0
        else:
            interventions = pd.Series([False] * len(merged))

        result = {
            "scenario": scenario,
            "rows": int(len(merged)),
            "interventions": int(interventions.sum()),
            "intervention_rate": float(interventions.mean()) if len(merged) else 0.0,
        }
        return result

    def _load_decisions(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        query = """
            SELECT id, timestamp, action_code
            FROM decision_history
            WHERE ticker = ? AND date(timestamp) BETWEEN date(?) AND date(?)
        """
        with self.dm.connection() as conn:
            return pd.read_sql(query, conn, params=[ticker, start_date, end_date])
