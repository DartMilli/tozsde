import json
from datetime import datetime
from typing import Dict

import numpy as np

from app.data_access.data_manager import DataManager
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)


class WalkForwardStabilityAnalyzer:
    """
    Computes stability metrics from stored walk-forward results.
    """

    def __init__(self):
        self.dm = DataManager()

    def analyze(self, ticker: str) -> Dict:
        results = self._load_results(ticker)
        if not results:
            metrics = {"status": "no_data", "ticker": ticker}
            self.dm.save_wf_stability_metrics(
                ticker=ticker, metrics_json=json.dumps(metrics)
            )
            return metrics

        params_list = []
        for r in results:
            params = r.get("best_params") or {}
            if params:
                params_list.append(params)

        param_variance = {}
        if params_list:
            keys = params_list[0].keys()
            for k in keys:
                vals = [p.get(k) for p in params_list if p.get(k) is not None]
                if vals:
                    param_variance[k] = float(np.var(vals))

        wf_scores = [r.get("wf_fitness") for r in results if r.get("wf_fitness")]
        consistency = float(np.std(wf_scores)) if wf_scores else None

        metrics = {
            "ticker": ticker,
            "runs": len(results),
            "param_variance": param_variance,
            "wf_score_std": consistency,
        }

        self.dm.save_wf_stability_metrics(
            ticker=ticker, metrics_json=json.dumps(metrics)
        )
        return metrics

    def _load_results(self, ticker: str):
        query = """
            SELECT result_json
            FROM walk_forward_results
            WHERE ticker = ?
            ORDER BY computed_at ASC
        """
        with self.dm.connection() as conn:
            rows = conn.execute(query, (ticker,)).fetchall()
        out = []
        for (raw,) in rows:
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        return out
