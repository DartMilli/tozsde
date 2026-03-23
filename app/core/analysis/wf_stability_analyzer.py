import json
from typing import Dict

import numpy as np

from app.bootstrap.build_settings import build_settings
from app.infrastructure.logger import setup_logger
from app.infrastructure.repositories import DataManagerRepository

logger = setup_logger(__name__)


class WalkForwardStabilityAnalyzer:
    """
    Computes stability metrics from stored walk-forward results.
    """

    def __init__(self, settings=None, data_manager=None):
        self.settings = settings or build_settings()
        self.dm = data_manager or DataManagerRepository(settings=self.settings)

    def analyze(self, ticker: str) -> Dict:
        results = self._load_results(ticker)
        if not results:
            metrics = {"status": "no_data", "ticker": ticker}
            self.dm.save_wf_stability_metrics(
                ticker=ticker, metrics_json=json.dumps(metrics)
            )
            return metrics

        params_list = []
        for result in results:
            params = result.get("best_params") or {}
            if params:
                params_list.append(params)

        param_variance = {}
        if params_list:
            keys = params_list[0].keys()
            for key in keys:
                vals = [p.get(key) for p in params_list if p.get(key) is not None]
                if vals:
                    param_variance[key] = float(np.var(vals))

        wf_scores = []
        for result in results:
            score = result.get("raw_fitness")
            if score is None:
                score = result.get("wf_fitness")
            if score is not None:
                wf_scores.append(score)
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
            SELECT result_json, computed_at
            FROM walk_forward_results
            WHERE ticker = ?
            ORDER BY computed_at ASC
        """
        with self.dm.connection() as conn:
            rows = conn.execute(query, (ticker,)).fetchall()
        out = []
        for raw, computed_at in rows:
            try:
                payload = json.loads(raw)
                payload["computed_at"] = computed_at
                out.append(payload)
            except json.JSONDecodeError:
                continue
        if getattr(self.settings, "AGGREGATION_MODE", "") == "latest_only" and out:
            latest = max(out, key=lambda result: result.get("computed_at") or "")
            run_id = latest.get("wf_run_id")
            if run_id:
                out = [result for result in out if result.get("wf_run_id") == run_id]
            else:
                out = [latest]
        return out
