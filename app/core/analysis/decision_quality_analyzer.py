import json
from typing import Dict, Optional

import pandas as pd

from app.bootstrap.build_settings import build_settings
from app.infrastructure.repositories import DataManagerRepository


class DecisionQualityAnalyzer:
    """
    Computes decision quality metrics from decision_history and outcomes.
    """

    def __init__(self, settings=None, data_manager=None):
        self.settings = settings or build_settings()
        self.dm = data_manager or DataManagerRepository(settings=self.settings)

    def analyze(
        self,
        ticker: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        buckets: int = 10,
    ) -> Dict:
        df = self._load_data(ticker=ticker, start_date=start_date, end_date=end_date)
        if df.empty:
            metrics = {
                "status": "no_data",
                "scope": {
                    "ticker": ticker or "ALL",
                    "start_date": start_date,
                    "end_date": end_date,
                    "rows": 0,
                },
            }
            self.dm.save_decision_quality_metrics(
                ticker=ticker or "ALL",
                start_date=start_date or "",
                end_date=end_date or "",
                metrics_json=json.dumps(metrics),
            )
            return metrics

        df = df.copy()
        df["confidence"] = df["confidence"].fillna(0.0).clip(0, 1)
        df["success"] = df["success"].fillna(0).astype(int)

        df["conf_bucket"] = pd.qcut(
            df["confidence"], q=buckets, labels=False, duplicates="drop"
        )

        bucket_stats = (
            df.groupby("conf_bucket")
            .agg(
                count=("id", "count"),
                hit_rate=("success", "mean"),
                avg_return=("pnl_pct", "mean"),
            )
            .reset_index()
        )

        total_buys = (df["action_code"] == 1).sum()
        total_non_buys = (df["action_code"] != 1).sum()

        fp = ((df["action_code"] == 1) & (df["success"] == 0)).sum()
        fn = ((df["action_code"] != 1) & (df["success"] == 1)).sum()

        fp_rate = fp / total_buys if total_buys else 0.0
        fn_rate = fn / total_non_buys if total_non_buys else 0.0

        override_flag = df["safety_override"] == 1
        with_override = df[override_flag]
        without_override = df[~override_flag]

        metrics = {
            "scope": {
                "ticker": ticker or "ALL",
                "start_date": start_date,
                "end_date": end_date,
                "rows": int(len(df)),
            },
            "bucket_stats": bucket_stats.to_dict(orient="records"),
            "false_positive_rate": round(fp_rate, 4),
            "false_negative_rate": round(fn_rate, 4),
            "decision_frequency": {
                "buy": int(total_buys),
                "non_buy": int(total_non_buys),
            },
            "safety_override_impact": {
                "with_override_count": int(len(with_override)),
                "without_override_count": int(len(without_override)),
                "with_override_hit_rate": (
                    float(with_override["success"].mean())
                    if len(with_override)
                    else None
                ),
                "without_override_hit_rate": (
                    float(without_override["success"].mean())
                    if len(without_override)
                    else None
                ),
            },
        }

        self.dm.save_decision_quality_metrics(
            ticker=ticker or "ALL",
            start_date=start_date or "",
            end_date=end_date or "",
            metrics_json=json.dumps(metrics),
        )

        return metrics

    def _load_data(
        self,
        ticker: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> pd.DataFrame:
        query = """
            SELECT dh.id, dh.timestamp, dh.ticker, dh.action_code, dh.confidence,
                   dh.safety_overrides_json, o.pnl_pct, o.success
            FROM decision_history dh
            LEFT JOIN outcomes o ON o.decision_id = dh.id
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

        with self.dm.connection() as conn:
            df = pd.read_sql(query, conn, params=params)

        if df.empty:
            return df

        safety_flags = []
        for raw in df["safety_overrides_json"].fillna("{}").tolist():
            try:
                val = json.loads(raw)
                safety_flags.append(1 if val.get("safety_override") else 0)
            except json.JSONDecodeError:
                safety_flags.append(0)

        df["safety_override"] = safety_flags
        return df
