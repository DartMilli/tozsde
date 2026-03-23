import json
from datetime import datetime, date
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from app.bootstrap.build_settings import build_settings
from app.infrastructure.repositories import DataManagerRepository
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)
DataManager = DataManagerRepository


def _create_data_repository(settings=None):
    try:
        return DataManager(settings=settings or build_settings())
    except TypeError:
        return DataManager()


class DecisionEffectivenessAnalyzer:
    """
    Computes profit-aware effectiveness scores for executed decisions.
    """

    def __init__(self, settings=None, data_manager=None):
        self.settings = settings or build_settings()
        self.dm = data_manager or _create_data_repository(self.settings)

    def compute_for_decision(self, decision_id: int) -> Optional[Dict]:
        row = self._load_single(decision_id)
        if not row:
            logger.info(
                "Effectiveness skipped: no outcome row for decision_id=%s", decision_id
            )
            return None

        score, components = self._compute_eds(row)
        self.dm.save_decision_effectiveness(
            decision_id=decision_id,
            ticker=row["ticker"],
            effectiveness_score=score,
            components_json=json.dumps(components),
        )
        return {"decision_id": decision_id, "effectiveness_score": score}

    def compute_for_range(
        self,
        ticker: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        window_days: int = 30,
    ) -> Dict:
        rows = self._load_range(ticker, start_date, end_date)
        if not rows:
            self._log_no_data_reason(ticker, start_date, end_date)
            return {"status": "no_data", "rows": 0}

        results = []
        for row in rows:
            score, components = self._compute_eds(row)
            self.dm.save_decision_effectiveness(
                decision_id=row["decision_id"],
                ticker=row["ticker"],
                effectiveness_score=score,
                components_json=json.dumps(components),
            )
            results.append({"decision_id": row["decision_id"], "score": score})

        if ticker:
            self._save_rolling_aggregate(
                ticker=ticker,
                window_days=window_days,
                as_of_date=end_date or datetime.now().date().isoformat(),
            )

        return {"status": "ok", "rows": len(results)}

    # ---------------- internal helpers ----------------

    def _load_single(self, decision_id: int) -> Optional[Dict]:
        query = """
            SELECT dh.id, dh.timestamp, dh.ticker, dh.safety_overrides_json,
                   o.pnl_pct, o.outcome_json, o.evaluated_at
            FROM decision_history dh
            JOIN outcomes o ON o.decision_id = dh.id
            WHERE dh.id = ?
        """
        with self.dm.connection() as conn:
            row = conn.execute(query, (decision_id,)).fetchone()

        if not row:
            return None

        return self._row_to_dict(row)

    def _load_range(
        self,
        ticker: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> list[Dict]:
        query = """
            SELECT dh.id, dh.timestamp, dh.ticker, dh.safety_overrides_json,
                   o.pnl_pct, o.outcome_json, o.evaluated_at
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

        with self.dm.connection() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_dict(r) for r in rows]

    def _log_no_data_reason(
        self,
        ticker: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> None:
        decisions_query = "SELECT COUNT(*) FROM decision_history WHERE 1=1"
        outcomes_query = "SELECT COUNT(*) FROM outcomes WHERE 1=1"
        joined_query = """
            SELECT COUNT(*)
            FROM decision_history dh
            JOIN outcomes o ON o.decision_id = dh.id
            WHERE 1=1
        """
        params = []
        out_params = []
        join_params = []
        if ticker:
            decisions_query += " AND ticker = ?"
            outcomes_query += " AND ticker = ?"
            joined_query += " AND dh.ticker = ?"
            params.append(ticker)
            out_params.append(ticker)
            join_params.append(ticker)
        if start_date:
            decisions_query += " AND date(timestamp) >= date(?)"
            outcomes_query += " AND date(decision_timestamp) >= date(?)"
            joined_query += " AND date(dh.timestamp) >= date(?)"
            params.append(start_date)
            out_params.append(start_date)
            join_params.append(start_date)
        if end_date:
            decisions_query += " AND date(timestamp) <= date(?)"
            outcomes_query += " AND date(decision_timestamp) <= date(?)"
            joined_query += " AND date(dh.timestamp) <= date(?)"
            params.append(end_date)
            out_params.append(end_date)
            join_params.append(end_date)

        with self.dm.connection() as conn:
            decisions_count = conn.execute(decisions_query, params).fetchone()[0]
            outcomes_count = conn.execute(outcomes_query, out_params).fetchone()[0]
            joined_count = conn.execute(joined_query, join_params).fetchone()[0]

        logger.info(
            "Effectiveness skipped: no joined decisions/outcomes. "
            "decisions=%s outcomes=%s joined=%s ticker=%s start=%s end=%s",
            decisions_count,
            outcomes_count,
            joined_count,
            ticker or "ALL",
            start_date or "N/A",
            end_date or "N/A",
        )

    def _row_to_dict(self, row) -> Dict:
        decision_id, ts, ticker, safety_json, pnl_pct, outcome_json, evaluated_at = row
        return {
            "decision_id": decision_id,
            "timestamp": ts,
            "ticker": ticker,
            "safety_overrides_json": safety_json or "{}",
            "pnl_pct": float(pnl_pct) if pnl_pct is not None else 0.0,
            "outcome_json": outcome_json or "{}",
            "evaluated_at": evaluated_at,
        }

    def _compute_eds(self, row: Dict) -> Tuple[float, Dict]:
        cfg = get_settings()
        pnl_pct = row.get("pnl_pct", 0.0)
        safety_penalty = 0.0
        try:
            safety = json.loads(row.get("safety_overrides_json") or "{}")
            if safety.get("safety_override"):
                safety_penalty = getattr(cfg, "P6_SAFETY_OVERRIDE_PENALTY", 0.0)
        except json.JSONDecodeError:
            safety_penalty = 0.0

        entry_date = self._parse_date(row.get("timestamp"))
        exit_date = self._parse_date(row.get("evaluated_at"))

        drawdown_penalty, volatility_penalty, stats = self._compute_trade_stats(
            ticker=row.get("ticker"),
            entry_date=entry_date,
            exit_date=exit_date,
        )

        eds = float(pnl_pct) - drawdown_penalty - volatility_penalty - safety_penalty

        components = {
            "realized_return": float(pnl_pct),
            "drawdown_penalty": round(drawdown_penalty, 6),
            "volatility_penalty": round(volatility_penalty, 6),
            "safety_override_penalty": round(safety_penalty, 6),
            "max_adverse_excursion": stats.get("max_adverse_excursion"),
            "volatility": stats.get("volatility"),
        }

        return round(eds, 6), components

    def _compute_trade_stats(
        self,
        ticker: str,
        entry_date: Optional[date],
        exit_date: Optional[date],
    ) -> Tuple[float, float, Dict]:
        if not ticker or not entry_date or not exit_date:
            return 0.0, 0.0, {"max_adverse_excursion": 0.0, "volatility": 0.0}

        ohlcv = self.dm.load_ohlcv(ticker)
        if ohlcv.empty:
            return 0.0, 0.0, {"max_adverse_excursion": 0.0, "volatility": 0.0}

        prices = ohlcv.loc[
            (ohlcv.index.date >= entry_date) & (ohlcv.index.date <= exit_date),
            "Close",
        ]
        if prices.empty:
            return 0.0, 0.0, {"max_adverse_excursion": 0.0, "volatility": 0.0}

        entry_price = float(prices.iloc[0])
        returns = prices.pct_change().dropna()

        min_return = ((prices.min() / entry_price) - 1.0) if entry_price else 0.0
        max_adverse_excursion = float(min_return) if min_return < 0 else 0.0

        cfg = get_settings()
        drawdown_penalty = abs(max_adverse_excursion) * getattr(
            cfg, "P6_DRAWNDOWN_PENALTY_FACTOR", 0.0
        )
        volatility = float(returns.std()) if not returns.empty else 0.0
        volatility_penalty = volatility * getattr(
            cfg, "P6_VOLATILITY_PENALTY_FACTOR", 0.0
        )

        stats = {
            "max_adverse_excursion": round(max_adverse_excursion, 6),
            "volatility": round(volatility, 6),
        }

        return drawdown_penalty, volatility_penalty, stats

    def _parse_date(self, raw: Optional[str]) -> Optional[date]:
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw).date()
        except ValueError:
            try:
                return datetime.strptime(raw, "%Y-%m-%d").date()
            except ValueError:
                return None

    def _save_rolling_aggregate(
        self, ticker: str, window_days: int, as_of_date: str
    ) -> None:
        query = """
            SELECT effectiveness_score
            FROM decision_effectiveness
            WHERE ticker = ?
            ORDER BY computed_at DESC
            LIMIT ?
        """
        with self.dm.connection() as conn:
            rows = conn.execute(query, (ticker, window_days)).fetchall()

        if not rows:
            return

        scores = [r[0] for r in rows if r[0] is not None]
        if not scores:
            return

        metrics = {
            "avg_effectiveness": float(np.mean(scores)),
            "median_effectiveness": float(np.median(scores)),
            "min_effectiveness": float(np.min(scores)),
            "max_effectiveness": float(np.max(scores)),
            "count": int(len(scores)),
        }

        self.dm.save_decision_effectiveness_rolling(
            ticker=ticker,
            window_days=window_days,
            as_of_date=as_of_date,
            metrics_json=json.dumps(metrics),
        )
