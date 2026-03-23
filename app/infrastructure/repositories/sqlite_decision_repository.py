from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Dict, List

import pandas as pd

from app.config.build_settings import build_settings
from app.core.ports.idecision_repository import IDecisionRepository
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)


class SqliteDecisionRepository(IDecisionRepository):
    def __init__(self, settings=None):
        cfg = settings or build_settings()
        db_path = getattr(cfg, "DB_PATH", None)
        if not db_path:
            raise RuntimeError("DB_PATH is not configured")
        self.db_path = str(db_path)

    @contextmanager
    def connection(self):
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def save_decision(self, decision: dict) -> int:
        return self.save_history_record(**decision)

    def fetch_decision(self, decision_id: int) -> dict:
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT id, timestamp, ticker, action_code, action_label,
                       confidence, wf_score, decision_blob, audit_blob
                FROM decision_history WHERE id = ?
                """,
                (decision_id,),
            ).fetchone()
        if not row:
            return {}
        return {
            "id": row[0],
            "timestamp": row[1],
            "ticker": row[2],
            "action_code": row[3],
            "action_label": row[4],
            "confidence": row[5],
            "wf_score": row[6],
            "decision": json.loads(row[7]) if row[7] else {},
            "audit": json.loads(row[8]) if row[8] else {},
        }

    def fetch_decisions_for_ticker(
        self, ticker: str, start_date: str, end_date: str
    ) -> list[dict]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT id FROM decision_history
                WHERE ticker = ? AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
                """,
                (ticker, start_date, end_date),
            ).fetchall()
        return [self.fetch_decision(row[0]) for row in rows]

    def save_history_record(
        self,
        ticker,
        action_code,
        label,
        confidence,
        wf_score,
        d_blob,
        a_blob,
        model_id=None,
        model_version=None,
        as_of_date: str = None,
        execution_price: float = None,
        features_hash: str = None,
        reliability_score: float = None,
        explanation_json=None,
        model_votes_json=None,
        safety_overrides_json=None,
        position_sizing_json=None,
        decision_source=None,
        timestamp: str = None,
    ):
        query = """
            INSERT INTO decision_history
            (timestamp, as_of_date, ticker, model_id, model_version, action_code, action_label, confidence, wf_score,
             reliability_score, execution_price, features_hash, decision_blob, audit_blob, explanation_json, model_votes_json,
             safety_overrides_json, position_sizing_json, decision_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self.connection() as conn:
            cursor = conn.execute(
                query,
                (
                    timestamp or datetime.now(timezone.utc).isoformat(),
                    as_of_date,
                    ticker,
                    model_id,
                    model_version,
                    action_code,
                    label,
                    confidence,
                    wf_score,
                    reliability_score,
                    execution_price,
                    features_hash,
                    d_blob,
                    a_blob,
                    explanation_json or "{}",
                    model_votes_json or "[]",
                    safety_overrides_json or "{}",
                    position_sizing_json or "{}",
                    decision_source,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def fetch_history_records_by_ticker(self, ticker: str) -> List[Dict]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, ticker, action_code, action_label,
                      model_id, model_version, confidence, wf_score, reliability_score,
                      execution_price, features_hash, as_of_date, decision_blob, audit_blob,
                       explanation_json, model_votes_json, safety_overrides_json,
                      position_sizing_json, decision_source
                FROM decision_history
                WHERE ticker = ?
                ORDER BY timestamp ASC
                """,
                (ticker,),
            ).fetchall()
        out = []
        for (
            ts,
            tkr,
            ac,
            al,
            model_id,
            model_version,
            conf,
            wf,
            reliability_score,
            execution_price,
            features_hash,
            as_of_date,
            d_blob,
            a_blob,
            explanation_json,
            model_votes_json,
            safety_overrides_json,
            position_sizing_json,
            decision_source,
        ) in rows:
            out.append(
                {
                    "timestamp": ts,
                    "ticker": tkr,
                    "model_id": model_id,
                    "model_version": model_version,
                    "as_of_date": as_of_date,
                    "reliability_score": reliability_score,
                    "execution_price": execution_price,
                    "features_hash": features_hash,
                    "decision": json.loads(d_blob) if d_blob else {},
                    "audit": json.loads(a_blob) if a_blob else {},
                    "explanation": (
                        json.loads(explanation_json) if explanation_json else {}
                    ),
                    "model_votes": (
                        json.loads(model_votes_json) if model_votes_json else []
                    ),
                    "safety_overrides": (
                        json.loads(safety_overrides_json)
                        if safety_overrides_json
                        else {}
                    ),
                    "position_sizing": (
                        json.loads(position_sizing_json) if position_sizing_json else {}
                    ),
                    "decision_source": decision_source,
                }
            )
        return out

    def has_decision_for_date(self, ticker: str, as_of_date: date) -> bool:
        query = """
            SELECT 1
            FROM decision_history
            WHERE ticker = ? AND (
                as_of_date = ? OR date(timestamp) = date(?)
            )
            LIMIT 1
        """
        with self.connection() as conn:
            row = conn.execute(
                query,
                (ticker, as_of_date.isoformat(), as_of_date.isoformat()),
            ).fetchone()
        return row is not None

    def fetch_history_range(self, ticker: str, start_iso: str, end_iso: str):
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, action_label, decision_blob, audit_blob
                FROM decision_history
                WHERE ticker = ? AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
                """,
                (ticker, start_iso, end_iso),
            ).fetchall()
        return rows

    def fetch_recent_outcomes(self, ticker: str, n: int = 3) -> List[Dict]:
        query = """
            SELECT pnl_pct, success, future_return, evaluated_at, exit_reason, horizon_days, outcome_json
            FROM outcomes
            WHERE ticker = ?
            ORDER BY decision_timestamp DESC
            LIMIT ?
        """
        with self.connection() as conn:
            rows = conn.execute(query, (ticker, n)).fetchall()
        out = []
        for (
            pnl_pct,
            success,
            future_return,
            evaluated_at,
            exit_reason,
            horizon_days,
            outcome_json,
        ) in rows:
            out.append(
                {
                    "pnl_pct": pnl_pct,
                    "success": bool(success) if success is not None else None,
                    "future_return": future_return,
                    "evaluated_at": evaluated_at,
                    "exit_reason": exit_reason,
                    "horizon_days": horizon_days,
                    "details": json.loads(outcome_json) if outcome_json else {},
                }
            )
        return out

    def get_today_recommendations(self) -> List[Dict]:
        today = datetime.now(timezone.utc).date().isoformat()
        with self.connection() as conn:
            return pd.read_sql(
                "SELECT * FROM recommendations WHERE date = ?",
                conn,
                params=[today],
            ).to_dict("records")

    def save_outcome(
        self,
        decision_id: int,
        ticker: str,
        decision_timestamp: str,
        pnl_pct: float,
        success: bool,
        future_return: float,
        exit_reason: str,
        horizon_days: int,
        outcome_json: str,
    ) -> None:
        query = """
            INSERT OR REPLACE INTO outcomes
            (decision_id, ticker, decision_timestamp, pnl_pct, success, future_return, evaluated_at,
             exit_reason, horizon_days, outcome_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self.connection() as conn:
            conn.execute(
                query,
                (
                    decision_id,
                    ticker,
                    decision_timestamp,
                    pnl_pct,
                    int(bool(success)),
                    future_return,
                    datetime.now(timezone.utc).isoformat(),
                    exit_reason,
                    horizon_days,
                    outcome_json,
                ),
            )
            conn.commit()

        try:
            from app.analysis.decision_effectiveness import (
                DecisionEffectivenessAnalyzer,
            )

            DecisionEffectivenessAnalyzer().compute_for_decision(decision_id)
        except Exception as exc:
            logger.error(
                f"Failed to compute effectiveness for decision {decision_id}: {exc}"
            )

    def get_unevaluated_buy_decisions(self, limit=100):
        query = """
            SELECT dh.id, dh.timestamp, dh.ticker, dh.audit_blob
            FROM decision_history dh
            LEFT JOIN outcomes o ON o.decision_id = dh.id
            WHERE dh.action_code = 1 AND o.decision_id IS NULL
              AND (dh.audit_blob IS NULL OR dh.audit_blob NOT LIKE '%"outcome"%')
            ORDER BY dh.timestamp DESC
            LIMIT ?
        """
        with self.connection() as conn:
            return conn.execute(query, (limit,)).fetchall()

    def fetch_latest_decision_quality_metrics(self, ticker: str) -> Dict:
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT metrics_json
                FROM decision_quality_metrics
                WHERE ticker = ?
                ORDER BY computed_at DESC
                LIMIT 1
                """,
                (ticker,),
            ).fetchone()
        if not row:
            return {}
        return json.loads(row[0]) if row[0] else {}

    def save_walk_forward_result(self, ticker: str, result_json: str) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO walk_forward_results
                (ticker, computed_at, result_json)
                VALUES (?, ?, ?)
                """,
                (ticker, datetime.now(timezone.utc).isoformat(), result_json),
            )
            conn.commit()

    def initialize_tables(self) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_id INTEGER UNIQUE,
                    ticker TEXT,
                    decision_timestamp TEXT,
                    pnl_pct REAL,
                    success INTEGER,
                    future_return REAL,
                    evaluated_at TEXT,
                    exit_reason TEXT,
                    horizon_days INTEGER,
                    outcome_json TEXT,
                    FOREIGN KEY(decision_id) REFERENCES decision_history(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS decision_effectiveness (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_id INTEGER UNIQUE,
                    ticker TEXT,
                    effectiveness_score REAL,
                    components_json TEXT,
                    computed_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS decision_effectiveness_rolling (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    window_days INTEGER,
                    as_of_date TEXT,
                    metrics_json TEXT,
                    computed_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    cash REAL,
                    equity REAL,
                    pnl_pct REAL,
                    positions_json TEXT,
                    source TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS decision_quality_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    computed_at TEXT,
                    metrics_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS confidence_calibration (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    method TEXT,
                    computed_at TEXT,
                    params_json TEXT,
                    metrics_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS wf_stability_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    computed_at TEXT,
                    metrics_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS safety_stress_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    scenario TEXT,
                    computed_at TEXT,
                    results_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS validation_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    computed_at TEXT,
                    report_json TEXT
                )
                """
            )
            conn.commit()

    def save_decision_effectiveness(
        self,
        decision_id: int,
        ticker: str,
        effectiveness_score: float,
        components_json: str,
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO decision_effectiveness
                (decision_id, ticker, effectiveness_score, components_json, computed_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    decision_id,
                    ticker,
                    effectiveness_score,
                    components_json,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()

    def save_decision_effectiveness_rolling(
        self,
        ticker: str,
        window_days: int,
        as_of_date: str,
        metrics_json: str,
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO decision_effectiveness_rolling
                (ticker, window_days, as_of_date, metrics_json, computed_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    ticker,
                    window_days,
                    as_of_date,
                    metrics_json,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()

    def save_portfolio_state(
        self,
        timestamp: str,
        cash: float,
        equity: float,
        pnl_pct: float,
        positions_json: str,
        source: str,
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO portfolio_state
                (timestamp, cash, equity, pnl_pct, positions_json, source)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (timestamp, cash, equity, pnl_pct, positions_json, source),
            )
            conn.commit()

    def fetch_portfolio_state_range(self, start_iso: str, end_iso: str) -> List[Dict]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, cash, equity, pnl_pct, positions_json, source
                FROM portfolio_state
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
                """,
                (start_iso, end_iso),
            ).fetchall()
        out = []
        for ts, cash, equity, pnl_pct, positions_json, source in rows:
            out.append(
                {
                    "timestamp": ts,
                    "cash": cash,
                    "equity": equity,
                    "pnl_pct": pnl_pct,
                    "positions": json.loads(positions_json) if positions_json else {},
                    "source": source,
                }
            )
        return out

    def fetch_latest_portfolio_state(self, source: str = None) -> Dict:
        query = """
            SELECT timestamp, cash, equity, pnl_pct, positions_json, source
            FROM portfolio_state
        """
        params = []
        if source:
            query += " WHERE source = ?"
            params.append(source)
        query += " ORDER BY timestamp DESC LIMIT 1"

        with self.connection() as conn:
            row = conn.execute(query, params).fetchone()
        if not row:
            return {}

        ts, cash, equity, pnl_pct, positions_json, src = row
        return {
            "timestamp": ts,
            "cash": cash,
            "equity": equity,
            "pnl_pct": pnl_pct,
            "positions": json.loads(positions_json) if positions_json else {},
            "source": src,
        }

    def save_decision_quality_metrics(
        self, ticker: str, start_date: str, end_date: str, metrics_json: str
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO decision_quality_metrics
                (ticker, start_date, end_date, computed_at, metrics_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    ticker,
                    start_date,
                    end_date,
                    datetime.now(timezone.utc).isoformat(),
                    metrics_json,
                ),
            )
            conn.commit()

    def save_confidence_calibration(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        method: str,
        params_json: str,
        metrics_json: str,
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO confidence_calibration
                (ticker, start_date, end_date, method, computed_at, params_json, metrics_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker,
                    start_date,
                    end_date,
                    method,
                    datetime.now(timezone.utc).isoformat(),
                    params_json,
                    metrics_json,
                ),
            )
            conn.commit()

    def fetch_latest_confidence_calibration(
        self, ticker: str, as_of_date: str | None = None
    ) -> Dict[str, str]:
        query = """
            SELECT params_json, metrics_json, computed_at
            FROM confidence_calibration
            WHERE ticker = ?
        """
        params = [ticker]
        if as_of_date:
            query += " AND date(computed_at) <= date(?)"
            params.append(as_of_date)
        query += " ORDER BY computed_at DESC LIMIT 1"

        with self.connection() as conn:
            row = conn.execute(query, params).fetchone()
        if not row:
            return {}

        params_json, metrics_json, computed_at = row
        return {
            "params_json": params_json,
            "metrics_json": metrics_json,
            "computed_at": computed_at,
        }

    def save_wf_stability_metrics(self, ticker: str, metrics_json: str) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO wf_stability_metrics
                (ticker, computed_at, metrics_json)
                VALUES (?, ?, ?)
                """,
                (ticker, datetime.now(timezone.utc).isoformat(), metrics_json),
            )
            conn.commit()

    def save_safety_stress_results(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        scenario: str,
        results_json: str,
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO safety_stress_results
                (ticker, start_date, end_date, scenario, computed_at, results_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker,
                    start_date,
                    end_date,
                    scenario,
                    datetime.now(timezone.utc).isoformat(),
                    results_json,
                ),
            )
            conn.commit()

    def save_validation_report(self, report_json: str) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO validation_reports
                (computed_at, report_json)
                VALUES (?, ?)
                """,
                (datetime.now(timezone.utc).isoformat(), report_json),
            )
            conn.commit()

    def fetch_latest_validation_report(self) -> Dict:
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT report_json
                FROM validation_reports
                ORDER BY computed_at DESC
                LIMIT 1
                """
            ).fetchone()
        if not row:
            return {}
        return json.loads(row[0]) if row[0] else {}

    def update_history_audit(self, row_id, new_audit_blob) -> None:
        with self.connection() as conn:
            conn.execute(
                "UPDATE decision_history SET audit_blob = ? WHERE id = ?",
                (new_audit_blob, row_id),
            )
            conn.commit()

    def save_model_reliability(self, ticker, date, score_details) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO model_reliability
                (ticker, date, score_details)
                VALUES (?, ?, ?)
                """,
                (ticker, date, score_details),
            )
            conn.commit()

    def save_market_data(self, symbol: str, df: pd.DataFrame):
        if df is None or df.empty:
            return
        data = (
            df[["Close"]]
            .rename(columns={"Close": "value"})
            .assign(symbol=symbol)
            .reset_index()
        )
        data["date"] = pd.to_datetime(data["Date"]).dt.strftime("%Y-%m-%d")

        with self.connection() as conn:
            data[["symbol", "date", "value"]].to_sql(
                "temp_market_metadata", conn, if_exists="replace", index=False
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO market_metadata(symbol,date,value)
                SELECT symbol,date,value FROM temp_market_metadata
                """
            )
            conn.execute("DROP TABLE temp_market_metadata")
            conn.commit()

    def get_market_data(self, symbol: str, days: int = 1):
        with self.connection() as conn:
            return conn.execute(
                """
                SELECT date, value
                FROM market_metadata
                WHERE symbol=?
                ORDER BY date DESC
                LIMIT ?
                """,
                (symbol, days),
            ).fetchall()

    def get_ticker_historical_recommendations(self, ticker, start, end):
        query = "SELECT * FROM recommendations WHERE ticker=? AND date>=? AND date<=?"
        with self.connection() as conn:
            return pd.read_sql(query, conn, params=[ticker, start, end]).to_dict(
                "records"
            )

    def get_strategy_accuracy(self, ticker, lookback_decisions=20):
        query = """
            SELECT decision_blob, timestamp FROM decision_history
            WHERE ticker = ? AND action_code = 1
            ORDER BY timestamp DESC LIMIT ?
        """
        with self.connection() as conn:
            return conn.execute(query, (ticker, lookback_decisions)).fetchall()
