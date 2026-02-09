import sqlite3
import pandas as pd
import logging
import json
from datetime import datetime, timezone, timedelta, date
from typing import List, Dict
from app.config.config import Config
from contextlib import contextmanager

from app.indicators.technical import sma

logger = logging.getLogger(__name__)


class DataManager:
    """
    Data Access Layer (DAL) - Az egyetlen osztály, amely közvetlenül érintkezik az SQLite adatbázissal.
    Felelős az adatok tárolásáért, lekérdezéséért és a séma integritásáért.
    """

    def __init__(self):
        self.db_path = str(Config.DB_PATH)  # Convert Path to string for sqlite3

    @contextmanager
    def _get_conn(self):
        """Privát metódus az adatbázis kapcsolat létrehozásához."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    # --- SÉMA ÉS INICIALIZÁLÁS (P9/Ops) ---

    def initialize_tables(self):
        """
        Létrehozza a szükséges táblákat, ha azok még nem léteznek.
        Ezt hívja az apply_schema.py is.
        """
        logger.info("Initializing database tables...")
        with self._get_conn() as conn:
            cur = conn.cursor()

            # 1. Piaci adatok (OHLCV)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS ohlcv (
                    ticker TEXT, date TEXT, open REAL, high REAL, 
                    low REAL, close REAL, volume REAL,
                    PRIMARY KEY (ticker, date)
                )"""
            )

            # 2. Portfólió/Tranzakciók (Váltja a portfolio.py-t)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT, ticker TEXT, side TEXT, 
                    qty REAL, price REAL, strategy TEXT
                )"""
            )

            # 3. Napi Ajánlások (Váltja a recommendation_logger.py-t)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS recommendations (
                    date TEXT, ticker TEXT, signal TEXT, 
                    confidence REAL, wf_score REAL, params TEXT,
                    PRIMARY KEY (date, ticker)
                )"""
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS model_reliability (
                    ticker TEXT,
                    date TEXT,
                    score_details TEXT,
                    PRIMARY KEY (ticker, date)
                )"""
            )

            # Decision History tábla az auditálhatósághoz (P4.5 Roadmap)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS decision_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    ticker TEXT,
                    model_id TEXT,
                    action_code INTEGER,
                    action_label TEXT,
                    confidence REAL,
                    wf_score REAL,
                    decision_blob TEXT,
                    audit_blob TEXT,
                    explanation_json TEXT,
                    model_votes_json TEXT,
                    safety_overrides_json TEXT,
                    position_sizing_json TEXT,
                    decision_source TEXT
                )
            """
            )
            # Ensure new columns exist for existing DBs
            existing_cols = {
                row[1]
                for row in cur.execute("PRAGMA table_info(decision_history)").fetchall()
            }
            required_cols = {
                "model_id",
                "explanation_json",
                "model_votes_json",
                "safety_overrides_json",
                "position_sizing_json",
                "decision_source",
            }
            for col in required_cols - existing_cols:
                cur.execute(f"ALTER TABLE decision_history ADD COLUMN {col} TEXT")
            # Market metadata (e.g., VIX / IRX)
            cur.execute(
                """
            CREATE TABLE IF NOT EXISTS market_metadata (
                symbol TEXT,
                date   TEXT,
                value  REAL,
                PRIMARY KEY(symbol, date)
            )"""
            )

            # Outcomes linked to decisions
            cur.execute(
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
            existing_outcome_cols = {
                row[1] for row in cur.execute("PRAGMA table_info(outcomes)").fetchall()
            }
            if "exit_reason" not in existing_outcome_cols:
                cur.execute("ALTER TABLE outcomes ADD COLUMN exit_reason TEXT")
            if "horizon_days" not in existing_outcome_cols:
                cur.execute("ALTER TABLE outcomes ADD COLUMN horizon_days INTEGER")

            # Decision effectiveness (P6.1)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS decision_effectiveness (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_id INTEGER UNIQUE,
                    ticker TEXT,
                    effectiveness_score REAL,
                    components_json TEXT,
                    computed_at TEXT,
                    FOREIGN KEY(decision_id) REFERENCES decision_history(id)
                )
                """
            )
            cur.execute(
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

            # Model trust metrics (P6.3)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS model_trust_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_path TEXT,
                    ticker TEXT,
                    trust_weight REAL,
                    metrics_json TEXT,
                    computed_at TEXT
                )
                """
            )

            # Model registry (P6.5)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS model_registry (
                    model_id TEXT PRIMARY KEY,
                    ticker TEXT,
                    model_type TEXT,
                    wf_score REAL,
                    model_path TEXT,
                    status TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    metadata_json TEXT
                )
                """
            )

            # Portfolio state snapshots (paper/live/backtest)
            cur.execute(
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

            # Decision quality metrics
            cur.execute(
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

            # Confidence calibration parameters
            cur.execute(
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

            # Walk-forward results and stability metrics
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS walk_forward_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    computed_at TEXT,
                    result_json TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS wf_stability_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    computed_at TEXT,
                    metrics_json TEXT
                )
                """
            )

            # Safety stress test results
            cur.execute(
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

            # Unified validation reports
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS validation_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    computed_at TEXT,
                    report_json TEXT
                )
                """
            )

            # Indexek a teljesítmény javítására
            cur.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker ON ohlcv(ticker)")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_dh_ticker_ts ON decision_history(ticker, timestamp)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_outcomes_ticker_ts ON outcomes(ticker, decision_timestamp)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_portfolio_ts ON portfolio_state(timestamp)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_dqm_ticker ON decision_quality_metrics(ticker)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_calibration_ticker ON confidence_calibration(ticker)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_wf_results_ticker ON walk_forward_results(ticker)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_wf_metrics_ticker ON wf_stability_metrics(ticker)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_stress_ticker ON safety_stress_results(ticker)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_rec_date ON recommendations(date)"
            )

            # Pipeline metrics table (P9 - Monitoring)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS pipeline_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    ticker TEXT NOT NULL,
                    status TEXT NOT NULL,
                    duration_sec REAL,
                    error_message TEXT,
                    execution_date DATE
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_metrics_date ON pipeline_metrics(execution_date)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_metrics_ticker ON pipeline_metrics(ticker)"
            )

            conn.commit()
        logger.info("Schema verified and ready.")

    # --- PIACI ADATOK KEZELÉSE (OHLCV) ---

    def save_ohlcv(self, ticker, df):
        """Elmenti a DataFrame-et az ohlcv táblába (Upsert logika)."""
        if df is None or df.empty:
            return

        # Előkészítés a mentésre
        df_to_save = df.copy()
        if "Ticker" not in df_to_save.columns:
            df_to_save["ticker"] = ticker

        # Reset index, hogy a dátum is oszlop legyen a mentéshez
        df_to_save = df_to_save.reset_index()

        def _norm_col(col):
            if isinstance(col, tuple):
                col = col[0]
            return str(col).lower()

        df_to_save.columns = [_norm_col(c) for c in df_to_save.columns]

        with self._get_conn() as conn:
            # Ideiglenes tábla a gyors upserthez
            df_to_save.to_sql("temp_ohlcv", conn, if_exists="replace", index=False)
            conn.execute(
                """
                INSERT OR REPLACE INTO ohlcv (ticker, date, open, high, low, close, volume)
                SELECT
                    ticker,
                    date,
                    open,
                    high,
                    low,
                    close,
                    volume
                FROM temp_ohlcv
                """
            )
            conn.execute("DROP TABLE temp_ohlcv")
            conn.commit()

    def load_ohlcv(self, ticker, start_date=None):
        """
        Betölti a piaci adatokat a DB-ből.
        Paraméterezett lekérdezést használ az SQL injection ellen.
        """
        params = [ticker]
        query = "SELECT * FROM ohlcv WHERE ticker = ?"

        if start_date:
            query += " AND date >= ?"
            params.append(
                start_date.isoformat()
                if hasattr(start_date, "isoformat")
                else str(start_date)
            )

        query += " ORDER BY date ASC"

        with self._get_conn() as conn:
            df = pd.read_sql(query, conn, params=params, parse_dates=["date"])
            if not df.empty:
                df.set_index("date", inplace=True)
                # Oszlopnevek visszaállítása a várt formátumra
                df.columns = [c.capitalize() for c in df.columns]
            return df

    def load_ohlcv_batch(self, tickers, start_date=None) -> Dict[str, pd.DataFrame]:
        """
        Batch load OHLCV data for multiple tickers.
        Returns: {ticker: DataFrame}
        """
        if not tickers:
            return {}

        placeholders = ",".join(["?"] * len(tickers))
        params = list(tickers)
        query = f"SELECT * FROM ohlcv WHERE ticker IN ({placeholders})"

        if start_date:
            query += " AND date >= ?"
            params.append(
                start_date.isoformat()
                if hasattr(start_date, "isoformat")
                else str(start_date)
            )

        query += " ORDER BY date ASC"

        with self._get_conn() as conn:
            df = pd.read_sql(query, conn, params=params, parse_dates=["date"])

        if df.empty:
            return {}

        out = {}
        for tkr, group in df.groupby("ticker"):
            group = group.copy()
            group.set_index("date", inplace=True)
            group.columns = [c.capitalize() for c in group.columns]
            out[tkr] = group
        return out

    # --- DÖNTÉSI ELŐZMÉNYEK (History & Audit) ---

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
        explanation_json=None,
        model_votes_json=None,
        safety_overrides_json=None,
        position_sizing_json=None,
        decision_source=None,
        timestamp: str = None,
    ):
        """Enkapszulált mentés a history táblába."""
        query = """
            INSERT INTO decision_history 
            (timestamp, ticker, model_id, action_code, action_label, confidence, wf_score, decision_blob, audit_blob,
             explanation_json, model_votes_json, safety_overrides_json, position_sizing_json, decision_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._get_conn() as conn:
            cursor = conn.execute(
                query,
                (
                    timestamp or datetime.now(timezone.utc).isoformat(),
                    ticker,
                    model_id,
                    action_code,
                    label,
                    confidence,
                    wf_score,
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

    def get_history_range(self, ticker, start_iso, end_iso):
        """Lekérdezi a korábbi döntéseket egy adott időintervallumban."""
        query = """
            SELECT timestamp, action_label, decision_blob, audit_blob 
            FROM decision_history 
            WHERE ticker = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
        """
        with self._get_conn() as conn:
            return conn.execute(query, (ticker, start_iso, end_iso)).fetchall()

    # --- ELEMZÉSI SEGÉDFÜGGVÉNYEK (DataManager szinten) ---

    def get_market_regime_is_bear(self, benchmark="SPY", period=200, ref_date=None):
        """
        ref_date: string (YYYY-MM-DD) vagy date object. Ha None, a mai nap.
        """
        # Itt fontos: load_ohlcv-nél szűrni kell, vagy utólag vágni
        # A load_ohlcv támogat start_date-et, de end_date-et jelenleg nem.
        # Ezért betöltjük, és Pandas-ban vágjuk.

        df = self.load_ohlcv(benchmark)

        if df.empty:
            return False

        # Időbeli vágás (Look-ahead bias ellen)
        if ref_date:
            df = df[df.index <= pd.Timestamp(ref_date)]

        if len(df) < period:
            return False

        closes = df["Close"].values
        sma_values = sma(closes, period)

        return closes[-1] < sma_values[-1]

    def get_correlation_matrix(self, tickers, lookback_days=90, ref_date=None):
        combined_data = {}

        start_date = None
        if ref_date:
            start_date = pd.Timestamp(ref_date) - timedelta(days=lookback_days * 2)

        batch = self.load_ohlcv_batch(tickers, start_date=start_date)

        for t in tickers:
            df = batch.get(t)
            if df is not None and not df.empty:
                if ref_date:
                    df = df[df.index <= pd.Timestamp(ref_date)]

                combined_data[t] = df["Close"].tail(lookback_days)

        if not combined_data:
            return pd.DataFrame()

        prices_df = pd.DataFrame(combined_data)
        return prices_df.pct_change().dropna().corr()

    # --- AJÁNLÁSOK KEZELÉSE ---
    def log_recommendation(self, ticker, signal, confidence, params=None):
        today = datetime.now().strftime("%Y-%m-%d")
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO recommendations (date, ticker, signal, confidence, params)
                VALUES (?, ?, ?, ?, ?)
            """,
                (today, ticker, signal, confidence, json.dumps(params)),
            )
            conn.commit()

    def get_today_recommendations(self):
        today = datetime.now().strftime("%Y-%m-%d")
        with self._get_conn() as conn:
            return pd.read_sql(
                f"SELECT * FROM recommendations WHERE date='{today}'", conn
            ).to_dict("records")

    def save_model_reliability(self, ticker, date, score_details):
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO model_reliability
                (ticker, date, score_details)
                VALUES (?, ?, ?)
                """,
                (ticker, date, score_details),
            )
            conn.commit()

    @contextmanager
    def connection(self):
        with self._get_conn() as conn:
            yield conn

    def save_market_data(self, symbol: str, df: pd.DataFrame):
        if df.empty:
            return
        data = (
            df[["Close"]]
            .rename(columns={"Close": "value"})
            .assign(symbol=symbol)
            .reset_index()
        )
        data["date"] = pd.to_datetime(data["Date"]).dt.strftime("%Y-%m-%d")
        with self._get_conn() as conn:
            data[["symbol", "date", "value"]].to_sql(
                "temp_market_metadata", conn, if_exists="replace", index=False
            )
            conn.execute(
                """INSERT OR REPLACE INTO market_metadata(symbol,date,value)
                            SELECT symbol,date,value FROM temp_market_metadata"""
            )
            conn.execute("DROP TABLE temp_market_metadata")
            conn.commit()

    def get_market_data(self, symbol: str, days: int = 1):
        with self._get_conn() as conn:
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

    # ---------- Decision history DAO (no raw SQL in HistoryStore) ----------
    def fetch_history_records_by_ticker(self, ticker: str) -> List[Dict]:
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, ticker, action_code, action_label,
                       model_id, confidence, wf_score, decision_blob, audit_blob,
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
            model_id,
            ac,
            al,
            conf,
            wf,
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
            WHERE ticker = ? AND date(timestamp) = date(?)
            LIMIT 1
        """
        with self._get_conn() as conn:
            row = conn.execute(query, (ticker, as_of_date.isoformat())).fetchone()
        return row is not None

    def fetch_history_range(
        self, ticker: str, start_iso: str, end_iso: str
    ) -> List[Dict]:
        with self._get_conn() as conn:
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
        with self._get_conn() as conn:
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
        with self._get_conn() as conn:
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
        except Exception as e:
            logger.error(
                f"Failed to compute effectiveness for decision {decision_id}: {e}"
            )

    # --- P6.1 Decision Effectiveness ---

    def save_decision_effectiveness(
        self,
        decision_id: int,
        ticker: str,
        effectiveness_score: float,
        components_json: str,
    ) -> None:
        query = """
            INSERT OR REPLACE INTO decision_effectiveness
            (decision_id, ticker, effectiveness_score, components_json, computed_at)
            VALUES (?, ?, ?, ?, ?)
        """
        with self._get_conn() as conn:
            conn.execute(
                query,
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
        query = """
            INSERT INTO decision_effectiveness_rolling
            (ticker, window_days, as_of_date, metrics_json, computed_at)
            VALUES (?, ?, ?, ?, ?)
        """
        with self._get_conn() as conn:
            conn.execute(
                query,
                (
                    ticker,
                    window_days,
                    as_of_date,
                    metrics_json,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()

    # --- P6.3 Model Trust ---

    def save_model_trust_metrics(
        self,
        model_path: str,
        ticker: str,
        trust_weight: float,
        metrics_json: str,
    ) -> None:
        query = """
            INSERT INTO model_trust_metrics
            (model_path, ticker, trust_weight, metrics_json, computed_at)
            VALUES (?, ?, ?, ?, ?)
        """
        with self._get_conn() as conn:
            conn.execute(
                query,
                (
                    model_path,
                    ticker,
                    trust_weight,
                    metrics_json,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()

    def fetch_latest_model_trust_weights(self, ticker: str) -> Dict[str, float]:
        query = """
            SELECT model_path, trust_weight, MAX(computed_at)
            FROM model_trust_metrics
            WHERE ticker = ?
            GROUP BY model_path
        """
        with self._get_conn() as conn:
            rows = conn.execute(query, (ticker,)).fetchall()
        return {model_path: (tw or 0.0) for model_path, tw, _ in rows}

    # --- P6.5 Model Registry ---

    def register_model(
        self,
        model_id: str,
        ticker: str,
        model_type: str,
        wf_score: float,
        model_path: str,
        status: str = "candidate",
        metadata_json: str = None,
    ) -> None:
        query = """
            INSERT OR REPLACE INTO model_registry
            (model_id, ticker, model_type, wf_score, model_path, status, created_at, updated_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute(
                query,
                (
                    model_id,
                    ticker,
                    model_type,
                    wf_score,
                    model_path,
                    status,
                    now,
                    now,
                    metadata_json or "{}",
                ),
            )
            conn.commit()

    def update_model_status(self, model_id: str, status: str) -> None:
        query = """
            UPDATE model_registry
            SET status = ?, updated_at = ?
            WHERE model_id = ?
        """
        with self._get_conn() as conn:
            conn.execute(
                query,
                (status, datetime.now(timezone.utc).isoformat(), model_id),
            )
            conn.commit()

    def fetch_active_models(self, ticker: str, limit: int = 3) -> List[Dict]:
        query = """
            SELECT model_id, model_path, wf_score, model_type
            FROM model_registry
            WHERE ticker = ? AND status = 'active'
            ORDER BY updated_at DESC
            LIMIT ?
        """
        with self._get_conn() as conn:
            rows = conn.execute(query, (ticker, limit)).fetchall()
        return [
            {
                "model_id": model_id,
                "model_path": model_path,
                "wf_score": wf_score,
                "model_type": model_type,
            }
            for model_id, model_path, wf_score, model_type in rows
        ]

    def fetch_latest_decision_quality_metrics(self, ticker: str) -> Dict:
        query = """
            SELECT metrics_json
            FROM decision_quality_metrics
            WHERE ticker = ?
            ORDER BY computed_at DESC
            LIMIT 1
        """
        with self._get_conn() as conn:
            row = conn.execute(query, (ticker,)).fetchone()
        if not row:
            return {}
        return json.loads(row[0]) if row[0] else {}

    def save_portfolio_state(
        self,
        timestamp: str,
        cash: float,
        equity: float,
        pnl_pct: float,
        positions_json: str,
        source: str,
    ) -> None:
        query = """
            INSERT INTO portfolio_state
            (timestamp, cash, equity, pnl_pct, positions_json, source)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        with self._get_conn() as conn:
            conn.execute(
                query,
                (timestamp, cash, equity, pnl_pct, positions_json, source),
            )
            conn.commit()

    def fetch_portfolio_state_range(self, start_iso: str, end_iso: str) -> List[Dict]:
        query = """
            SELECT timestamp, cash, equity, pnl_pct, positions_json, source
            FROM portfolio_state
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
        """
        with self._get_conn() as conn:
            rows = conn.execute(query, (start_iso, end_iso)).fetchall()
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

        with self._get_conn() as conn:
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

    # --- Phase 5 analytics persistence ---

    def save_decision_quality_metrics(
        self, ticker: str, start_date: str, end_date: str, metrics_json: str
    ) -> None:
        query = """
            INSERT INTO decision_quality_metrics
            (ticker, start_date, end_date, computed_at, metrics_json)
            VALUES (?, ?, ?, ?, ?)
        """
        with self._get_conn() as conn:
            conn.execute(
                query,
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
        query = """
            INSERT INTO confidence_calibration
            (ticker, start_date, end_date, method, computed_at, params_json, metrics_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self._get_conn() as conn:
            conn.execute(
                query,
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

    def save_walk_forward_result(self, ticker: str, result_json: str) -> None:
        query = """
            INSERT INTO walk_forward_results
            (ticker, computed_at, result_json)
            VALUES (?, ?, ?)
        """
        with self._get_conn() as conn:
            conn.execute(
                query,
                (ticker, datetime.now(timezone.utc).isoformat(), result_json),
            )
            conn.commit()

    def save_wf_stability_metrics(self, ticker: str, metrics_json: str) -> None:
        query = """
            INSERT INTO wf_stability_metrics
            (ticker, computed_at, metrics_json)
            VALUES (?, ?, ?)
        """
        with self._get_conn() as conn:
            conn.execute(
                query,
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
        query = """
            INSERT INTO safety_stress_results
            (ticker, start_date, end_date, scenario, computed_at, results_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        with self._get_conn() as conn:
            conn.execute(
                query,
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
        query = """
            INSERT INTO validation_reports
            (computed_at, report_json)
            VALUES (?, ?)
        """
        with self._get_conn() as conn:
            conn.execute(
                query,
                (datetime.now(timezone.utc).isoformat(), report_json),
            )
            conn.commit()

    def fetch_latest_validation_report(self) -> Dict:
        query = """
            SELECT report_json
            FROM validation_reports
            ORDER BY computed_at DESC
            LIMIT 1
        """
        with self._get_conn() as conn:
            row = conn.execute(query).fetchone()
        if not row:
            return {}
        return json.loads(row[0])

    # ---------- Convenience connection() ----------

    # --- RÉGI KOMPATIBILITÁSI METÓDUSOK ---

    def get_ticker_historical_recommendations(self, ticker, start, end):
        """Régi API kompatibilitás a recommendation táblához (ha még használatban van valahol)."""
        with self._get_conn() as conn:
            query = (
                "SELECT * FROM recommendations WHERE ticker=? AND date>=? AND date<=?"
            )
            return pd.read_sql(query, conn, params=[ticker, start, end]).to_dict(
                "records"
            )

    def get_strategy_accuracy(self, ticker, lookback_decisions=20):
        """
        Visszaadja az utolsó X döntés sikerességét.
        Ez egy egyszerűsített implementáció: megnézi, hogy vételi jel után emelkedett-e az ár.
        """
        # FIGYELEM: Ez komplex SQL-t igényelne a jövőbeli árakkal való joinoláshoz.
        # A snapshot egyszerűsége miatt most csak a 'decision_blob'-okat adjuk vissza,
        # és Pythonban dolgozzuk fel.
        query = """
            SELECT decision_blob, timestamp FROM decision_history 
            WHERE ticker = ? AND action_code = 1 
            ORDER BY timestamp DESC LIMIT ?
        """
        with self._get_conn() as conn:
            rows = conn.execute(query, (ticker, lookback_decisions)).fetchall()
        return rows

    def get_unevaluated_buy_decisions(self, limit=100):
        """P8: Visszaadja azokat a vételi döntéseket, amiknél még nincs outcome beírva."""
        query = """
            SELECT dh.id, dh.timestamp, dh.ticker, dh.audit_blob
            FROM decision_history dh
            LEFT JOIN outcomes o ON o.decision_id = dh.id
            WHERE dh.action_code = 1 AND o.decision_id IS NULL
              AND (dh.audit_blob IS NULL OR dh.audit_blob NOT LIKE '%"outcome"%')
            ORDER BY dh.timestamp DESC
            LIMIT ?
        """
        with self._get_conn() as conn:
            return conn.execute(query, (limit,)).fetchall()

    def get_top_models(self, ticker: str, limit: int = 3):
        """Return active models ordered by trust weight when available."""
        try:
            models = self.fetch_active_models(ticker=ticker, limit=limit)
            if not models:
                return []

            trust = self.fetch_latest_model_trust_weights(ticker=ticker)
            ranked = []
            for m in models:
                model_path = m.get("model_path")
                weight = trust.get(model_path, 0.0)
                ranked.append((model_path, weight))

            ranked.sort(key=lambda x: x[1], reverse=True)
            return ranked[:limit]
        except sqlite3.OperationalError:
            return []

    def update_history_audit(self, row_id, new_audit_blob):
        """P8: Frissíti egy korábbi döntés audit logját (pl. eredménnyel)."""
        query = "UPDATE decision_history SET audit_blob = ? WHERE id = ?"
        with self._get_conn() as conn:
            conn.execute(query, (new_audit_blob, row_id))
            conn.commit()

    # --- PIPELINE METRICS (P9 - MONITORING) ---

    def log_pipeline_execution(
        self, ticker: str, status: str, duration_sec: float, error_message: str = None
    ) -> bool:
        """Log pipeline execution to database."""
        try:
            with self._get_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO pipeline_metrics
                    (ticker, status, duration_sec, error_message, execution_date)
                    VALUES (?, ?, ?, ?, DATE('now'))
                    """,
                    (ticker, status, duration_sec, error_message),
                )
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to log pipeline execution: {e}")
            return False

    def log_backtest_execution(
        self,
        ticker: str,
        wf_score: float,
        trades_count: int,
        profit_factor: float,
        error_message: str = None,
    ) -> bool:
        """Log backtest execution to database."""
        try:
            with self._get_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO pipeline_metrics
                    (ticker, status, duration_sec, error_message, execution_date)
                    VALUES (?, ?, ?, ?, DATE('now'))
                    """,
                    (
                        ticker,
                        "success" if error_message is None else "error",
                        0.0,
                        error_message
                        or f"WF={wf_score:.3f} PF={profit_factor:.2f} trades={trades_count}",
                    ),
                )
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to log backtest execution: {e}")
            return False

    def get_recent_metrics(self, hours: int = 24) -> dict:
        """Get aggregated metrics for the past N hours."""
        try:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    """
                    SELECT status, COUNT(*) as count, AVG(duration_sec) as avg_duration
                    FROM pipeline_metrics
                    WHERE timestamp >= datetime('now', '-' || ? || ' hours')
                    GROUP BY status
                    """,
                    (hours,),
                )

                results = cursor.fetchall()
                total = sum(r[1] for r in results)
                success_count = next((r[1] for r in results if r[0] == "success"), 0)
                error_count = next((r[1] for r in results if r[0] == "error"), 0)
                avg_duration = (
                    next((r[2] for r in results if r[0] == "success"), 0) or 0
                )

                success_rate = success_count / total if total > 0 else 0

                # Get last success/error timestamps
                cursor = conn.execute(
                    """
                    SELECT timestamp FROM pipeline_metrics
                    WHERE status = 'success'
                    ORDER BY timestamp DESC LIMIT 1
                    """
                )
                last_success = cursor.fetchone()

                cursor = conn.execute(
                    """
                    SELECT timestamp FROM pipeline_metrics
                    WHERE status = 'error'
                    ORDER BY timestamp DESC LIMIT 1
                    """
                )
                last_error = cursor.fetchone()

                return {
                    "success_rate": round(success_rate, 3),
                    "avg_duration_sec": round(avg_duration, 2),
                    "errors_count": error_count,
                    "total_executions": total,
                    "last_success": last_success[0] if last_success else None,
                    "last_error": last_error[0] if last_error else None,
                }
        except Exception as e:
            logger.error(f"Failed to get recent metrics: {e}")
            return {
                "success_rate": 0.0,
                "avg_duration_sec": 0.0,
                "errors_count": 0,
                "total_executions": 0,
                "last_success": None,
                "last_error": None,
            }

    def get_daily_summary(self, date: str) -> dict:
        """Get summary metrics for a specific date."""
        try:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    """
                    SELECT COUNT(*), SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END),
                           SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END),
                           AVG(duration_sec)
                    FROM pipeline_metrics
                    WHERE execution_date = ?
                    """,
                    (date,),
                )

                row = cursor.fetchone()
                total, successes, failures, avg_duration = (
                    row[0] or 0,
                    row[1] or 0,
                    row[2] or 0,
                    row[3] or 0.0,
                )

                # Get list of tickers processed
                cursor = conn.execute(
                    """
                    SELECT DISTINCT ticker FROM pipeline_metrics
                    WHERE execution_date = ?
                    ORDER BY ticker
                    """,
                    (date,),
                )

                tickers = [row[0] for row in cursor.fetchall()]

                return {
                    "date": date,
                    "executions": total,
                    "successes": successes,
                    "failures": failures,
                    "avg_duration_sec": round(avg_duration, 2),
                    "tickers_processed": tickers,
                }
        except Exception as e:
            logger.error(f"Failed to get daily summary for {date}: {e}")
            return {
                "date": date,
                "executions": 0,
                "successes": 0,
                "failures": 0,
                "avg_duration_sec": 0.0,
                "tickers_processed": [],
            }
