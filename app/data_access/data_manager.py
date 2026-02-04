import sqlite3
import pandas as pd
import logging
import json
from datetime import datetime, timezone
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
                    action_code INTEGER,
                    action_label TEXT,
                    confidence REAL,
                    wf_score REAL,
                    decision_blob TEXT,
                    audit_blob TEXT
                )
            """
            )
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

            # Indexek a teljesítmény javítására
            cur.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker ON ohlcv(ticker)")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_dh_ticker_ts ON decision_history(ticker, timestamp)"
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
        df_to_save.columns = [c.lower() for c in df_to_save.columns]

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

    # --- DÖNTÉSI ELŐZMÉNYEK (History & Audit) ---

    def save_history_record(
        self, ticker, action_code, label, confidence, wf_score, d_blob, a_blob
    ):
        """Enkapszulált mentés a history táblába."""
        query = """
            INSERT INTO decision_history 
            (timestamp, ticker, action_code, action_label, confidence, wf_score, decision_blob, audit_blob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._get_conn() as conn:
            conn.execute(
                query,
                (
                    datetime.now(timezone.utc).isoformat(),
                    ticker,
                    action_code,
                    label,
                    confidence,
                    wf_score,
                    d_blob,
                    a_blob,
                ),
            )
            conn.commit()

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

        for t in tickers:
            df = self.load_ohlcv(t)
            if not df.empty:
                # Időbeli vágás
                if ref_date:
                    df = df[df.index <= pd.Timestamp(ref_date)]

                # Csak a vágás után vesszük a végét!
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
                       confidence, wf_score, decision_blob, audit_blob
                FROM decision_history
                WHERE ticker = ?
                ORDER BY timestamp ASC
                """,
                (ticker,),
            ).fetchall()
        out = []
        for ts, tkr, ac, al, conf, wf, d_blob, a_blob in rows:
            out.append(
                {
                    "timestamp": ts,
                    "ticker": tkr,
                    "decision": json.loads(d_blob) if d_blob else {},
                    "audit": json.loads(a_blob) if a_blob else {},
                }
            )
        return out

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
        """
        Placeholder: project snapshot does not define a persistent 'outcomes' table.
        Return empty list to avoid blocking SafetyRuleEngine.
        """
        return []

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
        # Mivel JSON-ben van az outcome, LIKE-kal keresünk (nem szép, de sémaváltás nélkül ez van)
        query = """
            SELECT id, timestamp, ticker, audit_blob 
            FROM decision_history 
            WHERE action_code = 1 
              AND audit_blob NOT LIKE '%"outcome"%'
            ORDER BY timestamp DESC
            LIMIT ?
        """
        with self._get_conn() as conn:
            return conn.execute(query, (limit,)).fetchall()

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
