from typing import Any, Optional
import sqlite3
import pandas as pd
from pathlib import Path

from app.core.ports import IOhlcvRepository
from app.config.build_settings import build_settings


class SqliteOhlcvRepository(IOhlcvRepository):
    def __init__(self, data_manager=None, settings: Optional[Any] = None):
        # Accept either a legacy DataManager or settings for DI
        self._dm = data_manager
        self._settings = settings

    def _resolve_db_path(self) -> str:
        # Prefer DataManager.db_path when available
        if self._dm is not None and hasattr(self._dm, "db_path"):
            return str(getattr(self._dm, "db_path"))
        # Otherwise, resolve from settings or build_settings
        settings = self._settings or build_settings()
        db_path = getattr(settings, "DB_PATH", None)
        if db_path is None:
            # Fallback to a local default DB in the project folder
            return str(Path.cwd() / "data.sqlite")
        return str(db_path)

    def _get_conn(self):
        db = self._resolve_db_path()
        # Ensure parent directory exists
        try:
            Path(db).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return sqlite3.connect(db)

    def load_ohlcv(self, ticker: str, start_date=None) -> Any:
        # If a legacy DataManager was injected and exposes load_ohlcv, prefer it
        if self._dm is not None and hasattr(self._dm, "load_ohlcv"):
            try:
                return self._dm.load_ohlcv(ticker, start_date=start_date)
            except TypeError:
                try:
                    # Some tests/mocks expose positional-only signatures.
                    return self._dm.load_ohlcv(ticker, start_date)
                except TypeError:
                    # And some expose only a single ticker parameter.
                    return self._dm.load_ohlcv(ticker)

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
                df.columns = [c.capitalize() for c in df.columns]
            return df

    def save_ohlcv(self, ticker: str, df: Any) -> None:
        # If a legacy DataManager was injected and exposes save_ohlcv, prefer it
        if self._dm is not None and hasattr(self._dm, "save_ohlcv"):
            try:
                return self._dm.save_ohlcv(ticker=ticker, df=df)
            except TypeError:
                return self._dm.save_ohlcv(ticker, df)
            except Exception:
                # fallback to repository SQL below
                pass

        if df is None or df.empty:
            return

        df_to_save = df.copy()
        if "Ticker" not in df_to_save.columns and "ticker" not in df_to_save.columns:
            df_to_save["ticker"] = ticker

        df_to_save = df_to_save.reset_index()

        def _norm_col(col):
            if isinstance(col, tuple):
                col = col[0]
            return str(col).lower()

        df_to_save.columns = [_norm_col(c) for c in df_to_save.columns]

        with self._get_conn() as conn:
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
