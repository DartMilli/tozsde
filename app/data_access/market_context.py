import yfinance as yf
from datetime import datetime, timedelta
from app.data_access.data_manager import DataManager


def update_macro_context():
    """Letölti a VIX-et és a 13-hetes kincstárjegy hozamot (Risk-free rate)."""
    dm = DataManager()
    symbols = {"^VIX": "VOLATILITY", "^IRX": "RISK_FREE_RATE"}

    for sym, category in symbols.items():
        ticker = yf.Ticker(sym)
        df = ticker.history(period="5d")
        if not df.empty:
            with dm.connection() as conn:
                for idx, row in df.iterrows():
                    conn.execute(
                        """INSERT OR REPLACE INTO market_metadata 
                                 (symbol, date, value) VALUES (?, ?, ?)""",
                        (category, idx.strftime("%Y-%m-%d"), row["Close"]),
                    )


def get_risk_free_rate():
    """Visszaadja az aktuális évesített kockázatmentes hozamot (decimálisban)."""
    dm = DataManager()
    with dm.connection() as conn:
        res = conn.execute(
            "SELECT value FROM market_metadata WHERE symbol='RISK_FREE_RATE' ORDER BY date DESC LIMIT 1"
        ).fetchone()
        return (res[0] / 100.0) if res else 0.045  # Fallback 4.5%
