import yfinance as yf
from datetime import datetime, timedelta
from app.infrastructure.repositories.sqlite_metrics_repository import (
    SqliteMetricsRepository,
)


def update_macro_context():
    """Letolti a VIX-et es a 13-hetes kincstarjegy hozamot (Risk-free rate)."""
    metrics_repo = SqliteMetricsRepository()
    symbols = {"^VIX": "VOLATILITY", "^IRX": "RISK_FREE_RATE"}

    for sym, category in symbols.items():
        ticker = yf.Ticker(sym)
        df = ticker.history(period="5d")
        if not df.empty:
            for idx, row in df.iterrows():
                metrics_repo.save_metrics(
                    {
                        "symbol": category,
                        "date": idx.strftime("%Y-%m-%d"),
                        "value": row["Close"],
                    }
                )


def get_risk_free_rate():
    """Visszaadja az aktualis evesitett kockazatmentes hozamot (decimalisban)."""
    metrics_repo = SqliteMetricsRepository()
    res = metrics_repo.fetch_metrics("RISK_FREE_RATE", None, None)
    if res and len(res) > 0:
        return res[0]["value"] / 100.0
    return 0.045  # Fallback 4.5%
