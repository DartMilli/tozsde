import sqlite3
from pathlib import Path
import datetime

import app.utils.router as rtr

DB_PATH = rtr.PORTFOLIO_DB_PATH


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS portfolio (
            date TEXT PRIMARY KEY,
            ticker TEXT,
            position TEXT,
            quantity INTEGER,
            price REAL,
            value REAL
        )
    """
    )
    conn.commit()
    conn.close()


def record_transaction(ticker: str, position: str, quantity: int, price: float):
    value = quantity * price
    date = datetime.date.today().isoformat()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "REPLACE INTO portfolio (date, ticker, position, quantity, price, value) VALUES (?, ?, ?, ?, ?, ?)",
        (date, ticker, position, quantity, price, value),
    )
    conn.commit()
    conn.close()


def get_latest_portfolio():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM portfolio ORDER BY date DESC LIMIT 10")
    rows = c.fetchall()
    conn.close()
    return rows
