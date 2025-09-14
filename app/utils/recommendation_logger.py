import sqlite3
from datetime import date
from typing import List, Dict, Any

import app.utils.router as rtr

DB_PATH = rtr.RECOMMENDATION_DB_PATH  # Path object


def init_db() -> None:
    # Ensure parent dir exists (Path API)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.0,
                UNIQUE(date, ticker)
            );
            '''
        )
        # Helpful indexes
        conn.execute('CREATE INDEX IF NOT EXISTS idx_rec_date ON recommendations(date);')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_rec_ticker ON recommendations(ticker);')


def log_recommendation(ticker: str, recommendation: str, confidence: float) -> None:
    init_db()
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.execute(
            '''
            INSERT INTO recommendations (date, ticker, recommendation, confidence)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(date, ticker) DO UPDATE SET
                recommendation = excluded.recommendation,
                confidence = excluded.confidence
            ;
            ''',
            (date.today().isoformat(), ticker, recommendation, float(confidence)),
        )


def load_today_recommendations() -> List[Dict[str, Any]]:
    init_db()
    today = date.today().isoformat()
    with sqlite3.connect(str(DB_PATH)) as conn:
        cursor = conn.execute(
            '''
            SELECT date, ticker, recommendation, confidence
            FROM recommendations
            WHERE date = ?
            ORDER BY ticker ASC;
            ''',
            (today,),
        )
        return [
            {"date": row[0], "ticker": row[1], "recommendation": row[2], "confidence": row[3]}
            for row in cursor.fetchall()
        ]


def load_recommendations_by_ticker_and_range(ticker: str, start: str, end: str) -> List[Dict[str, Any]]:
    init_db()
    with sqlite3.connect(str(DB_PATH)) as conn:
        cursor = conn.execute(
            '''
            SELECT date, recommendation, confidence
            FROM recommendations
            WHERE ticker = ? AND date BETWEEN ? AND ?
            ORDER BY date ASC;
            ''',
            (ticker, start, end),
        )
        return [
            {"date": row[0], "recommendation": row[1], "confidence": row[2]}
            for row in cursor.fetchall()
        ]


if __name__ == "__main__":
    # Minimal sanity test
    init_db()
    log_recommendation("VOO", "VÉTEL", 0.87)
    print("TODAY:", load_today_recommendations())
    print("HISTORY:", load_recommendations_by_ticker_and_range("VOO", "2025-01-01", "2025-12-31"))
