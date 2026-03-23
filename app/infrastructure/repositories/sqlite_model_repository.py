from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from app.config.build_settings import build_settings
from app.core.ports.imodel_repository import IModelRepository


class SqliteModelRepository(IModelRepository):
    def __init__(self, settings=None):
        cfg = settings or build_settings()
        db_path = getattr(cfg, "DB_PATH", None)
        if not db_path:
            raise RuntimeError("DB_PATH is not configured")
        self.db_path = str(db_path)
        self._initialize_tables()

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

    def _initialize_tables(self) -> None:
        with self.connection() as conn:
            conn.execute(
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
            conn.execute(
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
            conn.commit()

    def save_model(self, model_info: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        model_id = model_info.get("model_id")
        if not model_id:
            raise ValueError("model_id is required")

        metadata = model_info.get("metadata_json")
        if metadata is None:
            metadata_obj = model_info.get("metadata")
            metadata = json.dumps(metadata_obj, default=str) if metadata_obj else "{}"

        query = """
            INSERT OR REPLACE INTO model_registry
            (model_id, ticker, model_type, wf_score, model_path, status, created_at, updated_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self.connection() as conn:
            conn.execute(
                query,
                (
                    model_id,
                    model_info.get("ticker"),
                    model_info.get("model_type"),
                    model_info.get("wf_score"),
                    model_info.get("model_path"),
                    model_info.get("status", "candidate"),
                    model_info.get("created_at", now),
                    model_info.get("updated_at", now),
                    metadata,
                ),
            )
            conn.commit()

    def fetch_model(self, model_id: str) -> dict:
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT model_id, ticker, model_type, wf_score, model_path, status,
                       created_at, updated_at, metadata_json
                FROM model_registry
                WHERE model_id = ?
                """,
                (model_id,),
            ).fetchone()

        if not row:
            return {}

        return {
            "model_id": row[0],
            "ticker": row[1],
            "model_type": row[2],
            "wf_score": row[3],
            "model_path": row[4],
            "status": row[5],
            "created_at": row[6],
            "updated_at": row[7],
            "metadata": json.loads(row[8]) if row[8] else {},
        }

    def fetch_models_for_ticker(self, ticker: str) -> list[dict]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT model_id, ticker, model_type, wf_score, model_path, status,
                       created_at, updated_at, metadata_json
                FROM model_registry
                WHERE ticker = ?
                ORDER BY updated_at DESC
                """,
                (ticker,),
            ).fetchall()

        return [
            {
                "model_id": row[0],
                "ticker": row[1],
                "model_type": row[2],
                "wf_score": row[3],
                "model_path": row[4],
                "status": row[5],
                "created_at": row[6],
                "updated_at": row[7],
                "metadata": json.loads(row[8]) if row[8] else {},
            }
            for row in rows
        ]

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
        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO model_registry
                (model_id, ticker, model_type, wf_score, model_path, status, created_at, updated_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
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
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE model_registry
                SET status = ?, updated_at = ?
                WHERE model_id = ?
                """,
                (status, datetime.now(timezone.utc).isoformat(), model_id),
            )
            conn.commit()

    def fetch_active_models(self, ticker: str, limit: int = 3) -> List[Dict]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT model_id, model_path, wf_score, model_type
                FROM model_registry
                WHERE ticker = ? AND status = 'active'
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (ticker, limit),
            ).fetchall()
        return [
            {
                "model_id": model_id,
                "model_path": model_path,
                "wf_score": wf_score,
                "model_type": model_type,
            }
            for model_id, model_path, wf_score, model_type in rows
        ]

    def save_model_trust_metrics(
        self,
        model_path: str,
        ticker: str,
        trust_weight: float,
        metrics_json: str,
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO model_trust_metrics
                (model_path, ticker, trust_weight, metrics_json, computed_at)
                VALUES (?, ?, ?, ?, ?)
                """,
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
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT model_path, trust_weight, MAX(computed_at)
                FROM model_trust_metrics
                WHERE ticker = ?
                GROUP BY model_path
                """,
                (ticker,),
            ).fetchall()
        return {model_path: (tw or 0.0) for model_path, tw, _ in rows}

    def get_top_models(self, ticker: str, limit: int = 3):
        models = self.fetch_active_models(ticker=ticker, limit=limit)
        if not models:
            return []

        trust = self.fetch_latest_model_trust_weights(ticker=ticker)
        ranked = []
        for model in models:
            model_path = model.get("model_path")
            ranked.append((model_path, trust.get(model_path, 0.0)))

        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:limit]
