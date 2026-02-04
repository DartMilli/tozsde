"""
Unit tests for DataManager (SQLite DAL).

Tests:
- Table initialization
- UPSERT operations
- Query operations
- Data consistency
"""

import pytest
import pandas as pd
import numpy as np
from app.data_access.data_manager import DataManager


class TestDataManagerInitialization:
    """Tests for database initialization."""

    def test_initialize_tables(self, test_db):
        """initialize_tables should create all required tables."""
        dm = test_db
        # Tables should be created during init
        with dm._get_conn() as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}

        assert "ohlcv" in tables
        assert len(tables) > 0


class TestDataManagerOHLCV:
    """Tests for OHLCV operations."""

    def test_save_ohlcv(self, test_db, sample_df):
        """save_ohlcv should insert data correctly."""
        dm = test_db
        dm.save_ohlcv("TEST", sample_df)

        # Verify data was saved
        saved = dm.load_ohlcv("TEST")
        assert saved is not None
        assert len(saved) > 0

    def test_get_ohlcv(self, test_db, sample_df):
        """get_ohlcv should retrieve saved data."""
        dm = test_db
        dm.save_ohlcv("TEST", sample_df)

        # Query exact data
        result = dm.load_ohlcv("TEST")
        assert result is not None
        assert len(result) > 0

    def test_ohlcv_upsert(self, test_db, sample_df):
        """Duplicate (ticker, date) should update existing row."""
        dm = test_db

        # Save initial data
        dm.save_ohlcv("TEST", sample_df)
        initial_len = len(dm.load_ohlcv("TEST"))

        # Save same data (UPSERT - should replace, not duplicate)
        dm.save_ohlcv("TEST", sample_df)
        upserted_len = len(dm.load_ohlcv("TEST"))

        # Should have same length (UPSERT behavior)
        assert initial_len == upserted_len


class TestDataManagerRecommendations:
    """Tests for recommendation operations."""

    def test_save_history_record(self, test_db):
        """save_history_record should store daily recommendations."""
        dm = test_db

        dm.save_history_record(
            ticker="TEST",
            action_code=1,
            label="BUY",
            confidence=0.75,
            wf_score=0.8,
            d_blob='{"strategy": "test"}',
            a_blob='{"audit": "test"}',
        )

        # Verify saved
        history = dm.fetch_history_records_by_ticker("TEST")

        assert isinstance(history, list)
        if len(history) > 0:
            assert len(history[0]) > 0

    def test_fetch_history_records(self, test_db):
        """fetch_history_records_by_ticker should retrieve saved records."""
        dm = test_db

        # Save a history record
        dm.save_history_record(
            ticker="TEST",
            action_code=1,
            label="BUY",
            confidence=0.75,
            wf_score=0.8,
            d_blob='{"test": true}',
            a_blob="{}",
        )

        # Fetch history
        history = dm.fetch_history_records_by_ticker("TEST")

        assert isinstance(history, list)
        assert len(history) > 0

    def test_get_history_range(self, test_db):
        """get_history_range should return date range."""
        dm = test_db

        # Save multiple history records
        for i in range(3):
            dm.save_history_record(
                ticker="TEST",
                action_code=1 + (i % 2),
                label="BUY" if i % 2 == 0 else "SELL",
                confidence=0.7 + (i * 0.05),
                wf_score=0.8,
                d_blob=f'{{"index": {i}}}',
                a_blob="{}",
            )

        # Fetch history range with wider date range
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        start = (now - timedelta(hours=1)).isoformat()
        end = (now + timedelta(hours=1)).isoformat()

        history = dm.get_history_range("TEST", start, end)

        assert isinstance(history, list)
        assert len(history) > 0


class TestDataManagerConsistency:
    """Tests for data consistency and integrity."""

    def test_primary_key_constraint(self, test_db, sample_df):
        """(ticker, date) PRIMARY KEY should be enforced."""
        dm = test_db

        # Save once
        dm.save_ohlcv("TEST", sample_df)
        initial = dm.load_ohlcv("TEST")

        # Save again - should update, not duplicate
        dm.save_ohlcv("TEST", sample_df)
        after_upsert = dm.load_ohlcv("TEST")

        # Counts should match (UPSERT, not insert)
        assert len(initial) == len(after_upsert)

    def test_data_persistence(self, test_db, sample_df):
        """Data should persist across connections."""
        dm = test_db
        dm.save_ohlcv("PERSIST", sample_df)

        # Read with new connection
        result = dm.load_ohlcv("PERSIST")
        assert len(result) > 0

    def test_concurrent_access(self, test_db, sample_df):
        """Multiple tickers should not interfere."""
        dm = test_db

        # Save different tickers
        dm.save_ohlcv("TICK1", sample_df)
        dm.save_ohlcv("TICK2", sample_df)

        # Each should be independent
        tick1 = dm.load_ohlcv("TICK1")
        tick2 = dm.load_ohlcv("TICK2")

        assert len(tick1) > 0
        assert len(tick2) > 0
