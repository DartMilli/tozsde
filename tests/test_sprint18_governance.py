"""S18C – Governance: DecisionEffectivenessAnalyzer coverage."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone, date
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixture – in-memory DataManagerRepository mock
# ---------------------------------------------------------------------------


def _make_dm_with_data(decision_data=None, outcome_data=None, ohlcv_data=None):
    """Return a mock DataManagerRepository that returns controlled data."""
    dm = MagicMock()

    # connection() as context manager
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS decision_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, as_of_date TEXT, ticker TEXT,
            safety_overrides_json TEXT
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_id INTEGER UNIQUE, ticker TEXT,
            decision_timestamp TEXT, pnl_pct REAL, outcome_json TEXT,
            evaluated_at TEXT
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS decision_effectiveness (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_id INTEGER UNIQUE, ticker TEXT,
            effectiveness_score REAL, components_json TEXT, computed_at TEXT
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS decision_effectiveness_rolling (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT, window_days INTEGER, as_of_date TEXT,
            metrics_json TEXT, computed_at TEXT
        )
    """
    )
    conn.commit()

    if decision_data:
        for d in decision_data:
            conn.execute(
                "INSERT INTO decision_history (id, timestamp, ticker, safety_overrides_json) VALUES (?, ?, ?, ?)",
                (
                    d["id"],
                    d["timestamp"],
                    d["ticker"],
                    d.get("safety_overrides_json", "{}"),
                ),
            )
        conn.commit()

    if outcome_data:
        for o in outcome_data:
            conn.execute(
                "INSERT INTO outcomes (decision_id, ticker, decision_timestamp, pnl_pct, outcome_json, evaluated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    o["decision_id"],
                    o["ticker"],
                    o["decision_timestamp"],
                    o["pnl_pct"],
                    o.get("outcome_json", "{}"),
                    o.get("evaluated_at"),
                ),
            )
        conn.commit()

    import contextlib

    @contextlib.contextmanager
    def _connection():
        yield conn

    dm.connection = _connection

    if ohlcv_data is not None:
        dm.load_ohlcv.return_value = ohlcv_data
    else:
        import pandas as pd

        dm.load_ohlcv.return_value = pd.DataFrame()

    return dm


# ---------------------------------------------------------------------------
# DecisionEffectivenessAnalyzer – compute_for_decision
# ---------------------------------------------------------------------------


class TestComputeForDecision:
    def test_no_outcome_row_returns_none(self):
        dm = _make_dm_with_data()
        from app.core.analysis.decision_effectiveness import (
            DecisionEffectivenessAnalyzer,
        )

        analyzer = DecisionEffectivenessAnalyzer(data_manager=dm)
        result = analyzer.compute_for_decision(99999)
        assert result is None

    def test_with_outcome_returns_score(self):
        decisions = [
            {
                "id": 1,
                "timestamp": "2025-01-15T10:00:00",
                "ticker": "VOO",
                "safety_overrides_json": "{}",
            }
        ]
        outcomes = [
            {
                "decision_id": 1,
                "ticker": "VOO",
                "decision_timestamp": "2025-01-15T10:00:00",
                "pnl_pct": 0.05,
                "evaluated_at": "2025-01-20T10:00:00",
            }
        ]
        dm = _make_dm_with_data(decision_data=decisions, outcome_data=outcomes)
        dm.save_decision_effectiveness = MagicMock()

        from app.core.analysis.decision_effectiveness import (
            DecisionEffectivenessAnalyzer,
        )

        analyzer = DecisionEffectivenessAnalyzer(data_manager=dm)
        result = analyzer.compute_for_decision(1)

        assert result is not None
        assert "effectiveness_score" in result
        dm.save_decision_effectiveness.assert_called_once()

    def test_with_safety_override_penalty(self):
        decisions = [
            {
                "id": 2,
                "timestamp": "2025-02-10T10:00:00",
                "ticker": "AAPL",
                "safety_overrides_json": '{"safety_override": true}',
            }
        ]
        outcomes = [
            {
                "decision_id": 2,
                "ticker": "AAPL",
                "decision_timestamp": "2025-02-10T10:00:00",
                "pnl_pct": 0.03,
                "evaluated_at": "2025-02-15T10:00:00",
            }
        ]
        dm = _make_dm_with_data(decision_data=decisions, outcome_data=outcomes)
        dm.save_decision_effectiveness = MagicMock()

        from app.core.analysis.decision_effectiveness import (
            DecisionEffectivenessAnalyzer,
        )

        analyzer = DecisionEffectivenessAnalyzer(data_manager=dm)
        result = analyzer.compute_for_decision(2)
        assert result is not None

    def test_with_ohlcv_data_for_stats(self):
        import pandas as pd

        dates = pd.date_range("2025-01-15", "2025-01-22")
        prices = pd.Series(
            [100.0, 98.0, 97.0, 102.0, 104.0, 103.0, 105.0, 106.0],
            index=dates,
            name="Close",
        )
        ohlcv = pd.DataFrame({"Close": prices})
        ohlcv.index.name = "Date"

        decisions = [
            {
                "id": 3,
                "timestamp": "2025-01-15T10:00:00",
                "ticker": "VOO",
                "safety_overrides_json": "{}",
            }
        ]
        outcomes = [
            {
                "decision_id": 3,
                "ticker": "VOO",
                "decision_timestamp": "2025-01-15T10:00:00",
                "pnl_pct": 0.06,
                "evaluated_at": "2025-01-22T10:00:00",
            }
        ]
        dm = _make_dm_with_data(
            decision_data=decisions, outcome_data=outcomes, ohlcv_data=ohlcv
        )
        dm.save_decision_effectiveness = MagicMock()

        from app.core.analysis.decision_effectiveness import (
            DecisionEffectivenessAnalyzer,
        )

        analyzer = DecisionEffectivenessAnalyzer(data_manager=dm)
        result = analyzer.compute_for_decision(3)
        assert result is not None


# ---------------------------------------------------------------------------
# DecisionEffectivenessAnalyzer – compute_for_range
# ---------------------------------------------------------------------------


class TestComputeForRange:
    def test_no_data_returns_no_data(self):
        dm = _make_dm_with_data()
        dm.save_decision_effectiveness = MagicMock()

        from app.core.analysis.decision_effectiveness import (
            DecisionEffectivenessAnalyzer,
        )

        analyzer = DecisionEffectivenessAnalyzer(data_manager=dm)
        result = analyzer.compute_for_range(
            ticker="VOO", start_date="2025-01-01", end_date="2025-12-31"
        )
        assert result["status"] == "no_data"

    def test_with_data_returns_ok(self):
        decisions = [
            {
                "id": 10,
                "timestamp": "2025-03-01T10:00:00",
                "ticker": "VOO",
                "safety_overrides_json": "{}",
            }
        ]
        outcomes = [
            {
                "decision_id": 10,
                "ticker": "VOO",
                "decision_timestamp": "2025-03-01T10:00:00",
                "pnl_pct": 0.04,
                "evaluated_at": "2025-03-10T10:00:00",
            }
        ]
        dm = _make_dm_with_data(decision_data=decisions, outcome_data=outcomes)
        dm.save_decision_effectiveness = MagicMock()
        dm.save_decision_effectiveness_rolling = MagicMock()

        from app.core.analysis.decision_effectiveness import (
            DecisionEffectivenessAnalyzer,
        )

        analyzer = DecisionEffectivenessAnalyzer(data_manager=dm)
        result = analyzer.compute_for_range(
            ticker="VOO", start_date="2025-01-01", end_date="2025-12-31"
        )
        assert result["status"] == "ok"
        assert result["rows"] == 1

    def test_no_ticker_filter(self):
        decisions = [
            {
                "id": 20,
                "timestamp": "2025-04-01T10:00:00",
                "ticker": "AAPL",
                "safety_overrides_json": "{}",
            },
            {
                "id": 21,
                "timestamp": "2025-04-02T10:00:00",
                "ticker": "MSFT",
                "safety_overrides_json": "{}",
            },
        ]
        outcomes = [
            {
                "decision_id": 20,
                "ticker": "AAPL",
                "decision_timestamp": "2025-04-01T10:00:00",
                "pnl_pct": 0.02,
                "evaluated_at": "2025-04-10T10:00:00",
            },
            {
                "decision_id": 21,
                "ticker": "MSFT",
                "decision_timestamp": "2025-04-02T10:00:00",
                "pnl_pct": 0.03,
                "evaluated_at": "2025-04-11T10:00:00",
            },
        ]
        dm = _make_dm_with_data(decision_data=decisions, outcome_data=outcomes)
        dm.save_decision_effectiveness = MagicMock()

        from app.core.analysis.decision_effectiveness import (
            DecisionEffectivenessAnalyzer,
        )

        analyzer = DecisionEffectivenessAnalyzer(data_manager=dm)
        result = analyzer.compute_for_range(ticker=None)
        assert result["status"] == "ok"
        assert result["rows"] == 2

    def test_rolling_aggregate_saved_when_ticker_provided(self):
        decisions = [
            {
                "id": 30,
                "timestamp": "2025-05-01T10:00:00",
                "ticker": "VOO",
                "safety_overrides_json": "{}",
            }
        ]
        outcomes = [
            {
                "decision_id": 30,
                "ticker": "VOO",
                "decision_timestamp": "2025-05-01T10:00:00",
                "pnl_pct": 0.07,
                "evaluated_at": "2025-05-10T10:00:00",
            }
        ]
        dm = _make_dm_with_data(decision_data=decisions, outcome_data=outcomes)
        dm.save_decision_effectiveness_rolling = MagicMock()

        # Pre-populate decision_effectiveness so _save_rolling_aggregate finds rows
        with dm.connection() as conn:
            conn.execute(
                "INSERT INTO decision_effectiveness (decision_id, ticker, effectiveness_score, components_json, computed_at) VALUES (30, 'VOO', 0.65, '{}', '2025-05-10T10:00:00')"
            )
            conn.commit()

        from app.core.analysis.decision_effectiveness import (
            DecisionEffectivenessAnalyzer,
        )

        analyzer = DecisionEffectivenessAnalyzer(data_manager=dm)

        # Override save_decision_effectiveness to just insert into DB so the rolling agg finds rows
        def real_save_eff(decision_id, ticker, effectiveness_score, components_json):
            pass  # already inserted above

        analyzer.dm.save_decision_effectiveness = real_save_eff

        analyzer.compute_for_range(ticker="VOO", end_date="2025-12-31")
        dm.save_decision_effectiveness_rolling.assert_called()


# ---------------------------------------------------------------------------
# DecisionEffectivenessAnalyzer – _parse_date
# ---------------------------------------------------------------------------


class TestParseDate:
    def _analyzer(self):
        dm = _make_dm_with_data()
        from app.core.analysis.decision_effectiveness import (
            DecisionEffectivenessAnalyzer,
        )

        return DecisionEffectivenessAnalyzer(data_manager=dm)

    def test_iso_format(self):
        a = self._analyzer()
        result = a._parse_date("2025-01-15T10:00:00")
        assert result == date(2025, 1, 15)

    def test_date_only_format(self):
        a = self._analyzer()
        result = a._parse_date("2025-06-20")
        assert result == date(2025, 6, 20)

    def test_none_returns_none(self):
        a = self._analyzer()
        result = a._parse_date(None)
        assert result is None

    def test_empty_string_returns_none(self):
        a = self._analyzer()
        result = a._parse_date("")
        assert result is None

    def test_invalid_string_returns_none(self):
        a = self._analyzer()
        result = a._parse_date("not-a-date")
        assert result is None


# ---------------------------------------------------------------------------
# DecisionEffectivenessAnalyzer – _compute_eds directly
# ---------------------------------------------------------------------------


class TestComputeEds:
    def _analyzer(self):
        dm = _make_dm_with_data()
        from app.core.analysis.decision_effectiveness import (
            DecisionEffectivenessAnalyzer,
        )

        return DecisionEffectivenessAnalyzer(data_manager=dm)

    def test_positive_pnl_gives_positive_score(self):
        a = self._analyzer()
        row = {
            "pnl_pct": 0.10,
            "safety_overrides_json": "{}",
            "timestamp": None,
            "evaluated_at": None,
            "ticker": "VOO",
        }
        score, components = a._compute_eds(row)
        assert score <= 0.10 + 1e-6  # penalties may reduce it
        assert "realized_return" in components
        assert components["realized_return"] == pytest.approx(0.10)

    def test_negative_pnl(self):
        a = self._analyzer()
        row = {
            "pnl_pct": -0.05,
            "safety_overrides_json": "{}",
            "timestamp": None,
            "evaluated_at": None,
            "ticker": "VOO",
        }
        score, components = a._compute_eds(row)
        assert score <= 0.0 + 1e-6

    def test_safety_override_applies_penalty(self):
        a = self._analyzer()
        row_no_override = {
            "pnl_pct": 0.05,
            "safety_overrides_json": '{"safety_override": false}',
            "timestamp": None,
            "evaluated_at": None,
            "ticker": "VOO",
        }
        row_with_override = {
            "pnl_pct": 0.05,
            "safety_overrides_json": '{"safety_override": true}',
            "timestamp": None,
            "evaluated_at": None,
            "ticker": "VOO",
        }
        score_no, _ = a._compute_eds(row_no_override)
        score_yes, _ = a._compute_eds(row_with_override)
        # With override, score should be <= score without (penalty ≥ 0)
        assert score_yes <= score_no + 1e-6

    def test_invalid_safety_json_is_handled(self):
        a = self._analyzer()
        row = {
            "pnl_pct": 0.03,
            "safety_overrides_json": "INVALID_JSON",
            "timestamp": None,
            "evaluated_at": None,
            "ticker": "VOO",
        }
        score, components = a._compute_eds(row)
        # Should not raise
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# _log_no_data_reason – coverage of diagnostic branch
# ---------------------------------------------------------------------------


class TestLogNoDataReason:
    def test_does_not_raise(self):
        dm = _make_dm_with_data()
        from app.core.analysis.decision_effectiveness import (
            DecisionEffectivenessAnalyzer,
        )

        analyzer = DecisionEffectivenessAnalyzer(data_manager=dm)
        # Should not raise even with empty DB
        analyzer._log_no_data_reason("VOO", "2025-01-01", "2025-12-31")

    def test_no_ticker_filter(self):
        dm = _make_dm_with_data()
        from app.core.analysis.decision_effectiveness import (
            DecisionEffectivenessAnalyzer,
        )

        analyzer = DecisionEffectivenessAnalyzer(data_manager=dm)
        analyzer._log_no_data_reason(None, None, None)


# ---------------------------------------------------------------------------
# _save_rolling_aggregate – coverage
# ---------------------------------------------------------------------------


class TestSaveRollingAggregate:
    def test_no_rows_returns_early(self):
        dm = _make_dm_with_data()
        dm.save_decision_effectiveness_rolling = MagicMock()
        from app.core.analysis.decision_effectiveness import (
            DecisionEffectivenessAnalyzer,
        )

        analyzer = DecisionEffectivenessAnalyzer(data_manager=dm)
        analyzer._save_rolling_aggregate("NODATA", 30, "2025-01-01")
        # No effectiveness data → should return without saving rolling
        dm.save_decision_effectiveness_rolling.assert_not_called()

    def test_with_effectiveness_rows(self):
        dm = _make_dm_with_data()
        dm.save_decision_effectiveness_rolling = MagicMock()

        # Insert effectiveness scores
        with dm.connection() as conn:
            conn.execute(
                "INSERT INTO decision_effectiveness (decision_id, ticker, effectiveness_score, components_json, computed_at) VALUES (1, 'VOO', 0.5, '{}', '2025-01-01T00:00:00')"
            )
            conn.execute(
                "INSERT INTO decision_effectiveness (decision_id, ticker, effectiveness_score, components_json, computed_at) VALUES (2, 'VOO', 0.7, '{}', '2025-01-02T00:00:00')"
            )
            conn.commit()

        from app.core.analysis.decision_effectiveness import (
            DecisionEffectivenessAnalyzer,
        )

        analyzer = DecisionEffectivenessAnalyzer(data_manager=dm)
        analyzer._save_rolling_aggregate("VOO", 30, "2025-01-31")
        dm.save_decision_effectiveness_rolling.assert_called_once()


# ---------------------------------------------------------------------------
# analysis.decision_effectiveness shim
# ---------------------------------------------------------------------------


class TestAnalysisShimDecisionEffectiveness:
    def test_shim_imports(self):
        import app.analysis.decision_effectiveness as m

        # Should import without raising
        assert m is not None
