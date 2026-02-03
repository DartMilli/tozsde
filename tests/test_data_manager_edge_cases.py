"""Edge case tests for DataManager advanced methods."""

import json
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from app.data_access.data_manager import DataManager


def _make_ohlcv_df(start_date, days, base=100.0, step=1.0):
    dates = pd.date_range(start=start_date, periods=days, freq="D")
    prices = base + step * np.arange(days)
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": prices + 1,
            "Low": prices - 1,
            "Close": prices,
            "Volume": np.full(days, 1000.0),
        },
        index=dates,
    )
    df.index.name = "date"
    return df


def test_get_market_regime_is_bear_false_when_uptrend(test_db):
    dm = test_db
    df = _make_ohlcv_df("2025-01-01", 210, base=100.0, step=1.0)
    dm.save_ohlcv("SPY", df)

    assert bool(dm.get_market_regime_is_bear(benchmark="SPY", period=200)) is False


def test_get_market_regime_is_bear_true_when_downtrend(test_db):
    dm = test_db
    df = _make_ohlcv_df("2025-01-01", 210, base=300.0, step=-1.0)
    dm.save_ohlcv("SPY", df)

    assert bool(dm.get_market_regime_is_bear(benchmark="SPY", period=200)) is True


def test_get_correlation_matrix_multiple_tickers(test_db):
    dm = test_db
    df1 = _make_ohlcv_df("2025-01-01", 30, base=100.0, step=1.0)
    df2 = _make_ohlcv_df("2025-01-01", 30, base=200.0, step=2.0)
    dm.save_ohlcv("AAA", df1)
    dm.save_ohlcv("BBB", df2)

    corr = dm.get_correlation_matrix(["AAA", "BBB"], lookback_days=20)

    assert not corr.empty
    assert set(corr.columns) == {"AAA", "BBB"}


def test_recommendations_roundtrip(test_db):
    dm = test_db
    dm.log_recommendation("AAPL", "BUY", 0.75, params={"x": 1})

    recs = dm.get_today_recommendations()

    assert len(recs) == 1
    assert recs[0]["ticker"] == "AAPL"
    assert recs[0]["signal"] == "BUY"


def test_get_ticker_historical_recommendations(test_db):
    dm = test_db
    today = datetime.now().strftime("%Y-%m-%d")
    with dm._get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO recommendations (date, ticker, signal, confidence, params) VALUES (?, ?, ?, ?, ?)",
            (today, "MSFT", "SELL", 0.6, json.dumps({})),
        )
        conn.commit()

    recs = dm.get_ticker_historical_recommendations("MSFT", today, today)
    assert len(recs) == 1
    assert recs[0]["ticker"] == "MSFT"


def test_save_and_get_market_data(test_db):
    dm = test_db
    dates = pd.date_range(start="2025-01-01", periods=3, freq="D")
    df = pd.DataFrame({"Close": [10.0, 11.0, 12.0]}, index=dates)
    df.index.name = "Date"

    dm.save_market_data("^VIX", df)

    rows = dm.get_market_data("^VIX", days=2)
    assert len(rows) == 2
    assert rows[0][0] == "2025-01-03"


def test_fetch_history_records_parses_json(test_db):
    dm = test_db
    dm.save_history_record(
        ticker="AAPL",
        action_code=1,
        label="BUY",
        confidence=0.7,
        wf_score=0.8,
        d_blob=json.dumps({"k": "v"}),
        a_blob=json.dumps({"audit": True}),
    )

    records = dm.fetch_history_records_by_ticker("AAPL")

    assert len(records) == 1
    assert records[0]["decision"]["k"] == "v"
    assert records[0]["audit"]["audit"] is True


def test_get_unevaluated_buy_decisions_and_update(test_db):
    dm = test_db
    with dm._get_conn() as conn:
        conn.execute(
            "INSERT INTO decision_history (timestamp, ticker, action_code, action_label, confidence, wf_score, decision_blob, audit_blob) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.utcnow().isoformat(),
                "AAPL",
                1,
                "BUY",
                0.7,
                0.8,
                json.dumps({"d": 1}),
                json.dumps({}),
            ),
        )
        conn.commit()

    rows = dm.get_unevaluated_buy_decisions(limit=10)
    assert len(rows) == 1
    row_id = rows[0][0]

    dm.update_history_audit(row_id, json.dumps({"outcome": "win"}))

    rows_after = dm.get_unevaluated_buy_decisions(limit=10)
    assert len(rows_after) == 0


def test_pipeline_metrics_logging_and_summary(test_db):
    dm = test_db

    assert dm.log_pipeline_execution("AAPL", "success", 1.2) is True
    assert dm.log_pipeline_execution("AAPL", "error", 0.5, "oops") is True
    assert dm.log_backtest_execution("AAPL", 0.5, 10, 1.2) is True

    recent = dm.get_recent_metrics(hours=24)
    assert recent["total_executions"] >= 2
    assert recent["errors_count"] >= 1

    today = datetime.utcnow().strftime("%Y-%m-%d")
    summary = dm.get_daily_summary(today)
    assert summary["executions"] >= 2
    assert "AAPL" in summary["tickers_processed"]
