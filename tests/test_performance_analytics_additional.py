"""Additional tests for performance_analytics to improve coverage."""

import math
import sqlite3
from datetime import datetime, timedelta

from app.reporting.performance_analytics import PerformanceAnalytics


def test_rolling_metrics_not_enough_data():
    analytics = PerformanceAnalytics()
    returns = [0.01] * 5
    dates = [datetime(2025, 1, i + 1) for i in range(5)]

    rolling = analytics.calculate_rolling_metrics(returns, dates, window_days=10)

    assert rolling.returns == []
    assert rolling.volatilities == []
    assert rolling.sharpe_ratios == []


def test_sortino_ratio_no_negative_returns():
    analytics = PerformanceAnalytics(risk_free_rate=0.0)
    returns = [0.01, 0.02, 0.01]
    dates = [datetime(2025, 1, i + 1) for i in range(3)]

    sortino = analytics._calculate_sortino_ratio(returns, dates)

    assert math.isinf(sortino)


def test_trade_statistics_and_extremes():
    analytics = PerformanceAnalytics()
    trades = [
        {"pnl": 10},
        {"pnl": -5},
        {"pnl": 7},
    ]

    win_rate, profit_factor, total, winning, losing = analytics._calculate_trade_statistics(trades)
    avg_win, avg_loss, best, worst = analytics._calculate_trade_extremes(trades)

    assert total == 3
    assert winning == 2
    assert losing == 1
    assert win_rate > 0
    assert profit_factor > 0
    assert avg_win > 0
    assert avg_loss < 0
    assert best == 10
    assert worst == -5


def test_load_returns_from_db(tmp_path):
    db_path = tmp_path / "perf.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE portfolio_equity (date TEXT, equity_value REAL)")

    today = datetime.now().date()
    rows = [
        (today.isoformat(), 100.0),
        ((today + timedelta(days=1)).isoformat(), 110.0),
    ]
    cursor.executemany("INSERT INTO portfolio_equity (date, equity_value) VALUES (?, ?)", rows)
    conn.commit()
    conn.close()

    analytics = PerformanceAnalytics(db_path=str(db_path))
    returns, dates = analytics.load_returns_from_db(days_back=10)

    assert len(returns) == 1
    assert len(dates) == 1
    assert abs(returns[0] - 0.1) < 1e-6
