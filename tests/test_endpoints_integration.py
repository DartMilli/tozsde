"""Admin dashboard endpoint tests using Flask test client."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from dataclasses import replace

import pytest

import app.ui.app as ui_app
from app.ui.app import app
from app.decision.decision_history_analyzer import StrategyStats


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _auth_headers():
    return {"X-Admin-Key": "key"}


def _set_admin_key(monkeypatch, test_settings):
    from app.ui import set_settings as set_ui_settings

    new_settings = replace(test_settings, ADMIN_API_KEY="key")
    set_ui_settings(new_settings)
    monkeypatch.setattr(ui_app, "settings", new_settings)


def test_admin_health_endpoint(monkeypatch, client, test_settings):
    class DummyMetrics:
        def get_health_status(self):
            return {"status": "healthy"}

    _set_admin_key(monkeypatch, test_settings)
    monkeypatch.setattr(
        "app.infrastructure.metrics.get_metrics", lambda: DummyMetrics()
    )

    res = client.get("/admin/health", headers=_auth_headers())
    assert res.status_code == 200
    assert res.get_json()["status"] == "healthy"


def test_performance_summary_endpoint(monkeypatch, client, test_settings):
    class DummyAnalytics:
        def load_returns_from_db(self, days_back=30):
            return [0.01, 0.02], [
                datetime.now(timezone.utc),
                datetime.now(timezone.utc),
            ]

        def calculate_performance_metrics(self, returns, dates):
            return SimpleNamespace(
                total_return=0.1,
                annualized_return=0.2,
                volatility=0.05,
                sharpe_ratio=1.2,
                sortino_ratio=1.4,
                calmar_ratio=1.1,
                max_drawdown=0.1,
                win_rate=0.6,
                profit_factor=1.5,
                total_trades=10,
                period_start=dates[0],
                period_end=dates[-1],
            )

    _set_admin_key(monkeypatch, test_settings)
    monkeypatch.setattr(
        "app.reporting.performance_analytics.PerformanceAnalytics", DummyAnalytics
    )

    res = client.get("/admin/performance/summary", headers=_auth_headers())
    assert res.status_code == 200
    data = res.get_json()
    assert "total_return" in data


def test_performance_drawdown_endpoint(monkeypatch, client, test_settings):
    class DummyAnalytics:
        def load_returns_from_db(self, days_back=90):
            return [0.01, -0.02], [
                datetime.now(timezone.utc),
                datetime.now(timezone.utc),
            ]

        def analyze_drawdowns(self, returns, dates):
            return SimpleNamespace(
                max_drawdown=0.2,
                max_drawdown_duration_days=5,
                current_drawdown=0.1,
                drawdown_start=dates[0],
                drawdown_end=dates[-1],
                recovery_date=dates[-1] + timedelta(days=3),
                time_to_recovery_days=3,
                drawdowns=[{"start": dates[0], "end": dates[-1]}],
            )

    _set_admin_key(monkeypatch, test_settings)
    monkeypatch.setattr(
        "app.reporting.performance_analytics.PerformanceAnalytics", DummyAnalytics
    )

    res = client.get("/admin/performance/drawdown", headers=_auth_headers())
    assert res.status_code == 200
    data = res.get_json()
    assert "max_drawdown" in data


def test_performance_rolling_endpoint(monkeypatch, client, test_settings):
    class DummyAnalytics:
        def load_returns_from_db(self, days_back=90):
            return [0.01, 0.02], [
                datetime.now(timezone.utc),
                datetime.now(timezone.utc),
            ]

        def calculate_rolling_metrics(self, returns, dates, window_days=30):
            return SimpleNamespace(
                window_size_days=window_days,
                returns=[0.01, 0.02],
                volatilities=[0.1, 0.2],
                sharpe_ratios=[1.0, 1.1],
                dates=dates,
            )

    _set_admin_key(monkeypatch, test_settings)
    monkeypatch.setattr(
        "app.reporting.performance_analytics.PerformanceAnalytics", DummyAnalytics
    )

    res = client.get("/admin/performance/rolling", headers=_auth_headers())
    assert res.status_code == 200
    data = res.get_json()
    assert "rolling_returns" in data


def test_error_summary_endpoint(monkeypatch, client, test_settings):
    class DummyStats:
        total_errors = 2
        errors_by_severity = {"ERROR": 2}
        errors_by_type = {"ValueError": 1}
        errors_by_module = {"mod": 2}
        error_rate_per_hour = 0.5
        most_common_error = "ValueError"
        critical_errors = 0

    class DummyReporter:
        def get_error_statistics(self, hours_back=24):
            return DummyStats()

    _set_admin_key(monkeypatch, test_settings)
    monkeypatch.setattr(
        "app.infrastructure.error_reporter.ErrorReporter", DummyReporter
    )

    res = client.get("/admin/errors/summary", headers=_auth_headers())
    assert res.status_code == 200
    assert res.get_json()["total_errors"] == 2


def test_error_recent_endpoint(monkeypatch, client, test_settings):
    class DummyReporter:
        def get_recent_errors(self, limit=50, severity=None):
            return [{"msg": "oops"}]

    _set_admin_key(monkeypatch, test_settings)
    monkeypatch.setattr(
        "app.infrastructure.error_reporter.ErrorReporter", DummyReporter
    )

    res = client.get("/admin/errors/recent", headers=_auth_headers())
    assert res.status_code == 200
    assert res.get_json()["count"] == 1


def test_error_trends_endpoint(monkeypatch, client, test_settings):
    class DummyReporter:
        def get_error_trends(self, days=7):
            return {"days": days, "counts": [1, 2, 3]}

    _set_admin_key(monkeypatch, test_settings)
    monkeypatch.setattr(
        "app.infrastructure.error_reporter.ErrorReporter", DummyReporter
    )

    res = client.get("/admin/errors/trends", headers=_auth_headers())
    assert res.status_code == 200
    data = res.get_json()
    assert data["days"] == 7


def test_capital_utilization_endpoint(monkeypatch, client, test_settings):
    class DummyOptimizer:
        def get_position_history(self):
            return [
                {"portfolio_weight": 0.1, "kelly_fraction": 0.2},
                {"portfolio_weight": 0.2, "kelly_fraction": 0.3},
            ]

    _set_admin_key(monkeypatch, test_settings)
    monkeypatch.setattr(
        "app.decision.capital_optimizer.CapitalUtilizationOptimizer", DummyOptimizer
    )

    res = client.get("/admin/capital/utilization", headers=_auth_headers())
    assert res.status_code == 200
    assert "average_utilization" in res.get_json()


def test_no_trade_decisions_endpoint(monkeypatch, client, test_settings):
    class DummyLogger:
        def get_no_trade_analysis(self, days_back=7):
            return {"count": 2}

    _set_admin_key(monkeypatch, test_settings)
    monkeypatch.setattr(
        "app.infrastructure.decision_logger.NoTradeDecisionLogger", DummyLogger
    )

    res = client.get("/admin/decisions/no-trades", headers=_auth_headers())
    assert res.status_code == 200
    assert res.get_json()["count"] == 2


def test_strategy_performance_endpoint(monkeypatch, client, test_settings):
    class DummyAnalyzer:
        def analyze_strategy_performance(self, strategy_name, days=30):
            return StrategyStats(
                strategy_name=strategy_name,
                total_trades=10,
                win_rate=0.6,
                avg_pnl=0.01,
                sharpe_ratio=1.1,
                max_drawdown=0.1,
                last_30d_performance=0.6,
                status="GOOD",
                trades_analyzed=10,
            )

    _set_admin_key(monkeypatch, test_settings)
    monkeypatch.setattr(
        "app.decision.decision_history_analyzer.DecisionHistoryAnalyzer", DummyAnalyzer
    )

    res = client.get("/admin/strategies/performance", headers=_auth_headers())
    assert res.status_code == 200
    data = res.get_json()
    assert "strategies" in data


def test_confidence_distribution_endpoint(monkeypatch, client, test_settings):
    class DummyBucket:
        def __init__(self, value):
            self.value = value

    class DummyStats:
        def __init__(self):
            self.count = 3
            self.avg_confidence = 0.7
            self.total_capital = 1000.0
            self.avg_multiplier = 1.2

    class DummyAllocator:
        def get_bucket_statistics(self):
            return {DummyBucket("high"): DummyStats()}

    _set_admin_key(monkeypatch, test_settings)
    monkeypatch.setattr(
        "app.decision.confidence_allocator.ConfidenceBucketAllocator", DummyAllocator
    )

    res = client.get("/admin/confidence/distribution", headers=_auth_headers())
    assert res.status_code == 200
    data = res.get_json()
    assert "distribution" in data
