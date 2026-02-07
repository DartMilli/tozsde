import json
from datetime import datetime, timedelta, date
import pandas as pd
import pytest

from app.config.config import Config
from app.decision.confidence import (
    normalize_dqn_confidence,
    normalize_ppo_confidence,
    apply_confidence,
    clamp,
    normalize_final_confidence,
)
from app.decision.ensemble_quality import bucket_ensemble_quality
from app.decision.volatility_bucket import bucket_volatility, VolatilityBucket
from app.decision.drift_detector import (
    PerformanceDriftDetector,
    batch_check_drift,
    get_drifting_tickers,
)
from app.reporting.audit_builder import build_audit_metadata
from app.ui.app import app
from app.ui import admin_dashboard as admin_mod
from app.reporting.performance_analytics import (
    PerformanceMetrics,
    DrawdownAnalysis,
    RollingMetrics,
)
import main


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_confidence_utils():
    class Dummy:
        def __init__(self):
            self.confidence = None

    q_values = type(
        "T",
        (),
        {
            "detach": lambda s: s,
            "cpu": lambda s: s,
            "numpy": lambda s: [[1.0, 2.0, 3.0]],
        },
    )()
    conf = normalize_dqn_confidence(q_values, 2)
    assert 0.0 <= conf <= 1.0

    assert normalize_ppo_confidence(0.42) == 0.42
    dummy = Dummy()
    apply_confidence(dummy, 0.9)
    assert dummy.confidence == 0.9

    assert clamp(2.0, 0.0, 1.0) == 1.0
    assert normalize_final_confidence(-1) == 0.05


def test_ensemble_quality_and_volatility_bucket():
    bucket = bucket_ensemble_quality(0.9)
    assert bucket.value in {"STRONG", "NORMAL", "WEAK", "CHAOTIC"}

    assert bucket_volatility(None) == VolatilityBucket.NORMAL
    assert bucket_volatility(0.01) == VolatilityBucket.LOW
    assert bucket_volatility(0.02) == VolatilityBucket.NORMAL
    assert bucket_volatility(0.04) == VolatilityBucket.HIGH
    assert bucket_volatility(0.1) == VolatilityBucket.EXTREME


def test_drift_detector_metrics_and_alert(monkeypatch):
    detector = PerformanceDriftDetector(lookback_days=5)
    monkeypatch.setattr(
        detector, "_load_historical_scores", lambda t: [0.8, 0.7, 0.75, 0.72]
    )

    info = detector.check_drift("TEST", current_score=0.4)
    assert info["alert_level"] in {"WARNING", "CRITICAL", "NONE"}

    alert = detector.generate_alert("TEST", info)
    if info["drifting"]:
        assert "performance drift" in alert
    else:
        assert alert is None


def test_drift_batch_and_filter(monkeypatch):
    monkeypatch.setattr(
        PerformanceDriftDetector,
        "check_drift",
        lambda *a, **k: {"drifting": True, "alert_level": "WARNING"},
    )
    results = batch_check_drift({"A": 0.5, "B": 0.4})
    assert "A" in results

    drifting = get_drifting_tickers({"A": 0.5, "B": 0.4}, alert_level="WARNING")
    assert "A" in drifting and "B" in drifting


def test_audit_builder_with_drift(monkeypatch):
    monkeypatch.setattr(Config, "ENABLE_DRIFT_DETECTION", True)

    class DummyDetector:
        def check_drift(self, ticker, current_score):
            return "WARNING"

    monkeypatch.setattr(
        "app.decision.drift_detector.PerformanceDriftDetector", DummyDetector
    )

    payload = {"ticker": "TEST", "model_votes": [{"action": 1}], "volatility": 0.02}
    decision = {
        "action_code": 1,
        "action": "BUY",
        "confidence": 0.8,
        "wf_score": 0.7,
        "ensemble_quality": 0.8,
        "quality_score": 0.9,
        "no_trade": False,
    }

    audit = build_audit_metadata(payload, decision)
    assert audit["drift_status"] == "WARNING"


def test_admin_dashboard_endpoints(monkeypatch, client):
    monkeypatch.setattr(Config, "ADMIN_API_KEY", "key")

    class DummyAnalytics:
        def load_returns_from_db(self, days_back=30):
            return [0.01, -0.005], [datetime(2025, 1, 1), datetime(2025, 1, 2)]

        def calculate_performance_metrics(self, returns, dates):
            return PerformanceMetrics(
                total_return=0.1,
                annualized_return=0.2,
                volatility=0.1,
                sharpe_ratio=1.0,
                sortino_ratio=1.0,
                calmar_ratio=1.0,
                max_drawdown=-0.1,
                win_rate=0.6,
                profit_factor=1.2,
                total_trades=10,
                winning_trades=6,
                losing_trades=4,
                avg_win=0.02,
                avg_loss=-0.01,
                best_trade=0.05,
                worst_trade=-0.03,
                period_start=dates[0],
                period_end=dates[-1],
            )

        def analyze_drawdowns(self, returns, dates):
            return DrawdownAnalysis(
                max_drawdown=-0.1,
                max_drawdown_duration_days=5,
                current_drawdown=-0.02,
                drawdown_start=dates[0],
                drawdown_end=dates[-1],
                recovery_date=None,
                time_to_recovery_days=None,
                drawdowns=[(dates[0], dates[-1], -0.1)],
            )

        def calculate_rolling_metrics(self, returns, dates, window_days=30):
            return RollingMetrics(
                window_size_days=window_days,
                returns=returns,
                volatilities=[0.1, 0.2],
                sharpe_ratios=[1.0, 0.8],
                dates=dates,
            )

    class DummyErrorStats:
        total_errors = 1
        errors_by_severity = {"WARNING": 1}
        errors_by_type = {"E": 1}
        errors_by_module = {"m": 1}
        error_rate_per_hour = 0.1
        most_common_error = "E"
        critical_errors = 0

    class DummyErrorReporter:
        def get_error_statistics(self, hours_back=24):
            return DummyErrorStats()

        def get_recent_errors(self, limit=50, severity=None):
            return [{"error": "x"}]

        def get_error_trends(self, days=7):
            return {"days": days, "trend": []}

    class DummyCapitalOptimizer:
        def get_position_history(self):
            return [
                {"portfolio_weight": 0.5, "kelly_fraction": 0.25},
                {"portfolio_weight": 0.4, "kelly_fraction": 0.2},
            ]

    class DummyNoTradeLogger:
        def get_no_trade_analysis(self, days_back=7):
            return {"days": days_back, "count": 1}

    class DummyStrategyStats:
        trades_analyzed = 5
        win_rate = 0.6
        sharpe_ratio = 1.1
        max_drawdown = -0.1

    class DummyAnalyzer:
        def analyze_strategy_performance(self, strategy, days=30):
            return DummyStrategyStats()

    class DummyAllocator:
        def get_bucket_statistics(self):
            class Stat:
                def __init__(
                    self, count, avg_confidence, total_capital, avg_multiplier
                ):
                    self.count = count
                    self.avg_confidence = avg_confidence
                    self.total_capital = total_capital
                    self.avg_multiplier = avg_multiplier

            from app.decision.confidence_allocator import ConfidenceBucket

            return {
                ConfidenceBucket.STRONG: Stat(1, 0.8, 1000.0, 1.5),
                ConfidenceBucket.NORMAL: Stat(1, 0.6, 800.0, 1.0),
            }

    class DummyPyFolio:
        def generate_report(self, series):
            return {"sharpe_ratio": 1.0, "calmar_ratio": 0.5, "error": None}

    monkeypatch.setattr(
        "app.reporting.performance_analytics.PerformanceAnalytics", DummyAnalytics
    )
    monkeypatch.setattr(
        "app.infrastructure.error_reporter.ErrorReporter", DummyErrorReporter
    )
    monkeypatch.setattr(
        "app.decision.capital_optimizer.CapitalUtilizationOptimizer",
        DummyCapitalOptimizer,
    )
    monkeypatch.setattr(
        "app.infrastructure.decision_logger.NoTradeDecisionLogger", DummyNoTradeLogger
    )
    monkeypatch.setattr(
        "app.decision.decision_history_analyzer.DecisionHistoryAnalyzer", DummyAnalyzer
    )
    monkeypatch.setattr(
        "app.decision.confidence_allocator.ConfidenceBucketAllocator", DummyAllocator
    )
    monkeypatch.setattr(
        "app.reporting.pyfolio_report.PyFolioReportGenerator", DummyPyFolio
    )

    headers = {"X-Admin-Key": "key"}

    assert client.get("/admin/performance/summary", headers=headers).status_code == 200
    assert client.get("/admin/performance/drawdown", headers=headers).status_code == 200
    assert client.get("/admin/performance/rolling", headers=headers).status_code == 200
    assert client.get("/admin/errors/summary", headers=headers).status_code == 200
    assert client.get("/admin/errors/recent", headers=headers).status_code == 200
    assert client.get("/admin/errors/trends", headers=headers).status_code == 200
    assert client.get("/admin/capital/utilization", headers=headers).status_code == 200
    assert client.get("/admin/decisions/no-trades", headers=headers).status_code == 200
    assert (
        client.get("/admin/strategies/performance", headers=headers).status_code == 200
    )
    assert (
        client.get("/admin/confidence/distribution", headers=headers).status_code == 200
    )
    assert client.get("/admin/performance/pyfolio", headers=headers).status_code == 200


def test_ui_params_and_admin_routes(monkeypatch, client):
    monkeypatch.setattr(Config, "ADMIN_API_KEY", "key")
    monkeypatch.setattr("app.ui.app.get_params", lambda *a, **k: {"sma": 20})
    monkeypatch.setattr("app.ui.app.Config.get_supported_tickers", lambda: ["VOO"])
    monkeypatch.setattr("app.ui.app.render_template", lambda *a, **k: "OK")

    res = client.get("/params?ticker=VOO")
    assert res.status_code == 200

    class DummyMetrics:
        def get_health_status(self):
            return {"status": "ok"}

        def get_recent_metrics(self, hours=24):
            return {"total_executions": 1}

        def get_daily_summary(self, date_str):
            return {"date": date_str}

    monkeypatch.setattr(
        "app.infrastructure.metrics.get_metrics", lambda: DummyMetrics()
    )

    class DummyDM:
        def get_today_recommendations(self):
            return [{"ticker": "VOO"}]

    monkeypatch.setattr("app.ui.app.DataManager", lambda: DummyDM())

    res = client.get("/admin/dashboard", headers={"X-Admin-Key": "key"})
    assert res.status_code == 200

    res = client.get("/admin/metrics", headers={"X-Admin-Key": "key"})
    assert res.status_code == 200

    res = client.get("/admin/health", headers={"X-Admin-Key": "key"})
    assert res.status_code == 200

    res = client.post("/admin/force-rebalance", headers={"X-Admin-Key": "key"})
    assert res.status_code == 200


def test_main_weekly_monthly_and_manual(monkeypatch):
    monkeypatch.setattr(Config, "get_supported_tickers", lambda: ["AAA"])
    monkeypatch.setattr(Config, "TICKERS", ["AAA"])
    monkeypatch.setattr(
        "app.models.model_reliability.ModelReliabilityAnalyzer.analyze",
        lambda *a, **k: {"m": {}},
    )
    monkeypatch.setattr(
        "app.models.model_reliability.save_reliability_scores", lambda *a, **k: None
    )

    main.run_weekly(dry_run=True)

    monkeypatch.setattr(
        "main.run_walk_forward", lambda *a, **k: {"normalized_score": 0.5}
    )
    monkeypatch.setattr("main.train_rl_agent", lambda *a, **k: None)
    monkeypatch.setattr(Config, "ENABLE_RL", True)

    main.run_monthly(dry_run=True)
    main.run_walk_forward_manual("AAA", dry_run=True)
    main.run_train_rl_manual("AAA", dry_run=True)
