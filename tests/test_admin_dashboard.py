"""
Unit tests for Admin Dashboard Routes (P9 - Engineering Hardening).

Tests:
- Authentication/authorization (admin key validation)
- Dashboard endpoint responses
- Health check endpoint
- Metrics queries
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone


@pytest.fixture
def client():
    """Create Flask test client."""
    from app.ui.app import app

    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def admin_headers():
    """Valid admin headers."""
    return {"X-Admin-Key": "admin_key_12345", "Content-Type": "application/json"}


@pytest.fixture
def no_auth_headers():
    """No auth headers."""
    return {"Content-Type": "application/json"}


class TestAdminAuth:
    """Test admin authentication."""

    def test_dashboard_rejects_no_auth(self, client, no_auth_headers):
        """Should reject requests without auth."""
        response = client.get("/admin/dashboard", headers=no_auth_headers)
        assert response.status_code == 401

    def test_metrics_rejects_no_auth(self, client, no_auth_headers):
        """Should reject metrics without auth."""
        response = client.get("/admin/metrics", headers=no_auth_headers)
        assert response.status_code == 401

    def test_health_rejects_no_auth(self, client, no_auth_headers):
        """Should reject health without auth."""
        response = client.get("/admin/health", headers=no_auth_headers)
        assert response.status_code == 401

    def test_rebalance_rejects_no_auth(self, client, no_auth_headers):
        """Should reject rebalance without auth."""
        response = client.post("/admin/force-rebalance", headers=no_auth_headers)
        assert response.status_code == 401


class TestAdminDashboard:
    """Test dashboard endpoint with mocked metrics."""

    @patch("app.data_access.data_manager.DataManager.get_today_recommendations")
    @patch("app.infrastructure.metrics.get_metrics")
    def test_dashboard_success(
        self, mock_get_metrics, mock_recs, client, admin_headers
    ):
        """Should return dashboard data."""
        mock_recs.return_value = [{"ticker": "VOO"}]

        mock_metrics = MagicMock()
        mock_metrics.get_health_status.return_value = {
            "status": "healthy",
            "uptime_pct": 0.99,
            "error_rate": 0.01,
            "avg_response_time_sec": 45.0,
            "last_check": datetime.now(timezone.utc).isoformat(),
        }
        mock_metrics.get_recent_metrics.return_value = {
            "success_rate": 0.99,
            "avg_duration_sec": 45.0,
            "errors_count": 1,
            "total_executions": 100,
            "last_success": None,
            "last_error": None,
        }
        mock_metrics.get_daily_summary.return_value = {
            "date": datetime.today().strftime("%Y-%m-%d"),
            "executions": 50,
            "successes": 49,
            "failures": 1,
            "avg_duration_sec": 45.0,
            "tickers_processed": ["VOO"],
        }
        mock_get_metrics.return_value = mock_metrics

        response = client.get("/admin/dashboard", headers=admin_headers)
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "health" in data
        assert "metrics" in data


class TestAdminHealth:
    """Test health endpoint."""

    @patch("app.infrastructure.metrics.get_metrics")
    def test_health_check_response(self, mock_get_metrics, client, admin_headers):
        """Should return health status."""
        mock_metrics = MagicMock()
        mock_metrics.get_health_status.return_value = {
            "status": "healthy",
            "uptime_pct": 0.99,
            "error_rate": 0.01,
            "avg_response_time_sec": 45.0,
            "last_check": datetime.now(timezone.utc).isoformat(),
        }
        mock_get_metrics.return_value = mock_metrics

        response = client.get("/admin/health", headers=admin_headers)
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "healthy"
        assert data["uptime_pct"] == 0.99


class TestAdminMetrics:
    """Test metrics endpoint."""

    @patch("app.infrastructure.metrics.get_metrics")
    def test_metrics_default(self, mock_get_metrics, client, admin_headers):
        """Should return recent metrics."""
        mock_metrics = MagicMock()
        mock_metrics.get_recent_metrics.return_value = {
            "success_rate": 0.99,
            "avg_duration_sec": 45.0,
            "errors_count": 1,
            "total_executions": 100,
            "last_success": None,
            "last_error": None,
        }
        mock_metrics.get_health_status.return_value = {
            "status": "healthy",
            "uptime_pct": 0.99,
            "error_rate": 0.01,
            "avg_response_time_sec": 45.0,
            "last_check": datetime.now(timezone.utc).isoformat(),
        }
        mock_get_metrics.return_value = mock_metrics

        response = client.get("/admin/metrics", headers=admin_headers)
        assert response.status_code == 200
        mock_metrics.get_recent_metrics.assert_called_with(hours=24)

    @patch("app.infrastructure.metrics.get_metrics")
    def test_metrics_with_hours(self, mock_get_metrics, client, admin_headers):
        """Should accept hours parameter."""
        mock_metrics = MagicMock()
        mock_metrics.get_recent_metrics.return_value = {
            "success_rate": 0.98,
            "avg_duration_sec": 50.0,
            "errors_count": 2,
            "total_executions": 100,
            "last_success": None,
            "last_error": None,
        }
        mock_metrics.get_health_status.return_value = {
            "status": "healthy",
            "uptime_pct": 0.98,
            "error_rate": 0.02,
            "avg_response_time_sec": 50.0,
            "last_check": datetime.now(timezone.utc).isoformat(),
        }
        mock_get_metrics.return_value = mock_metrics

        response = client.get("/admin/metrics?hours=12", headers=admin_headers)
        assert response.status_code == 200
        mock_metrics.get_recent_metrics.assert_called_with(hours=12)

    @patch("app.infrastructure.metrics.get_metrics")
    def test_metrics_with_date(self, mock_get_metrics, client, admin_headers):
        """Should accept date parameter."""
        mock_metrics = MagicMock()
        mock_metrics.get_daily_summary.return_value = {
            "date": "2025-01-20",
            "executions": 50,
            "successes": 49,
            "failures": 1,
            "avg_duration_sec": 45.0,
            "tickers_processed": [],
        }
        mock_get_metrics.return_value = mock_metrics

        response = client.get("/admin/metrics?date=2025-01-20", headers=admin_headers)
        assert response.status_code == 200
        mock_metrics.get_daily_summary.assert_called_with("2025-01-20")


class TestAdminPerformance:
    """Test performance endpoints."""

    @patch(
        "app.reporting.performance_analytics.PerformanceAnalytics.load_returns_from_db"
    )
    @patch(
        "app.reporting.performance_analytics.PerformanceAnalytics.calculate_performance_metrics"
    )
    def test_performance_summary(self, mock_calc, mock_load, client, admin_headers):
        """Test performance summary endpoint."""
        mock_load.return_value = ([], [])
        response = client.get(
            "/admin/performance/summary?days=30", headers=admin_headers
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "message" in data or "total_return" in data

    @patch(
        "app.reporting.performance_analytics.PerformanceAnalytics.load_returns_from_db"
    )
    def test_performance_drawdown(self, mock_load, client, admin_headers):
        """Test drawdown analysis endpoint."""
        mock_load.return_value = ([], [])
        response = client.get(
            "/admin/performance/drawdown?days=90", headers=admin_headers
        )
        assert response.status_code == 200

    @patch(
        "app.reporting.performance_analytics.PerformanceAnalytics.load_returns_from_db"
    )
    def test_performance_rolling(self, mock_load, client, admin_headers):
        """Test rolling metrics endpoint."""
        mock_load.return_value = ([], [])
        response = client.get(
            "/admin/performance/rolling?days=180", headers=admin_headers
        )
        assert response.status_code == 200


class TestAdminErrors:
    """Test error reporting endpoints."""

    @patch("app.infrastructure.error_reporter.ErrorReporter.get_error_statistics")
    def test_errors_summary(self, mock_stats, client, admin_headers):
        """Test error summary endpoint."""
        mock_obj = MagicMock()
        mock_obj.total_errors = 5
        mock_obj.errors_by_severity = {"ERROR": 3}
        mock_obj.errors_by_type = {}
        mock_obj.errors_by_module = {}
        mock_obj.error_rate_per_hour = 0.2
        mock_obj.most_common_error = "test"
        mock_obj.critical_errors = 1
        mock_stats.return_value = mock_obj

        response = client.get("/admin/errors/summary", headers=admin_headers)
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "total_errors" in data or "error" in data

    @patch("app.infrastructure.error_reporter.ErrorReporter.get_recent_errors")
    def test_errors_recent(self, mock_recent, client, admin_headers):
        """Test recent errors endpoint."""
        mock_recent.return_value = []
        response = client.get("/admin/errors/recent?limit=50", headers=admin_headers)
        assert response.status_code == 200

    @patch("app.infrastructure.error_reporter.ErrorReporter.get_error_trends")
    def test_errors_trends(self, mock_trends, client, admin_headers):
        """Test error trends endpoint."""
        mock_trends.return_value = {}
        response = client.get("/admin/errors/trends?days=30", headers=admin_headers)
        assert response.status_code == 200


class TestAdminCapital:
    """Test capital management endpoints."""

    @patch(
        "app.decision.capital_optimizer.CapitalUtilizationOptimizer.get_position_history"
    )
    def test_capital_utilization(self, mock_history, client, admin_headers):
        """Test capital utilization endpoint."""
        mock_history.return_value = []
        response = client.get("/admin/capital/utilization", headers=admin_headers)
        assert response.status_code == 200


class TestAdminDecisions:
    """Test decision monitoring endpoints."""

    @patch(
        "app.infrastructure.decision_logger.NoTradeDecisionLogger.get_no_trade_analysis"
    )
    def test_decisions_no_trades(self, mock_analysis, client, admin_headers):
        """Test no-trade decisions endpoint."""
        mock_analysis.return_value = {}
        response = client.get(
            "/admin/decisions/no-trades?days=7", headers=admin_headers
        )
        assert response.status_code == 200


class TestAdminStrategies:
    """Test strategy performance endpoints."""

    @patch(
        "app.decision.decision_history_analyzer.DecisionHistoryAnalyzer.analyze_strategy_performance"
    )
    def test_strategies_performance(self, mock_stats, client, admin_headers):
        """Test strategy performance endpoint."""
        from app.decision.decision_history_analyzer import StrategyStats

        mock_stats.return_value = StrategyStats(
            strategy_name="momentum",
            total_trades=10,
            win_rate=0.5,
            avg_pnl=0.01,
            sharpe_ratio=1.5,
            max_drawdown=0.1,
            last_30d_performance=0.5,
            status="GOOD",
            trades_analyzed=10,
        )
        response = client.get("/admin/strategies/performance", headers=admin_headers)
        assert response.status_code == 200


class TestAdminConfidence:
    """Test confidence distribution endpoints."""

    @patch(
        "app.decision.confidence_allocator.ConfidenceBucketAllocator.get_bucket_statistics"
    )
    def test_confidence_distribution(self, mock_dist, client, admin_headers):
        """Test confidence distribution endpoint."""
        mock_dist.return_value = {}
        response = client.get("/admin/confidence/distribution", headers=admin_headers)
        assert response.status_code == 200


class TestForceRebalance:
    """Test force rebalance endpoint."""

    def test_rebalance_post_success(self, client, admin_headers):
        """Should accept POST request."""
        response = client.post("/admin/force-rebalance", headers=admin_headers)
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "rebalance_initiated"

    def test_rebalance_get_rejected(self, client, admin_headers):
        """Should reject GET requests."""
        response = client.get("/admin/force-rebalance", headers=admin_headers)
        assert response.status_code in [405, 404]
