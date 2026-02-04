"""
Integration tests for Admin Dashboard endpoints (Sprint 10 Week 1 - Issue #4).

Tests all 11 admin endpoints with proper mocking and fixture setup:
1. /admin/health - System health check
2. /admin/performance/summary - Overall performance metrics
3. /admin/performance/drawdown - Drawdown analysis
4. /admin/performance/rolling - Rolling window metrics
5. /admin/errors/summary - Error statistics
6. /admin/errors/recent - Recent error records
7. /admin/errors/trends - Error time-series
8. /admin/capital/utilization - Capital usage metrics
9. /admin/decisions/no-trades - No-trade analysis
10. /admin/strategies/performance - Strategy breakdown
11. /admin/confidence/distribution - Confidence bucket stats

Coverage Target: AdminDashboard from 67% to 75%+
"""

import pytest
import json
from unittest.mock import patch, MagicMock, PropertyMock
from datetime import datetime, timedelta
from dataclasses import dataclass


@pytest.fixture
def client():
    """Create Flask test client."""
    from app.ui.app import app

    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def admin_headers():
    """Valid admin headers with authentication."""
    return {"X-Admin-Key": "admin_key_12345", "Content-Type": "application/json"}


@pytest.fixture
def invalid_headers():
    """Invalid admin headers."""
    return {"X-Admin-Key": "wrong_key", "Content-Type": "application/json"}


@pytest.fixture
def no_auth_headers():
    """No authentication headers."""
    return {"Content-Type": "application/json"}


class TestHealthEndpoint:
    """Test /admin/health endpoint."""

    @patch("app.infrastructure.metrics.get_metrics")
    def test_health_success(self, mock_get_metrics, client, admin_headers):
        """Should return health status with 200."""
        mock_metrics = MagicMock()
        mock_metrics.get_health_status.return_value = {
            "status": "healthy",
            "uptime_pct": 0.99,
            "error_rate": 0.001,
            "avg_response_time_sec": 45.0,
        }
        mock_get_metrics.return_value = mock_metrics

        response = client.get("/admin/health", headers=admin_headers)

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "healthy"
        assert data["uptime_pct"] == 0.99

    @patch("app.infrastructure.metrics.get_metrics")
    def test_health_error_handling(self, mock_get_metrics, client, admin_headers):
        """Should return 500 on error."""
        mock_get_metrics.side_effect = Exception("Metrics unavailable")

        response = client.get("/admin/health", headers=admin_headers)

        assert response.status_code == 500
        data = json.loads(response.data)
        assert "error" in data

    def test_health_unauthorized(self, client, invalid_headers):
        """Should reject unauthorized requests."""
        response = client.get("/admin/health", headers=invalid_headers)
        assert response.status_code == 401


class TestPerformanceSummaryEndpoint:
    """Test /admin/performance/summary endpoint."""

    @patch(
        "app.reporting.performance_analytics.PerformanceAnalytics.load_returns_from_db"
    )
    @patch(
        "app.reporting.performance_analytics.PerformanceAnalytics.calculate_performance_metrics"
    )
    def test_summary_success(self, mock_calc, mock_load, client, admin_headers):
        """Should return performance summary."""
        returns = [0.01, 0.02, -0.01, 0.015]
        dates = [datetime(2025, 1, i) for i in range(1, 5)]
        mock_load.return_value = (returns, dates)

        # Create mock metrics with proper attributes
        mock_metrics = MagicMock()
        mock_metrics.total_return = 0.065
        mock_metrics.annualized_return = 0.24
        mock_metrics.volatility = 0.12
        mock_metrics.sharpe_ratio = 1.5
        mock_metrics.sortino_ratio = 2.0
        mock_metrics.calmar_ratio = 1.2
        mock_metrics.max_drawdown = -0.03
        mock_metrics.win_rate = 0.75
        mock_metrics.profit_factor = 2.1
        mock_metrics.total_trades = 20
        mock_metrics.period_start = datetime(2025, 1, 1)
        mock_metrics.period_end = datetime(2025, 1, 4)

        mock_calc.return_value = mock_metrics

        response = client.get(
            "/admin/performance/summary?days=30", headers=admin_headers
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert "total_return" in data
        assert "sharpe_ratio" in data
        assert data["total_trades"] == 20

    @patch(
        "app.reporting.performance_analytics.PerformanceAnalytics.load_returns_from_db"
    )
    def test_summary_no_data(self, mock_load, client, admin_headers):
        """Should handle empty data gracefully."""
        mock_load.return_value = ([], [])

        response = client.get("/admin/performance/summary", headers=admin_headers)

        assert response.status_code == 200
        data = json.loads(response.data)
        assert "No performance data" in data["message"]

    @patch(
        "app.reporting.performance_analytics.PerformanceAnalytics.load_returns_from_db"
    )
    def test_summary_custom_period(self, mock_load, client, admin_headers):
        """Should respect custom days parameter."""
        mock_load.return_value = ([], [])

        response = client.get(
            "/admin/performance/summary?days=60", headers=admin_headers
        )

        assert response.status_code == 200
        mock_load.assert_called_with(days_back=60)


class TestDrawdownEndpoint:
    """Test /admin/performance/drawdown endpoint."""

    @patch("app.reporting.performance_analytics.PerformanceAnalytics.analyze_drawdowns")
    @patch(
        "app.reporting.performance_analytics.PerformanceAnalytics.load_returns_from_db"
    )
    def test_drawdown_success(self, mock_load, mock_analyze, client, admin_headers):
        """Should return drawdown analysis."""
        mock_load.return_value = (
            [0.01, -0.02, 0.015],
            [datetime(2025, 1, i) for i in range(1, 4)],
        )

        mock_dd = MagicMock()
        mock_dd.max_drawdown = -0.05
        mock_dd.max_drawdown_duration_days = 5
        mock_dd.current_drawdown = -0.02
        mock_dd.drawdown_start = datetime(2025, 1, 2)
        mock_dd.drawdown_end = datetime(2025, 1, 7)
        mock_dd.recovery_date = datetime(2025, 1, 10)
        mock_dd.time_to_recovery_days = 8
        mock_dd.drawdowns = [{}, {}, {}]

        mock_analyze.return_value = mock_dd

        response = client.get(
            "/admin/performance/drawdown?days=90", headers=admin_headers
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["max_drawdown"] == -5.0
        assert data["max_drawdown_duration_days"] == 5
        assert data["total_drawdowns"] == 3

    @patch(
        "app.reporting.performance_analytics.PerformanceAnalytics.load_returns_from_db"
    )
    def test_drawdown_no_data(self, mock_load, client, admin_headers):
        """Should handle empty drawdown data."""
        mock_load.return_value = ([], [])

        response = client.get("/admin/performance/drawdown", headers=admin_headers)

        assert response.status_code == 200
        data = json.loads(response.data)
        assert "No drawdown data" in data["message"]


class TestRollingPerformanceEndpoint:
    """Test /admin/performance/rolling endpoint."""

    @patch(
        "app.reporting.performance_analytics.PerformanceAnalytics.calculate_rolling_metrics"
    )
    @patch(
        "app.reporting.performance_analytics.PerformanceAnalytics.load_returns_from_db"
    )
    def test_rolling_success(self, mock_load, mock_rolling, client, admin_headers):
        """Should return rolling performance metrics."""
        dates = [datetime(2025, 1, i) for i in range(1, 11)]
        mock_load.return_value = ([0.01] * 10, dates)

        mock_rolling_data = MagicMock()
        mock_rolling_data.window_size_days = 30
        mock_rolling_data.returns = [0.01, 0.015, 0.012]
        mock_rolling_data.volatilities = [0.05, 0.06, 0.055]
        mock_rolling_data.sharpe_ratios = [1.0, 1.2, 1.1]
        mock_rolling_data.dates = dates[:3]

        mock_rolling.return_value = mock_rolling_data

        response = client.get(
            "/admin/performance/rolling?days=90&window=30", headers=admin_headers
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["window_size_days"] == 30
        assert len(data["rolling_returns"]) == 3
        assert len(data["rolling_volatilities"]) == 3

    @patch(
        "app.reporting.performance_analytics.PerformanceAnalytics.load_returns_from_db"
    )
    def test_rolling_no_data(self, mock_load, client, admin_headers):
        """Should handle empty rolling data."""
        mock_load.return_value = ([], [])

        response = client.get("/admin/performance/rolling", headers=admin_headers)

        assert response.status_code == 200
        data = json.loads(response.data)
        assert "No performance data" in data["message"]


class TestErrorSummaryEndpoint:
    """Test /admin/errors/summary endpoint."""

    @patch("app.infrastructure.error_reporter.ErrorReporter.get_error_statistics")
    def test_error_summary_success(self, mock_stats, client, admin_headers):
        """Should return error statistics."""
        mock_error_stats = MagicMock()
        mock_error_stats.total_errors = 10
        mock_error_stats.errors_by_severity = {"ERROR": 7, "WARNING": 3}
        mock_error_stats.errors_by_type = {"ValueError": 5, "KeyError": 5}
        mock_error_stats.errors_by_module = {"data_loader": 8, "backtester": 2}
        mock_error_stats.error_rate_per_hour = 0.42
        mock_error_stats.most_common_error = "ValueError"
        mock_error_stats.critical_errors = 0

        mock_stats.return_value = mock_error_stats

        response = client.get("/admin/errors/summary?hours=24", headers=admin_headers)

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["total_errors"] == 10
        assert data["error_rate_per_hour"] == 0.42
        assert data["most_common_error"] == "ValueError"

    @patch("app.infrastructure.error_reporter.ErrorReporter.get_error_statistics")
    def test_error_summary_no_errors(self, mock_stats, client, admin_headers):
        """Should handle zero errors."""
        mock_error_stats = MagicMock()
        mock_error_stats.total_errors = 0
        mock_error_stats.errors_by_severity = {}
        mock_error_stats.errors_by_type = {}
        mock_error_stats.errors_by_module = {}
        mock_error_stats.error_rate_per_hour = 0.0
        mock_error_stats.most_common_error = None
        mock_error_stats.critical_errors = 0

        mock_stats.return_value = mock_error_stats

        response = client.get("/admin/errors/summary", headers=admin_headers)

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["total_errors"] == 0


class TestRecentErrorsEndpoint:
    """Test /admin/errors/recent endpoint."""

    @patch("app.infrastructure.error_reporter.ErrorReporter.get_recent_errors")
    def test_recent_errors_success(self, mock_get_recent, client, admin_headers):
        """Should return recent errors."""
        mock_errors = [
            {
                "timestamp": datetime.now().isoformat(),
                "severity": "ERROR",
                "message": "Test error 1",
                "module": "data_loader",
            },
            {
                "timestamp": datetime.now().isoformat(),
                "severity": "WARNING",
                "message": "Test warning",
                "module": "backtester",
            },
        ]
        mock_get_recent.return_value = mock_errors

        response = client.get("/admin/errors/recent?limit=50", headers=admin_headers)

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["count"] == 2
        assert len(data["errors"]) == 2

    @patch("app.infrastructure.error_reporter.ErrorReporter.get_recent_errors")
    def test_recent_errors_with_severity_filter(
        self, mock_get_recent, client, admin_headers
    ):
        """Should filter by severity."""
        mock_errors = [
            {
                "timestamp": datetime.now().isoformat(),
                "severity": "ERROR",
                "message": "Test error",
            }
        ]
        mock_get_recent.return_value = mock_errors

        response = client.get(
            "/admin/errors/recent?limit=20&severity=ERROR", headers=admin_headers
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["count"] == 1

    @patch("app.infrastructure.error_reporter.ErrorReporter.get_recent_errors")
    def test_recent_errors_invalid_severity(
        self, mock_get_recent, client, admin_headers
    ):
        """Should reject invalid severity."""
        response = client.get(
            "/admin/errors/recent?severity=INVALID", headers=admin_headers
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "Invalid severity" in data["error"]


class TestErrorTrendsEndpoint:
    """Test /admin/errors/trends endpoint."""

    @patch("app.infrastructure.error_reporter.ErrorReporter.get_error_trends")
    def test_error_trends_success(self, mock_trends, client, admin_headers):
        """Should return error trends."""
        mock_trends_data = {
            "2025-01-01": 5,
            "2025-01-02": 3,
            "2025-01-03": 7,
            "2025-01-04": 2,
            "2025-01-05": 4,
        }
        mock_trends.return_value = mock_trends_data

        response = client.get("/admin/errors/trends?days=7", headers=admin_headers)

        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data) == 5
        assert data["2025-01-03"] == 7


class TestCapitalUtilizationEndpoint:
    """Test /admin/capital/utilization endpoint."""

    @patch(
        "app.decision.capital_optimizer.CapitalUtilizationOptimizer.get_position_history"
    )
    def test_capital_utilization_success(self, mock_history, client, admin_headers):
        """Should return capital utilization metrics."""
        mock_history.return_value = [
            {
                "ticker": "AAPL",
                "portfolio_weight": 0.05,
                "kelly_fraction": 0.02,
                "timestamp": datetime.now().isoformat(),
            },
            {
                "ticker": "MSFT",
                "portfolio_weight": 0.04,
                "kelly_fraction": 0.018,
                "timestamp": datetime.now().isoformat(),
            },
        ]

        response = client.get("/admin/capital/utilization", headers=admin_headers)

        assert response.status_code == 200
        data = json.loads(response.data)
        assert "average_utilization" in data
        assert "average_kelly_fraction" in data
        assert data["total_positions"] == 2

    @patch(
        "app.decision.capital_optimizer.CapitalUtilizationOptimizer.get_position_history"
    )
    def test_capital_utilization_no_data(self, mock_history, client, admin_headers):
        """Should handle empty position history."""
        mock_history.return_value = []

        response = client.get("/admin/capital/utilization", headers=admin_headers)

        assert response.status_code == 200
        data = json.loads(response.data)
        assert "No capital utilization data" in data["message"]


class TestNoTradeDecisionsEndpoint:
    """Test /admin/decisions/no-trades endpoint."""

    @patch(
        "app.infrastructure.decision_logger.NoTradeDecisionLogger.get_no_trade_analysis"
    )
    def test_no_trade_decisions_success(self, mock_analysis, client, admin_headers):
        """Should return no-trade analysis."""
        mock_analysis_data = {
            "total_skipped": 15,
            "by_reason": {
                "LOW_CONFIDENCE": 8,
                "HIGH_CORRELATION": 5,
                "POSITION_LIMIT_REACHED": 2,
            },
            "by_ticker": {"AAPL": 5, "MSFT": 4, "JNJ": 6},
            "average_confidence_when_skipped": 0.45,
            "period_days": 7,
        }
        mock_analysis.return_value = mock_analysis_data

        response = client.get(
            "/admin/decisions/no-trades?days=7", headers=admin_headers
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["total_skipped"] == 15
        assert data["by_reason"]["LOW_CONFIDENCE"] == 8


class TestStrategyPerformanceEndpoint:
    """Test /admin/strategies/performance endpoint."""

    @patch(
        "app.decision.decision_history_analyzer.DecisionHistoryAnalyzer.analyze_strategy_performance"
    )
    def test_strategy_performance_success(self, mock_analyze, client, admin_headers):
        """Should return strategy performance breakdown."""
        from app.decision.decision_history_analyzer import StrategyStats

        def side_effect(strategy, days=30):
            if strategy in ["momentum", "mean_reversion", "breakout"]:
                return StrategyStats(
                    strategy_name=strategy,
                    total_trades=25,
                    win_rate=0.65,
                    avg_pnl=0.02,
                    sharpe_ratio=1.2,
                    max_drawdown=0.08,
                    last_30d_performance=0.6,
                    status="GOOD",
                    trades_analyzed=25,
                )
            return None

        mock_analyze.side_effect = side_effect

        response = client.get(
            "/admin/strategies/performance?days=30", headers=admin_headers
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert "strategies" in data
        assert data["period_days"] == 30

    @patch(
        "app.decision.decision_history_analyzer.DecisionHistoryAnalyzer.analyze_strategy_performance"
    )
    def test_strategy_performance_no_data(self, mock_analyze, client, admin_headers):
        """Should handle no strategy data."""
        mock_analyze.return_value = None

        response = client.get("/admin/strategies/performance", headers=admin_headers)

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["strategies"] == {}


class TestConfidenceDistributionEndpoint:
    """Test /admin/confidence/distribution endpoint."""

    @patch(
        "app.decision.confidence_allocator.ConfidenceBucketAllocator.get_bucket_statistics"
    )
    def test_confidence_distribution_success(self, mock_stats, client, admin_headers):
        """Should return confidence bucket statistics."""
        from app.decision.confidence_allocator import ConfidenceBucket

        mock_bucket_stats = {
            ConfidenceBucket.STRONG: MagicMock(
                count=10, avg_confidence=0.85, total_capital=5000, avg_multiplier=1.5
            ),
            ConfidenceBucket.NORMAL: MagicMock(
                count=20, avg_confidence=0.65, total_capital=8000, avg_multiplier=1.0
            ),
            ConfidenceBucket.WEAK: MagicMock(
                count=5, avg_confidence=0.35, total_capital=1000, avg_multiplier=0.5
            ),
        }
        mock_stats.return_value = mock_bucket_stats

        response = client.get("/admin/confidence/distribution", headers=admin_headers)

        assert response.status_code == 200
        data = json.loads(response.data)
        assert "distribution" in data
        assert "strong" in data["distribution"] or "STRONG" in data["distribution"]


# Authentication Tests
class TestAdminAuthenticationAcrossEndpoints:
    """Test that all endpoints require proper authentication."""

    endpoints = [
        "/admin/health",
        "/admin/performance/summary",
        "/admin/performance/drawdown",
        "/admin/performance/rolling",
        "/admin/errors/summary",
        "/admin/errors/recent",
        "/admin/errors/trends",
        "/admin/capital/utilization",
        "/admin/decisions/no-trades",
        "/admin/strategies/performance",
        "/admin/confidence/distribution",
    ]

    def test_all_endpoints_reject_no_auth(self, client, no_auth_headers):
        """All endpoints should reject requests without auth."""
        for endpoint in self.endpoints:
            response = client.get(endpoint, headers=no_auth_headers)
            assert response.status_code == 401, f"Failed for {endpoint}"

    def test_all_endpoints_reject_invalid_auth(self, client, invalid_headers):
        """All endpoints should reject invalid auth keys."""
        for endpoint in self.endpoints:
            response = client.get(endpoint, headers=invalid_headers)
            assert response.status_code == 401, f"Failed for {endpoint}"


# Error Handling Tests
class TestAdminErrorHandling:
    """Test error handling across endpoints."""

    @patch("app.infrastructure.metrics.get_metrics")
    def test_health_exception_handling(self, mock_get_metrics, client, admin_headers):
        """Should handle exceptions in health check."""
        mock_get_metrics.side_effect = RuntimeError("System error")

        response = client.get("/admin/health", headers=admin_headers)

        assert response.status_code == 500
        data = json.loads(response.data)
        assert "error" in data

    @patch(
        "app.reporting.performance_analytics.PerformanceAnalytics.load_returns_from_db"
    )
    def test_performance_exception_handling(self, mock_load, client, admin_headers):
        """Should handle exceptions in performance endpoint."""
        mock_load.side_effect = RuntimeError("Database error")

        response = client.get("/admin/performance/summary", headers=admin_headers)

        assert response.status_code == 500
        data = json.loads(response.data)
        assert "error" in data

    @patch("app.infrastructure.error_reporter.ErrorReporter.get_error_statistics")
    def test_error_summary_exception_handling(self, mock_stats, client, admin_headers):
        """Should handle exceptions in error summary endpoint."""
        mock_stats.side_effect = RuntimeError("Reporter error")

        response = client.get("/admin/errors/summary", headers=admin_headers)

        assert response.status_code == 500
        data = json.loads(response.data)
        assert "error" in data


# Parameter Validation Tests
class TestParameterValidation:
    """Test parameter validation across endpoints."""

    @patch(
        "app.reporting.performance_analytics.PerformanceAnalytics.load_returns_from_db"
    )
    @patch(
        "app.reporting.performance_analytics.PerformanceAnalytics.calculate_performance_metrics"
    )
    def test_days_parameter_parsing(self, mock_calc, mock_load, client, admin_headers):
        """Should parse days parameter correctly."""
        mock_load.return_value = ([], [])

        client.get("/admin/performance/summary?days=60", headers=admin_headers)
        mock_load.assert_called_with(days_back=60)

    @patch(
        "app.reporting.performance_analytics.PerformanceAnalytics.load_returns_from_db"
    )
    @patch(
        "app.reporting.performance_analytics.PerformanceAnalytics.calculate_rolling_metrics"
    )
    def test_window_parameter_parsing(
        self, mock_rolling, mock_load, client, admin_headers
    ):
        """Should parse window parameter correctly."""
        mock_load.return_value = (
            [0.01] * 10,
            [datetime(2025, 1, i) for i in range(1, 11)],
        )
        mock_rolling.return_value = MagicMock(
            window_size_days=14, returns=[], volatilities=[], sharpe_ratios=[], dates=[]
        )

        client.get("/admin/performance/rolling?window=14", headers=admin_headers)
        mock_rolling.assert_called_with(
            [0.01] * 10, mock_load.return_value[1], window_days=14
        )

    @patch("app.infrastructure.error_reporter.ErrorReporter.get_recent_errors")
    def test_limit_parameter_parsing(self, mock_get_recent, client, admin_headers):
        """Should parse limit parameter correctly."""
        mock_get_recent.return_value = []

        client.get("/admin/errors/recent?limit=100", headers=admin_headers)
        mock_get_recent.assert_called_with(limit=100, severity=None)
