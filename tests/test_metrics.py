"""
Unit tests for SystemMetrics (P9 — Engineering Hardening).

Tests metrics logging via DataManager (NO direct SQL in metrics.py).
Each test gets a clean, isolated database.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from app.config.config import Config
from app.infrastructure.metrics import SystemMetrics
from app.data_access.data_manager import DataManager


@pytest.fixture
def fresh_db():
    """Create a fresh test database for each test (isolation)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Store originals
        original_data_dir = Config.DATA_DIR
        original_db_path = Config.DB_PATH
        original_log_dir = Config.LOG_DIR

        # Temp paths
        Config.DATA_DIR = Path(tmpdir) / "data"
        Config.DATA_DIR.mkdir(exist_ok=True)
        Config.DB_PATH = Config.DATA_DIR / "test.db"
        Config.LOG_DIR = Path(tmpdir) / "logs"
        Config.LOG_DIR.mkdir(exist_ok=True)

        # Init
        dm = DataManager()
        dm.initialize_tables()

        yield dm

        # Restore
        Config.DATA_DIR = original_data_dir
        Config.DB_PATH = original_db_path
        Config.LOG_DIR = original_log_dir


class TestPipelineLogging:
    """Test pipeline execution logging via DataManager."""

    def test_log_successful(self, fresh_db):
        """Should log successful execution."""
        metrics = SystemMetrics()
        result = metrics.log_pipeline_execution("VOO", "success", 45.5)
        assert result is True

        with fresh_db._get_conn() as conn:
            row = conn.execute(
                "SELECT ticker, status, duration_sec FROM pipeline_metrics WHERE ticker='VOO'"
            ).fetchone()
        assert row == ("VOO", "success", 45.5)

    def test_log_error(self, fresh_db):
        """Should log failed execution with message."""
        metrics = SystemMetrics()
        msg = "DB timeout"
        result = metrics.log_pipeline_execution("SPY", "error", 0.1, msg)
        assert result is True

        with fresh_db._get_conn() as conn:
            row = conn.execute(
                "SELECT status, error_message FROM pipeline_metrics WHERE ticker='SPY'"
            ).fetchone()
        assert row == ("error", msg)

    def test_log_multiple(self, fresh_db):
        """Should log multiple executions."""
        metrics = SystemMetrics()
        for i in range(3):
            metrics.log_pipeline_execution(f"TIC{i}", "success", 40.0)

        with fresh_db._get_conn() as conn:
            count = conn.execute("SELECT COUNT(*) FROM pipeline_metrics").fetchone()[0]
        assert count == 3


class TestBacktestLogging:
    """Test backtest execution logging via DataManager."""

    def test_log_backtest_success(self, fresh_db):
        """Should log backtest with metrics."""
        metrics = SystemMetrics()
        result = metrics.log_backtest_execution("VOO", 0.75, 42, 1.85)
        assert result is True

        with fresh_db._get_conn() as conn:
            row = conn.execute(
                "SELECT status, error_message FROM pipeline_metrics WHERE ticker='VOO'"
            ).fetchone()
        assert row[0] == "success"
        assert "WF=0.750" in row[1]
        assert "trades=42" in row[1]

    def test_log_backtest_error(self, fresh_db):
        """Should log backtest error."""
        metrics = SystemMetrics()
        msg = "Bad data"
        result = metrics.log_backtest_execution("TST", 0.0, 0, 0.0, msg)
        assert result is True

        with fresh_db._get_conn() as conn:
            row = conn.execute(
                "SELECT status, error_message FROM pipeline_metrics WHERE ticker='TST'"
            ).fetchone()
        assert row == ("error", msg)


class TestMetricsAggregation:
    """Test metrics aggregation (via DataManager methods)."""

    def test_empty_metrics(self, fresh_db):
        """Should return zeros for empty metrics."""
        metrics = SystemMetrics()
        result = metrics.get_recent_metrics(hours=24)

        assert result["success_rate"] == 0.0
        assert result["avg_duration_sec"] == 0.0
        assert result["errors_count"] == 0
        assert result["total_executions"] == 0

    def test_single_success(self, fresh_db):
        """Should return 100% success rate."""
        metrics = SystemMetrics()
        metrics.log_pipeline_execution("VOO", "success", 45.0)
        result = metrics.get_recent_metrics(hours=24)

        assert result["success_rate"] == 1.0
        assert result["total_executions"] == 1
        assert result["errors_count"] == 0
        assert result["avg_duration_sec"] == 45.0

    def test_mixed_statuses(self, fresh_db):
        """Should calculate correct success rate."""
        metrics = SystemMetrics()
        metrics.log_pipeline_execution("A", "success", 40.0)
        metrics.log_pipeline_execution("B", "success", 50.0)
        metrics.log_pipeline_execution("C", "success", 60.0)
        metrics.log_pipeline_execution("D", "error", 1.0)

        result = metrics.get_recent_metrics(hours=24)
        assert result["total_executions"] == 4
        assert result["errors_count"] == 1
        assert result["success_rate"] == 0.75

    def test_avg_duration(self, fresh_db):
        """Should calculate average duration correctly."""
        metrics = SystemMetrics()
        metrics.log_pipeline_execution("A", "success", 30.0)
        metrics.log_pipeline_execution("B", "success", 60.0)
        metrics.log_pipeline_execution("C", "error", 10.0)

        result = metrics.get_recent_metrics(hours=24)
        assert result["avg_duration_sec"] == 45.0  # (30+60)/2


class TestDailySummary:
    """Test daily summary aggregation."""

    def test_empty_summary(self, fresh_db):
        """Should return zeros for no data."""
        metrics = SystemMetrics()
        result = metrics.get_daily_summary("2025-01-20")

        assert result["date"] == "2025-01-20"
        assert result["executions"] == 0
        assert result["successes"] == 0
        assert result["failures"] == 0

    def test_summary_with_data(self, fresh_db):
        """Should aggregate daily data."""
        metrics = SystemMetrics()
        today = datetime.today().strftime("%Y-%m-%d")

        metrics.log_pipeline_execution("VOO", "success", 40.0)
        metrics.log_pipeline_execution("SPY", "success", 50.0)
        metrics.log_pipeline_execution("QQQ", "error", 5.0)

        result = metrics.get_daily_summary(today)
        assert result["executions"] == 3
        assert result["successes"] == 2
        assert result["failures"] == 1
        assert result["avg_duration_sec"] == round(95.0 / 3, 2)

    def test_tickers_list(self, fresh_db):
        """Should list unique tickers."""
        metrics = SystemMetrics()
        today = datetime.today().strftime("%Y-%m-%d")

        metrics.log_pipeline_execution("VOO", "success", 40.0)
        metrics.log_pipeline_execution("SPY", "success", 50.0)
        metrics.log_pipeline_execution("VOO", "error", 5.0)

        result = metrics.get_daily_summary(today)
        assert len(result["tickers_processed"]) == 2
        assert "VOO" in result["tickers_processed"]
        assert "SPY" in result["tickers_processed"]


class TestHealthStatus:
    """Test health status determination."""

    def test_healthy_status(self, fresh_db):
        """Should return healthy for <5% errors."""
        metrics = SystemMetrics()
        for i in range(20):
            metrics.log_pipeline_execution(f"T{i}", "success", 40.0)

        result = metrics.get_health_status()
        assert result["status"] == "healthy"
        assert result["uptime_pct"] == 1.0
        assert result["error_rate"] == 0.0

    def test_degraded_status(self, fresh_db):
        """Should return degraded for 5-10% errors."""
        metrics = SystemMetrics()
        for i in range(20):
            metrics.log_pipeline_execution(f"S{i}", "success", 40.0)
        for i in range(2):  # 2/22 ≈ 9%
            metrics.log_pipeline_execution(f"E{i}", "error", 1.0)

        result = metrics.get_health_status()
        assert result["status"] == "degraded"

    def test_critical_status(self, fresh_db):
        """Should return critical for >10% errors."""
        metrics = SystemMetrics()
        for i in range(20):
            metrics.log_pipeline_execution(f"S{i}", "success", 40.0)
        for i in range(3):  # 3/23 ≈ 13%
            metrics.log_pipeline_execution(f"E{i}", "error", 1.0)

        result = metrics.get_health_status()
        assert result["status"] == "critical"

    def test_health_structure(self, fresh_db):
        """Should return required fields."""
        metrics = SystemMetrics()
        result = metrics.get_health_status()

        assert "status" in result
        assert "uptime_pct" in result
        assert "error_rate" in result
        assert "avg_response_time_sec" in result
        assert "last_check" in result
