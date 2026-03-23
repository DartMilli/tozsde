"""
Unit Tests for Health Check Module (Sprint 5)

Tests health monitoring functionality for Raspberry Pi deployment:
- API health checks
- Database connectivity
- System resource monitoring
- Cron job tracking
- Alert generation
"""

import json
import os
import pytest
import sqlite3
import tempfile
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import urllib.error

from app.infrastructure.health_check import HealthChecker


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / "logs"
        data_dir = Path(tmpdir) / "data"
        log_dir.mkdir()
        data_dir.mkdir()

        yield {
            "log_dir": log_dir,
            "data_dir": data_dir,
            "db_path": data_dir / "test.db",
        }


@pytest.fixture
def health_checker(temp_dirs, test_settings):
    """Create HealthChecker instance with temporary paths."""
    settings = replace(
        test_settings,
        LOG_DIR=temp_dirs["log_dir"],
        DB_PATH=temp_dirs["db_path"],
    )
    return HealthChecker(settings=settings)


# --- API Health Check Tests ---


def test_check_api_health_success(health_checker):
    """Test successful API health check."""
    mock_response = MagicMock()
    mock_response.getcode.return_value = 200
    mock_response.__enter__ = Mock(return_value=mock_response)
    mock_response.__exit__ = Mock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_response):
        result = health_checker.check_api_health()

    assert result["healthy"] is True
    assert result["status_code"] == 200
    assert result["response_time"] is not None
    assert "API responding" in result["message"]


def test_check_api_health_non_200_status(health_checker):
    """Test API health check with non-200 status code."""
    mock_response = MagicMock()
    mock_response.getcode.return_value = 500
    mock_response.__enter__ = Mock(return_value=mock_response)
    mock_response.__exit__ = Mock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_response):
        result = health_checker.check_api_health()

    assert result["healthy"] is False
    assert result["status_code"] == 500
    assert "non-200 status" in result["message"]


def test_check_api_health_connection_error(health_checker):
    """Test API health check when connection fails."""
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("Connection refused"),
    ):
        result = health_checker.check_api_health()

    assert result["healthy"] is False
    assert result["status_code"] is None
    assert "not reachable" in result["message"]


# --- Database Health Check Tests ---


def test_check_database_health_success(health_checker, temp_dirs):
    """Test successful database health check."""
    # Create and initialize database
    conn = sqlite3.connect(str(temp_dirs["db_path"]))
    conn.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()

    result = health_checker.check_database_health()

    assert result["healthy"] is True
    assert result["writable"] is True
    assert result["tables"] >= 1
    assert "Database healthy" in result["message"]


def test_check_database_health_missing_db(health_checker):
    """Test database health check when database doesn't exist."""
    # Don't create database file
    result = health_checker.check_database_health()

    # SQLite creates file on connect, so we should get a healthy result
    # but with 0 tables
    assert result["tables"] == 0


def test_check_database_health_locked(health_checker, temp_dirs):
    """Test database health check when database is locked."""
    # Create database and keep it locked
    conn = sqlite3.connect(str(temp_dirs["db_path"]))
    conn.execute("CREATE TABLE test (id INTEGER)")
    conn.execute("BEGIN EXCLUSIVE")

    # Try to check health while locked
    with patch.object(health_checker, "db_path", str(temp_dirs["db_path"])):
        with patch(
            "sqlite3.connect",
            side_effect=sqlite3.OperationalError("database is locked"),
        ):
            result = health_checker.check_database_health()

    conn.close()

    assert result["healthy"] is False
    assert result["writable"] is False
    assert "locked" in result["message"].lower()


# --- Disk Space Check Tests ---


@patch("psutil.disk_usage")
def test_check_disk_space_healthy(mock_disk_usage, health_checker):
    """Test disk space check with sufficient space."""
    mock_usage = Mock()
    mock_usage.percent = 50.0  # 50% used = 50% free (healthy)
    mock_usage.free = 50 * 1024**3  # 50 GB free
    mock_usage.total = 100 * 1024**3  # 100 GB total
    mock_disk_usage.return_value = mock_usage

    result = health_checker.check_disk_space()

    assert result["healthy"] is True
    assert result["free_pct"] == 50.0
    assert result["free_gb"] == 50.0


@patch("psutil.disk_usage")
def test_check_disk_space_low(mock_disk_usage, health_checker):
    """Test disk space check with low space."""
    mock_usage = Mock()
    mock_usage.percent = 95.0  # 95% used = 5% free (unhealthy)
    mock_usage.free = 5 * 1024**3  # 5 GB free
    mock_usage.total = 100 * 1024**3
    mock_disk_usage.return_value = mock_usage

    result = health_checker.check_disk_space()

    assert result["healthy"] is False
    assert result["free_pct"] == 5.0
    assert "Low disk space" in result["message"]


# --- Memory Usage Check Tests ---


@patch("psutil.virtual_memory")
def test_check_memory_usage_healthy(mock_memory, health_checker):
    """Test memory check with normal usage."""
    mock_mem = Mock()
    mock_mem.percent = 50.0  # 50% used (healthy)
    mock_mem.available = 4 * 1024**3  # 4 GB available
    mock_mem.total = 8 * 1024**3  # 8 GB total
    mock_memory.return_value = mock_mem

    result = health_checker.check_memory_usage()

    assert result["healthy"] is True
    assert result["used_pct"] == 50.0
    assert result["available_gb"] == 4.0


@patch("psutil.virtual_memory")
def test_check_memory_usage_high(mock_memory, health_checker):
    """Test memory check with high usage."""
    mock_mem = Mock()
    mock_mem.percent = 90.0  # 90% used (unhealthy)
    mock_mem.available = 0.8 * 1024**3  # 0.8 GB available
    mock_mem.total = 8 * 1024**3
    mock_memory.return_value = mock_mem

    result = health_checker.check_memory_usage()

    assert result["healthy"] is False
    assert result["used_pct"] == 90.0
    assert "High memory usage" in result["message"]


# --- CPU Usage Check Tests ---


@patch("psutil.cpu_percent")
def test_check_cpu_usage_healthy(mock_cpu, health_checker):
    """Test CPU check with normal usage."""
    mock_cpu.return_value = 30.0  # 30% used (healthy)

    result = health_checker.check_cpu_usage()

    assert result["healthy"] is True
    assert result["used_pct"] == 30.0


@patch("psutil.cpu_percent")
def test_check_cpu_usage_high(mock_cpu, health_checker):
    """Test CPU check with high usage."""
    mock_cpu.return_value = 95.0  # 95% used (unhealthy)

    result = health_checker.check_cpu_usage()

    assert result["healthy"] is False
    assert result["used_pct"] == 95.0
    assert "High CPU usage" in result["message"]


# --- Cron Execution Check Tests ---


def test_check_cron_execution_recent(health_checker, temp_dirs):
    """Test cron check with recent pipeline execution."""
    metrics_log = temp_dirs["log_dir"] / "metrics.jsonl"

    # Create metrics log with recent execution
    recent_time = datetime.now(timezone.utc) - timedelta(hours=1)
    log_entry = {
        "event": "pipeline_execution",
        "timestamp": recent_time.isoformat(),
        "status": "success",
    }

    with open(metrics_log, "w") as f:
        f.write(json.dumps(log_entry) + "\n")

    result = health_checker.check_cron_execution()

    assert result["healthy"] is True
    assert result["hours_ago"] < 25  # Within threshold


def test_check_cron_execution_overdue(health_checker, temp_dirs):
    """Test cron check with overdue pipeline execution."""
    metrics_log = temp_dirs["log_dir"] / "metrics.jsonl"

    # Create metrics log with old execution
    old_time = datetime.now(timezone.utc) - timedelta(hours=30)
    log_entry = {
        "event": "pipeline_execution",
        "timestamp": old_time.isoformat(),
        "status": "success",
    }

    with open(metrics_log, "w") as f:
        f.write(json.dumps(log_entry) + "\n")

    result = health_checker.check_cron_execution()

    assert result["healthy"] is False
    assert result["hours_ago"] > 25
    assert "overdue" in result["message"].lower()


def test_check_cron_execution_no_log(health_checker):
    """Test cron check when no metrics log exists."""
    result = health_checker.check_cron_execution()

    assert result["healthy"] is False
    assert result["last_run"] is None
    assert "No metrics log found" in result["message"]


# --- Overall Health Check Tests ---


@patch("psutil.cpu_percent")
@patch("psutil.virtual_memory")
@patch("psutil.disk_usage")
def test_check_all_healthy(mock_disk, mock_memory, mock_cpu, health_checker, temp_dirs):
    """Test overall health check when all systems healthy."""
    # Mock system resources
    mock_cpu.return_value = 30.0
    mock_memory.return_value = Mock(
        percent=50.0, available=4 * 1024**3, total=8 * 1024**3
    )
    mock_disk.return_value = Mock(percent=50.0, free=50 * 1024**3, total=100 * 1024**3)

    # Create database
    conn = sqlite3.connect(str(temp_dirs["db_path"]))
    conn.execute("CREATE TABLE test (id INTEGER)")
    conn.close()

    # Create recent metrics log
    metrics_log = temp_dirs["log_dir"] / "metrics.jsonl"
    log_entry = {
        "event": "pipeline_execution",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(metrics_log, "w") as f:
        f.write(json.dumps(log_entry) + "\n")

    # Mock API check
    with patch.object(
        health_checker,
        "check_api_health",
        return_value={"healthy": True, "status_code": 200, "message": "OK"},
    ):
        result = health_checker.check_all()

    assert result["healthy"] is True
    assert len(result["issues"]) == 0
    assert "api" in result["checks"]
    assert "database" in result["checks"]


def test_check_all_with_issues(health_checker):
    """Test overall health check with multiple issues."""
    # Mock failing API
    with patch.object(
        health_checker,
        "check_api_health",
        return_value={"healthy": False, "message": "API down"},
    ):
        with patch.object(
            health_checker,
            "check_database_health",
            return_value={"healthy": False, "message": "DB locked"},
        ):
            with patch.object(
                health_checker,
                "check_disk_space",
                return_value={"healthy": True, "message": "OK"},
            ):
                with patch.object(
                    health_checker,
                    "check_memory_usage",
                    return_value={"healthy": True, "message": "OK"},
                ):
                    with patch.object(
                        health_checker,
                        "check_cpu_usage",
                        return_value={"healthy": True, "message": "OK"},
                    ):
                        with patch.object(
                            health_checker,
                            "check_cron_execution",
                            return_value={"healthy": True, "message": "OK"},
                        ):
                            result = health_checker.check_all()

    assert result["healthy"] is False
    assert len(result["issues"]) == 2
    assert any("api" in issue.lower() for issue in result["issues"])
    assert any("database" in issue.lower() for issue in result["issues"])


# --- Alert Generation Tests ---


def test_generate_alert_with_issues(health_checker):
    """Test alert generation when issues present."""
    status = {
        "healthy": False,
        "issues": ["api: API down", "database: DB locked"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    alert = health_checker.generate_alert(status)

    assert alert is not None
    assert "ALERT" in alert
    assert "2 issue(s)" in alert
    assert "api: API down" in alert
    assert "database: DB locked" in alert


def test_generate_alert_healthy(health_checker):
    """Test alert generation when healthy."""
    status = {
        "healthy": True,
        "issues": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    alert = health_checker.generate_alert(status)

    assert alert is None


# --- Health Log Tests ---


def test_log_health_status(health_checker, temp_dirs):
    """Test health status logging to JSONL."""
    status = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "healthy": True,
        "issues": [],
    }

    health_checker._log_health_status(status)

    health_log = temp_dirs["log_dir"] / "health_check.jsonl"
    assert health_log.exists()

    with open(health_log, "r") as f:
        log_line = f.readline()
        logged_status = json.loads(log_line)
        assert logged_status["healthy"] is True


def test_get_recent_health_logs(health_checker, temp_dirs):
    """Test retrieving recent health logs."""
    health_log = temp_dirs["log_dir"] / "health_check.jsonl"

    # Write 3 log entries
    for i in range(3):
        time_offset = datetime.now(timezone.utc) - timedelta(hours=i)
        log_entry = {"timestamp": time_offset.isoformat(), "healthy": True}
        with open(health_log, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    logs = health_checker.get_recent_health_logs(hours=24)

    assert len(logs) == 3
    assert all(log["healthy"] is True for log in logs)
