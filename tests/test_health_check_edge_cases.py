"""Additional edge case tests for HealthChecker error paths."""

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import app.infrastructure.health_check as health_check
from app.infrastructure.health_check import HealthChecker


@pytest.fixture
def checker(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    data_dir = tmp_path / "data"
    log_dir.mkdir()
    data_dir.mkdir()
    db_path = data_dir / "test.db"

    mock_config = SimpleNamespace(LOG_DIR=log_dir, DB_PATH=db_path)
    monkeypatch.setattr(health_check, "Config", mock_config)

    return HealthChecker()


def test_check_api_health_unexpected_exception(checker):
    with patch("urllib.request.urlopen", side_effect=ValueError("boom")):
        result = checker.check_api_health()

    assert result["healthy"] is False
    assert "API check failed" in result["message"]


def test_check_database_health_exception(checker):
    with patch("sqlite3.connect", side_effect=Exception("db fail")):
        result = checker.check_database_health()

    assert result["healthy"] is False
    assert "Database check failed" in result["message"]


def test_check_disk_space_exception(checker):
    with patch("psutil.disk_usage", side_effect=Exception("disk fail")):
        result = checker.check_disk_space()

    assert result["healthy"] is False
    assert "Disk check failed" in result["message"]


def test_check_memory_usage_exception(checker):
    with patch("psutil.virtual_memory", side_effect=Exception("mem fail")):
        result = checker.check_memory_usage()

    assert result["healthy"] is False
    assert "Memory check failed" in result["message"]


def test_check_cpu_usage_exception(checker):
    with patch("psutil.cpu_percent", side_effect=Exception("cpu fail")):
        result = checker.check_cpu_usage()

    assert result["healthy"] is False
    assert "CPU check failed" in result["message"]


def test_check_cron_execution_invalid_timestamp(checker):
    metrics_log = Path(checker.log_dir) / "metrics.jsonl"

    log_entry = {
        "event": "pipeline_execution",
        "timestamp": "INVALID_TIMESTAMP",
    }

    metrics_log.write_text(json.dumps(log_entry) + "\n")

    result = checker.check_cron_execution()

    assert result["healthy"] is False
    assert "Cron check failed" in result["message"]


def test_get_recent_health_logs_skips_invalid_lines(checker):
    health_log = Path(checker.log_dir) / "health_check.jsonl"

    valid_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "healthy": True,
    }
    invalid_json = "{not-json}"
    invalid_timestamp = {"timestamp": "bad", "healthy": True}

    with open(health_log, "w") as f:
        f.write(invalid_json + "\n")
        f.write(json.dumps(invalid_timestamp) + "\n")
        f.write(json.dumps(valid_entry) + "\n")

    logs = checker.get_recent_health_logs(hours=24)

    assert len(logs) == 1
    assert logs[0]["healthy"] is True


def test_log_health_status_write_failure(checker):
    status = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "healthy": True,
        "issues": [],
    }

    with patch("builtins.open", side_effect=OSError("write fail")):
        checker._log_health_status(status)
