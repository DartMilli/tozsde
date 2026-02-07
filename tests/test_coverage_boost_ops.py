import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

import pytest

from app.infrastructure.error_reporter import ErrorReporter, ErrorSeverity
from app.infrastructure.health_check import HealthChecker
from app.config.config import Config


def test_error_reporter_basic_flow(tmp_path):
    db_path = tmp_path / "errors.db"
    reporter = ErrorReporter(
        db_path=str(db_path), critical_threshold=1, error_rate_threshold=0.1
    )

    assert reporter.log_error_simple("TestError", "oops", severity=ErrorSeverity.ERROR)
    stats = reporter.get_error_statistics(hours_back=24)
    assert stats.total_errors >= 1

    recent = reporter.get_recent_errors(limit=5)
    assert len(recent) >= 1

    assert reporter.check_error_rate() is True or reporter.check_error_rate() is False

    out_file = tmp_path / "report.json"
    assert reporter.export_error_report(str(out_file), days_back=1) is True

    trends = reporter.get_error_trends(days=1)
    assert "counts" in trends

    deleted = reporter.clear_old_errors(days_to_keep=0)
    assert deleted >= 0


def test_health_checker_basic(monkeypatch, tmp_path):
    db_path = tmp_path / "health.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE IF NOT EXISTS dummy (id INTEGER)")
    conn.commit()
    conn.close()

    monkeypatch.setattr(Config, "DB_PATH", db_path)
    monkeypatch.setattr(Config, "LOG_DIR", tmp_path)

    checker = HealthChecker()

    class DummyResponse:
        def getcode(self):
            return 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: DummyResponse())

    class DummyDisk:
        total = 100
        free = 50

    class DummyMem:
        percent = 50

    monkeypatch.setattr("psutil.disk_usage", lambda *a, **k: DummyDisk())
    monkeypatch.setattr("psutil.virtual_memory", lambda *a, **k: DummyMem())
    monkeypatch.setattr("psutil.cpu_percent", lambda *a, **k: 10.0)

    api = checker.check_api_health()
    assert api["healthy"] is True

    db = checker.check_database_health()
    assert db["healthy"] is True

    disk = checker.check_disk_space()
    mem = checker.check_memory_usage()
    cpu = checker.check_cpu_usage()

    assert "healthy" in disk
    assert "healthy" in mem
    assert "healthy" in cpu

    status = checker.check_all()
    assert "checks" in status
