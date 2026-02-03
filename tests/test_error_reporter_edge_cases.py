"""Edge case tests for ErrorReporter (Sprint 10 Week 3)."""

import json
import os
import tempfile
from datetime import datetime, timedelta

from app.infrastructure.error_reporter import ErrorReporter, ErrorSeverity


def test_error_reporter_no_db_path():
    """Should handle missing db_path gracefully."""
    reporter = ErrorReporter(db_path=None)
    stats = reporter.get_error_statistics(hours_back=1)
    assert stats.total_errors == 0
    assert reporter.get_recent_errors() == []
    assert reporter.get_error_trends(days=3) == {"dates": [], "counts": [], "severity_breakdown": {}}
    assert reporter.export_error_report("/tmp/should_not_exist.json") is False


def test_log_and_stats_with_db():
    """Should log errors and calculate statistics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "errors.db")
        reporter = ErrorReporter(db_path=db_path)

        reporter.log_error_simple(
            error_type="ValueError",
            error_message="Test error",
            severity=ErrorSeverity.ERROR,
            module="test_module",
            function="test_func",
        )

        stats = reporter.get_error_statistics(hours_back=24)
        assert stats.total_errors == 1
        assert stats.errors_by_severity.get(ErrorSeverity.ERROR.value, 0) == 1
        assert stats.most_common_error == "ValueError"


def test_recent_errors_filtering():
    """Should return recent errors and filter by severity."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "errors.db")
        reporter = ErrorReporter(db_path=db_path)

        reporter.log_error_simple("TypeA", "msg1", severity=ErrorSeverity.ERROR)
        reporter.log_error_simple("TypeB", "msg2", severity=ErrorSeverity.CRITICAL)

        all_errors = reporter.get_recent_errors(limit=10)
        crit_errors = reporter.get_recent_errors(limit=10, severity=ErrorSeverity.CRITICAL)

        assert len(all_errors) == 2
        assert len(crit_errors) == 1
        assert crit_errors[0]["severity"] == ErrorSeverity.CRITICAL.value


def test_error_trends_and_cleanup():
    """Should generate trends and cleanup old errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "errors.db")
        reporter = ErrorReporter(db_path=db_path)

        # Log two errors
        reporter.log_error_simple("TypeA", "msg1", severity=ErrorSeverity.ERROR)
        reporter.log_error_simple("TypeB", "msg2", severity=ErrorSeverity.WARNING)

        trends = reporter.get_error_trends(days=7)
        assert "dates" in trends
        assert "counts" in trends

        # Cleanup should not fail
        deleted = reporter.clear_old_errors(days_to_keep=0)
        assert deleted >= 0


def test_export_error_report():
    """Should export a JSON report."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "errors.db")
        out_path = os.path.join(tmpdir, "report.json")
        reporter = ErrorReporter(db_path=db_path)

        reporter.log_error_simple("TypeA", "msg1", severity=ErrorSeverity.ERROR)

        success = reporter.export_error_report(out_path, days_back=1)
        assert success is True

        with open(out_path, "r") as f:
            data = json.load(f)

        assert "errors" in data
        assert data["total_errors"] >= 1
