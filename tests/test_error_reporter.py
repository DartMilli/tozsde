"""
Tests for ErrorReporter module.
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from app.infrastructure.error_reporter import (
    ErrorReporter,
    ErrorSeverity,
    ErrorRecord,
    ErrorStatistics
)


@pytest.fixture
def db_path():
    """Create a temporary database path."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def reporter(db_path):
    """Create a test reporter instance."""
    return ErrorReporter(db_path=db_path)


class TestErrorLogging:
    """Test error logging functionality."""
    
    def test_log_error_basic(self, reporter):
        """Test basic error logging."""
        try:
            raise ValueError("Test error")
        except Exception as e:
            result = reporter.log_error(e, severity=ErrorSeverity.ERROR)
        
        assert result is True
    
    def test_log_error_simple(self, reporter):
        """Test simple error logging."""
        result = reporter.log_error_simple(
            "TestError",
            "Test error message",
            severity=ErrorSeverity.WARNING
        )
        
        assert result is True
    
    def test_log_error_with_context(self, reporter):
        """Test error logging with context."""
        try:
            raise ValueError("Test error")
        except Exception as e:
            result = reporter.log_error(
                e,
                severity=ErrorSeverity.ERROR,
                context={'user_id': 123, 'action': 'test'}
            )
        
        assert result is True
        
        # Verify context was saved
        errors = reporter.get_recent_errors(limit=1)
        assert len(errors) == 1
        assert 'user_id' in errors[0].context
        assert errors[0].context['user_id'] == 123


class TestErrorRetrieval:
    """Test error retrieval functionality."""
    
    def test_get_recent_errors(self, reporter):
        """Test getting recent errors."""
        # Log multiple errors
        for i in range(5):
            reporter.log_error_simple(f"Error {i}", severity=ErrorSeverity.INFO)
        
        errors = reporter.get_recent_errors(limit=3)
        
        assert len(errors) == 3
        assert all(isinstance(e, ErrorRecord) for e in errors)
    
    def test_get_recent_errors_by_severity(self, reporter):
        """Test filtering by severity."""
        reporter.log_error_simple("Info", severity=ErrorSeverity.INFO)
        reporter.log_error_simple("Warning", severity=ErrorSeverity.WARNING)
        reporter.log_error_simple("Error", severity=ErrorSeverity.ERROR)
        
        errors = reporter.get_recent_errors(severity=ErrorSeverity.ERROR)
        
        assert len(errors) == 1
        assert errors[0].severity == ErrorSeverity.ERROR
    
    def test_get_recent_errors_by_type(self, reporter):
        """Test filtering by error type."""
        try:
            raise ValueError("Value error")
        except Exception as e:
            reporter.log_error(e)
        
        try:
            raise TypeError("Type error")
        except Exception as e:
            reporter.log_error(e)
        
        errors = reporter.get_recent_errors(error_type='ValueError')
        
        assert len(errors) == 1
        assert 'ValueError' in errors[0].error_type
    
    def test_get_recent_errors_empty(self, reporter):
        """Test getting errors when none exist."""
        errors = reporter.get_recent_errors()
        
        assert len(errors) == 0


class TestErrorStatistics:
    """Test error statistics functionality."""
    
    def test_get_statistics_basic(self, reporter):
        """Test basic statistics."""
        # Log various errors
        reporter.log_error_simple("Error 1", severity=ErrorSeverity.ERROR)
        reporter.log_error_simple("Error 2", severity=ErrorSeverity.WARNING)
        reporter.log_error_simple("Error 3", severity=ErrorSeverity.ERROR)
        
        stats = reporter.get_error_statistics(hours=24)
        
        assert isinstance(stats, ErrorStatistics)
        assert stats.total_errors == 3
        assert stats.by_severity['ERROR'] == 2
        assert stats.by_severity['WARNING'] == 1
    
    def test_get_statistics_empty(self, reporter):
        """Test statistics when no errors."""
        stats = reporter.get_error_statistics()
        
        assert stats.total_errors == 0
        assert stats.critical_errors == 0
        assert len(stats.by_severity) == 0
    
    def test_get_statistics_by_module(self, reporter):
        """Test statistics by module."""
        reporter.log_error_simple("Error", severity=ErrorSeverity.ERROR, module='module_a')
        reporter.log_error_simple("Error", severity=ErrorSeverity.ERROR, module='module_b')
        reporter.log_error_simple("Error", severity=ErrorSeverity.ERROR, module='module_a')
        
        stats = reporter.get_error_statistics()
        
        assert stats.by_module['module_a'] == 2
        assert stats.by_module['module_b'] == 1
    
    def test_get_statistics_critical_count(self, reporter):
        """Test critical error count."""
        reporter.log_error_simple("Critical 1", severity=ErrorSeverity.CRITICAL)
        reporter.log_error_simple("Critical 2", severity=ErrorSeverity.CRITICAL)
        reporter.log_error_simple("Error", severity=ErrorSeverity.ERROR)
        
        stats = reporter.get_error_statistics()
        
        assert stats.critical_errors == 2


class TestErrorRateMonitoring:
    """Test error rate monitoring."""
    
    def test_check_error_rate_normal(self, reporter):
        """Test error rate within threshold."""
        # Log a few errors (below threshold)
        for i in range(3):
            reporter.log_error_simple(f"Error {i}", severity=ErrorSeverity.ERROR)
        
        is_critical, rate = reporter.check_error_rate(hours=1, threshold=10)
        
        assert is_critical is False
        assert rate >= 0
    
    def test_check_error_rate_critical(self, reporter):
        """Test error rate exceeding threshold."""
        # Log many errors (above threshold)
        for i in range(15):
            reporter.log_error_simple(f"Error {i}", severity=ErrorSeverity.ERROR)
        
        is_critical, rate = reporter.check_error_rate(hours=1, threshold=10)
        
        assert is_critical is True
        assert rate > 10
    
    def test_check_critical_errors(self, reporter):
        """Test critical error count monitoring."""
        # Log several critical errors
        for i in range(6):
            reporter.log_error_simple(f"Critical {i}", severity=ErrorSeverity.CRITICAL)
        
        is_critical, count = reporter.check_critical_errors(hours=1, threshold=5)
        
        assert is_critical is True
        assert count == 6


class TestErrorTrends:
    """Test error trend analysis."""
    
    def test_get_error_trends_basic(self, reporter):
        """Test basic error trends."""
        # Log errors
        for i in range(5):
            reporter.log_error_simple(f"Error {i}", severity=ErrorSeverity.ERROR)
        
        trends = reporter.get_error_trends(days=1)
        
        assert len(trends) > 0
        assert all('date' in t and 'count' in t for t in trends)
    
    def test_get_error_trends_by_severity(self, reporter):
        """Test error trends by severity."""
        reporter.log_error_simple("Error 1", severity=ErrorSeverity.ERROR)
        reporter.log_error_simple("Warning 1", severity=ErrorSeverity.WARNING)
        
        trends = reporter.get_error_trends(days=1, severity=ErrorSeverity.ERROR)
        
        # Should only include ERROR severity
        assert len(trends) > 0
    
    def test_get_error_trends_empty(self, reporter):
        """Test trends when no errors."""
        trends = reporter.get_error_trends(days=1)
        
        assert len(trends) == 0


class TestErrorExport:
    """Test error export functionality."""
    
    def test_export_error_report(self, reporter, db_path):
        """Test exporting error report."""
        # Log some errors
        reporter.log_error_simple("Error 1", severity=ErrorSeverity.ERROR)
        reporter.log_error_simple("Error 2", severity=ErrorSeverity.WARNING)
        
        export_path = db_path.replace('.db', '_export.json')
        result = reporter.export_error_report(export_path, hours=24)
        
        assert result is True
        assert os.path.exists(export_path)
        
        # Cleanup
        if os.path.exists(export_path):
            os.remove(export_path)
    
    def test_export_empty_report(self, reporter, db_path):
        """Test exporting when no errors."""
        export_path = db_path.replace('.db', '_export.json')
        result = reporter.export_error_report(export_path, hours=24)
        
        assert result is True
        assert os.path.exists(export_path)
        
        # Cleanup
        if os.path.exists(export_path):
            os.remove(export_path)


class TestErrorCleanup:
    """Test error cleanup functionality."""
    
    def test_clear_old_errors(self, reporter):
        """Test clearing old errors."""
        # Log errors with old timestamp
        reporter.log_error_simple("Old error", severity=ErrorSeverity.ERROR)
        
        # Manually update timestamp to be old
        conn = reporter._get_connection()
        old_date = (datetime.now() - timedelta(days=60)).isoformat()
        conn.execute('UPDATE errors SET timestamp = ?', (old_date,))
        conn.commit()
        conn.close()
        
        # Clear old errors (older than 30 days)
        count = reporter.clear_old_errors(days=30)
        
        assert count > 0
    
    def test_clear_old_errors_none(self, reporter):
        """Test clearing when no old errors."""
        # Log recent error
        reporter.log_error_simple("Recent error", severity=ErrorSeverity.ERROR)
        
        # Try to clear old errors
        count = reporter.clear_old_errors(days=30)
        
        assert count == 0


class TestSeverityEnum:
    """Test ErrorSeverity enum."""
    
    def test_severity_values(self):
        """Test severity enum values."""
        assert ErrorSeverity.DEBUG.value == 'DEBUG'
        assert ErrorSeverity.INFO.value == 'INFO'
        assert ErrorSeverity.WARNING.value == 'WARNING'
        assert ErrorSeverity.ERROR.value == 'ERROR'
        assert ErrorSeverity.CRITICAL.value == 'CRITICAL'
    
    def test_severity_comparison(self):
        """Test severity comparison."""
        # Enums are comparable by value
        assert ErrorSeverity.CRITICAL.value > ErrorSeverity.ERROR.value
        assert ErrorSeverity.WARNING.value > ErrorSeverity.INFO.value


class TestEdgeCases:
    """Test edge cases."""
    
    def test_log_without_database(self):
        """Test logging without database path."""
        reporter = ErrorReporter(db_path=None)
        
        result = reporter.log_error_simple("Test", severity=ErrorSeverity.ERROR)
        
        # Should still work (logs to console)
        assert result is False  # No db path
    
    def test_concurrent_logging(self, reporter):
        """Test concurrent error logging."""
        # Log multiple errors quickly
        results = []
        for i in range(20):
            result = reporter.log_error_simple(f"Error {i}", severity=ErrorSeverity.ERROR)
            results.append(result)
        
        assert all(results)
        
        # Verify all logged
        errors = reporter.get_recent_errors(limit=20)
        assert len(errors) == 20
    
    def test_large_context_data(self, reporter):
        """Test logging with large context data."""
        large_context = {
            'data': 'x' * 10000,  # 10KB of data
            'nested': {'key': 'value' * 100}
        }
        
        result = reporter.log_error_simple(
            "Error with large context",
            severity=ErrorSeverity.ERROR,
            context=large_context
        )
        
        assert result is True
    
    def test_special_characters_in_message(self, reporter):
        """Test error message with special characters."""
        message = "Error with special chars: 中文, émojis 😀, quotes \"'`"
        
        result = reporter.log_error_simple(message, severity=ErrorSeverity.ERROR)
        
        assert result is True
        
        errors = reporter.get_recent_errors(limit=1)
        assert len(errors) == 1
        assert '中文' in errors[0].message


class TestMostCommonErrors:
    """Test most common error tracking."""
    
    def test_most_common_errors(self, reporter):
        """Test identifying most common errors."""
        # Log multiple instances of same error
        for i in range(5):
            reporter.log_error_simple("Common error A", severity=ErrorSeverity.ERROR)
        
        for i in range(3):
            reporter.log_error_simple("Common error B", severity=ErrorSeverity.ERROR)
        
        reporter.log_error_simple("Rare error", severity=ErrorSeverity.ERROR)
        
        stats = reporter.get_error_statistics()
        
        assert len(stats.most_common) > 0
        # Most common should be "Common error A"
        assert stats.most_common[0][1] == 5
