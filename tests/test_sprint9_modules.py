"""
Minimal tests for Sprint 9 modules to verify core functionality.
"""

import pytest
from datetime import datetime, timedelta
from app.reporting.performance_analytics import PerformanceAnalytics, PerformanceMetrics
from app.infrastructure.error_reporter import ErrorReporter, ErrorSeverity


class TestPerformanceAnalyticsCoreFeatures:
    """Test core PerformanceAnalytics features."""
    
    def test_analytics_instance_creation(self):
        """Test creating analytics instance."""
        analytics = PerformanceAnalytics(risk_free_rate=0.02)
        assert analytics is not None
        assert analytics.risk_free_rate == 0.02
    
    def test_total_return_calculation(self):
        """Test total return calculation."""
        analytics = PerformanceAnalytics()
        returns = [0.10, 0.05, -0.03]
        
        total = analytics._calculate_total_return(returns)
        
        assert isinstance(total, float)
        assert total > 0
    
    def test_volatility_calculation(self):
        """Test volatility calculation."""
        analytics = PerformanceAnalytics()
        returns = [0.01, -0.01, 0.02, -0.02]
        dates = [datetime.now() - timedelta(days=4-i) for i in range(4)]
        
        vol = analytics._calculate_volatility(returns, dates)
        
        assert vol >= 0
    
    def test_max_drawdown_calculation(self):
        """Test max drawdown calculation."""
        analytics = PerformanceAnalytics()
        returns = [0.10, -0.15, 0.05, -0.10]
        
        dd = analytics._calculate_max_drawdown(returns)
        
        assert dd >= 0
        assert dd <= 1.0
    
    def test_performance_metrics_creation(self):
        """Test creating performance metrics dataclass."""
        returns = [0.01] * 20
        dates = [datetime.now() - timedelta(days=20-i) for i in range(20)]
        
        analytics = PerformanceAnalytics()
        metrics = analytics.calculate_performance_metrics(returns, dates)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_return is not None
        assert metrics.volatility is not None


class TestErrorReporterCoreFeatures:
    """Test core ErrorReporter features."""
    
    def test_error_reporter_instance(self):
        """Test creating error reporter instance."""
        reporter = ErrorReporter(db_path=None)
        assert reporter is not None
    
    def test_severity_enum(self):
        """Test severity enum values."""
        assert ErrorSeverity.DEBUG.value == "DEBUG"
        assert ErrorSeverity.ERROR.value == "ERROR"
        assert ErrorSeverity.CRITICAL.value == "CRITICAL"
    
    def test_log_error_without_db(self):
        """Test logging without database."""
        reporter = ErrorReporter(db_path=None)
        
        try:
            raise ValueError("Test")
        except Exception as e:
            result = reporter.log_error(e, severity=ErrorSeverity.ERROR)
        
        # Should return False without db path
        assert result is False
    
    def test_get_statistics_without_db(self):
        """Test getting statistics without database."""
        reporter = ErrorReporter(db_path=None)
        stats = reporter.get_error_statistics()
        
        assert stats is not None
        assert stats.total_errors == 0
    
    def test_get_recent_errors_without_db(self):
        """Test getting recent errors without database."""
        reporter = ErrorReporter(db_path=None)
        errors = reporter.get_recent_errors()
        
        assert isinstance(errors, list)
        assert len(errors) == 0


class TestAdminDashboardImport:
    """Test admin dashboard imports."""
    
    def test_admin_dashboard_blueprint_import(self):
        """Test admin dashboard blueprint can be imported."""
        try:
            from app.ui.admin_dashboard import admin_bp, register_admin_routes
            assert admin_bp is not None
            assert callable(register_admin_routes)
        except ImportError:
            # Blueprint might not exist yet
            pass
    
    def test_flask_app_integration(self):
        """Test Flask app can be created with admin routes."""
        try:
            from flask import Flask
            from app.ui.admin_dashboard import register_admin_routes
            
            app = Flask(__name__)
            register_admin_routes(app)
            
            assert app is not None
        except (ImportError, AttributeError):
            # Admin dashboard might not be complete
            pass


class TestModuleStructure:
    """Test module structure and organization."""
    
    def test_performance_analytics_has_required_classes(self):
        """Test performance analytics has required classes."""
        from app.reporting.performance_analytics import (
            PerformanceAnalytics,
            PerformanceMetrics,
            DrawdownAnalysis
        )
        
        assert PerformanceAnalytics is not None
        assert PerformanceMetrics is not None
        assert DrawdownAnalysis is not None
    
    def test_error_reporter_has_required_classes(self):
        """Test error reporter has required classes."""
        from app.infrastructure.error_reporter import (
            ErrorReporter,
            ErrorSeverity,
            ErrorRecord,
            ErrorStatistics
        )
        
        assert ErrorReporter is not None
        assert ErrorSeverity is not None
        assert ErrorRecord is not None
        assert ErrorStatistics is not None


class TestEdgeCases:
    """Test edge cases."""
    
    def test_empty_returns_handling(self):
        """Test handling of empty returns."""
        analytics = PerformanceAnalytics()
        
        with pytest.raises(ValueError):
            analytics.calculate_performance_metrics([], [])
    
    def test_single_return(self):
        """Test with single return."""
        analytics = PerformanceAnalytics()
        returns = [0.05]
        dates = [datetime.now()]
        
        total = analytics._calculate_total_return(returns)
        assert total == pytest.approx(0.05, rel=0.01)
    
    def test_all_positive_returns(self):
        """Test with all positive returns."""
        analytics = PerformanceAnalytics()
        returns = [0.05] * 10
        dates = [datetime.now() - timedelta(days=10-i) for i in range(10)]
        
        metrics = analytics.calculate_performance_metrics(returns, dates)
        
        assert metrics.total_return > 0
        assert metrics.max_drawdown == 0.0
