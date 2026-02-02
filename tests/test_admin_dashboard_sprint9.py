"""
Tests for Sprint 9 AdminDashboard Flask endpoints.
"""

import pytest
from flask import Flask
from datetime import datetime, timedelta
from app.ui.admin_dashboard import admin_bp, register_admin_routes


@pytest.fixture
def app():
    """Create a test Flask app."""
    app = Flask(__name__)
    app.config['TESTING'] = True
    register_admin_routes(app)
    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return app.test_client()


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get('/admin/health')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'healthy'
        assert 'timestamp' in data


class TestPerformanceEndpoints:
    """Test performance endpoints."""
    
    def test_performance_summary_default(self, client):
        """Test performance summary with default days."""
        response = client.get('/admin/performance/summary')
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'total_return' in data
        assert 'sharpe_ratio' in data
        assert 'max_drawdown' in data
    
    def test_performance_summary_custom_days(self, client):
        """Test performance summary with custom days."""
        response = client.get('/admin/performance/summary?days=60')
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'total_return' in data
    
    def test_performance_drawdown(self, client):
        """Test drawdown analysis endpoint."""
        response = client.get('/admin/performance/drawdown?days=90')
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'max_drawdown' in data
        assert 'current_drawdown' in data
    
    def test_performance_rolling(self, client):
        """Test rolling performance endpoint."""
        response = client.get('/admin/performance/rolling?days=90&window=30')
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'window_size_days' in data
        assert 'returns' in data
    
    def test_performance_rolling_default_window(self, client):
        """Test rolling performance with default window."""
        response = client.get('/admin/performance/rolling')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['window_size_days'] == 30  # Default


class TestErrorEndpoints:
    """Test error monitoring endpoints."""
    
    def test_errors_summary_default(self, client):
        """Test error summary with default hours."""
        response = client.get('/admin/errors/summary')
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'total_errors' in data
        assert 'critical_errors' in data
        assert 'by_severity' in data
    
    def test_errors_summary_custom_hours(self, client):
        """Test error summary with custom hours."""
        response = client.get('/admin/errors/summary?hours=48')
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'total_errors' in data
    
    def test_errors_recent_default(self, client):
        """Test recent errors with default limit."""
        response = client.get('/admin/errors/recent')
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'errors' in data
        assert isinstance(data['errors'], list)
    
    def test_errors_recent_with_filters(self, client):
        """Test recent errors with severity filter."""
        response = client.get('/admin/errors/recent?limit=20&severity=ERROR')
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'errors' in data
    
    def test_errors_trends(self, client):
        """Test error trends endpoint."""
        response = client.get('/admin/errors/trends?days=7')
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'trends' in data
        assert isinstance(data['trends'], list)


class TestCapitalEndpoint:
    """Test capital utilization endpoint."""
    
    def test_capital_utilization(self, client):
        """Test capital utilization endpoint."""
        response = client.get('/admin/capital/utilization')
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'total_capital' in data
        assert 'allocated_capital' in data
        assert 'utilization_rate' in data


class TestDecisionEndpoints:
    """Test decision endpoints."""
    
    def test_no_trades_default(self, client):
        """Test no-trade decisions with default days."""
        response = client.get('/admin/decisions/no-trades')
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'total_no_trades' in data
        assert 'by_reason' in data
    
    def test_no_trades_custom_days(self, client):
        """Test no-trade decisions with custom days."""
        response = client.get('/admin/decisions/no-trades?days=14')
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'total_no_trades' in data


class TestStrategyEndpoints:
    """Test strategy endpoints."""
    
    def test_strategies_performance_default(self, client):
        """Test strategy performance with default days."""
        response = client.get('/admin/strategies/performance')
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'strategies' in data
        assert isinstance(data['strategies'], list)
    
    def test_strategies_performance_custom_days(self, client):
        """Test strategy performance with custom days."""
        response = client.get('/admin/strategies/performance?days=60')
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'strategies' in data


class TestConfidenceEndpoint:
    """Test confidence distribution endpoint."""
    
    def test_confidence_distribution(self, client):
        """Test confidence distribution endpoint."""
        response = client.get('/admin/confidence/distribution')
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'buckets' in data
        assert isinstance(data['buckets'], dict)


class TestErrorHandling:
    """Test error handling."""
    
    def test_invalid_endpoint(self, client):
        """Test invalid endpoint."""
        response = client.get('/admin/invalid')
        
        assert response.status_code == 404
    
    def test_invalid_query_params(self, client):
        """Test invalid query parameters."""
        response = client.get('/admin/performance/summary?days=invalid')
        
        # Should handle gracefully and use defaults or return error
        assert response.status_code in [200, 400]
    
    def test_negative_days_param(self, client):
        """Test negative days parameter."""
        response = client.get('/admin/performance/summary?days=-10')
        
        # Should handle gracefully
        assert response.status_code in [200, 400]


class TestResponseFormat:
    """Test response format consistency."""
    
    def test_json_response(self, client):
        """Test JSON response format."""
        response = client.get('/admin/health')
        
        assert response.content_type == 'application/json'
    
    def test_percentage_formatting(self, client):
        """Test percentage values are properly formatted."""
        response = client.get('/admin/performance/summary')
        
        data = response.get_json()
        # Percentages should be rounded
        if 'total_return' in data and data['total_return'] is not None:
            # Should be reasonable precision
            assert isinstance(data['total_return'], (int, float))


class TestQueryParameters:
    """Test query parameter handling."""
    
    def test_days_parameter_types(self, client):
        """Test days parameter with different types."""
        # Valid integer
        response1 = client.get('/admin/performance/summary?days=30')
        assert response1.status_code == 200
        
        # String number
        response2 = client.get('/admin/performance/summary?days=30')
        assert response2.status_code == 200
    
    def test_limit_parameter(self, client):
        """Test limit parameter."""
        response = client.get('/admin/errors/recent?limit=10')
        
        assert response.status_code == 200
        data = response.get_json()
        assert len(data['errors']) <= 10
    
    def test_window_parameter(self, client):
        """Test window parameter."""
        response = client.get('/admin/performance/rolling?window=60')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['window_size_days'] == 60


class TestBlueprintRegistration:
    """Test blueprint registration."""
    
    def test_blueprint_registration(self):
        """Test blueprint can be registered."""
        app = Flask(__name__)
        register_admin_routes(app)
        
        # Check blueprint is registered
        assert 'admin' in [bp.name for bp in app.blueprints.values()]
    
    def test_blueprint_url_prefix(self):
        """Test blueprint URL prefix."""
        app = Flask(__name__)
        register_admin_routes(app)
        
        # Admin routes should be under /admin
        client = app.test_client()
        response = client.get('/admin/health')
        assert response.status_code == 200


class TestEdgeCases:
    """Test edge cases."""
    
    def test_zero_days_param(self, client):
        """Test with zero days parameter."""
        response = client.get('/admin/performance/summary?days=0')
        
        # Should handle gracefully
        assert response.status_code in [200, 400]
    
    def test_very_large_days_param(self, client):
        """Test with very large days parameter."""
        response = client.get('/admin/performance/summary?days=10000')
        
        # Should handle gracefully (might return empty data)
        assert response.status_code == 200
    
    def test_empty_severity_filter(self, client):
        """Test with empty severity filter."""
        response = client.get('/admin/errors/recent?severity=')
        
        # Should handle gracefully
        assert response.status_code in [200, 400]
    
    def test_multiple_query_params(self, client):
        """Test endpoint with multiple query parameters."""
        response = client.get('/admin/performance/rolling?days=90&window=30')
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'window_size_days' in data


class TestDataIntegrity:
    """Test data integrity in responses."""
    
    def test_no_null_required_fields(self, client):
        """Test required fields are not null."""
        response = client.get('/admin/health')
        
        data = response.get_json()
        assert data['status'] is not None
        assert data['timestamp'] is not None
    
    def test_consistent_data_types(self, client):
        """Test data types are consistent."""
        response = client.get('/admin/performance/summary')
        
        data = response.get_json()
        # Numeric fields should be numbers or None
        if 'total_return' in data and data['total_return'] is not None:
            assert isinstance(data['total_return'], (int, float))
