"""
Tests for Cron Task Scheduler.

Author: AI Assistant
Date: 2026-02-01
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from app.infrastructure.cron_tasks import CronTaskScheduler


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        
        log_dir = temp_path / "logs"
        backup_dir = temp_path / "backups"
        data_dir = temp_path / "data"
        
        log_dir.mkdir()
        backup_dir.mkdir()
        data_dir.mkdir()
        
        yield {
            'log_dir': log_dir,
            'backup_dir': backup_dir,
            'data_dir': data_dir
        }


@pytest.fixture
def mock_config(temp_dirs):
    """Mock Config object."""
    config = Mock()
    config.db_path = temp_dirs['data_dir'] / "market_data.db"
    config.log_dir = temp_dirs['log_dir']
    config.backup_dir = temp_dirs['backup_dir']
    return config


@pytest.fixture
def scheduler(mock_config):
    """Create scheduler with mocked config."""
    with patch('app.infrastructure.cron_tasks.Config', return_value=mock_config):
        scheduler = CronTaskScheduler()
        return scheduler


# --- Daily Pipeline Tests ---

def test_daily_pipeline_success(scheduler, mock_config):
    """Test successful daily pipeline execution."""
    # Mock data manager
    mock_data_manager = Mock()
    mock_data_manager.update_market_data.return_value = {
        'success': True,
        'total_records': 100
    }
    
    # Mock decision engine
    mock_decision_engine = Mock()
    mock_decision_engine.generate_recommendations.return_value = [
        {'symbol': 'AAPL', 'action': 'BUY'}
    ]
    
    # Mock backup and log managers
    scheduler.backup_manager.backup_database = Mock(return_value={
        'success': True,
        'size_mb': 10.5
    })
    scheduler.backup_manager.cleanup_old_backups = Mock(return_value={
        'deleted_count': 2
    })
    scheduler.log_manager.rotate_logs = Mock(return_value={
        'archived_count': 3
    })
    
    with patch('app.data_access.data_manager.DataManager', return_value=mock_data_manager), \
         patch('app.decision.decision_engine.DecisionEngine', return_value=mock_decision_engine):
        
        result = scheduler.run_daily_pipeline()
    
    assert result['success'] is True
    assert result['task'] == 'daily_pipeline'
    assert 'steps' in result
    assert result['steps']['data_update']['success'] is True
    assert result['steps']['recommendations']['count'] == 1
    assert result['steps']['backup']['success'] is True
    assert 'duration_seconds' in result


def test_daily_pipeline_partial_failure(scheduler):
    """Test daily pipeline with partial failures."""
    # Mock successful data update
    mock_data_manager = Mock()
    mock_data_manager.update_market_data.return_value = {
        'success': True,
        'total_records': 100
    }
    
    # Mock failed recommendations
    mock_decision_engine = Mock()
    mock_decision_engine.generate_recommendations.return_value = []
    
    # Mock backup and log managers
    scheduler.backup_manager.backup_database = Mock(return_value={
        'success': True,
        'size_mb': 10.5
    })
    scheduler.backup_manager.cleanup_old_backups = Mock(return_value={
        'deleted_count': 0
    })
    scheduler.log_manager.rotate_logs = Mock(return_value={
        'archived_count': 0
    })
    
    with patch('app.data_access.data_manager.DataManager', return_value=mock_data_manager), \
         patch('app.decision.decision_engine.DecisionEngine', return_value=mock_decision_engine):
        
        result = scheduler.run_daily_pipeline()
    
    # Should complete but not be fully successful
    assert result['success'] is False  # recommendations failed
    assert result['steps']['data_update']['success'] is True
    assert result['steps']['recommendations']['success'] is False


def test_daily_pipeline_exception(scheduler):
    """Test daily pipeline handles exceptions."""
    # Mock data manager that raises exception
    mock_data_manager = Mock()
    mock_data_manager.update_market_data.side_effect = Exception("Database error")
    
    with patch('app.data_access.data_manager.DataManager', return_value=mock_data_manager):
        result = scheduler.run_daily_pipeline()
    
    assert result['success'] is False
    assert 'error' in result
    assert 'Database error' in result['error']
    assert 'duration_seconds' in result


# --- Weekly Audit Tests ---

def test_weekly_audit_success(scheduler):
    """Test successful weekly audit."""
    class MockBacktestAuditor:
        def __init__(self, config):
            pass
        
        def run_weekly_audit(self):
            return {
                'success': True,
                'metrics': {
                    'total_trades': 50,
                    'win_rate': 0.65,
                    'sharpe_ratio': 1.8
                }
            }
    
    with patch('app.infrastructure.cron_tasks.BacktestAuditor', MockBacktestAuditor):
        result = scheduler.run_weekly_audit()
    
    assert result['success'] is True
    assert result['task'] == 'weekly_audit'
    assert 'metrics' in result
    assert result['metrics']['win_rate'] == 0.65
    assert 'duration_seconds' in result


def test_weekly_audit_failure(scheduler):
    """Test weekly audit handles failures."""
    class MockBacktestAuditor:
        def __init__(self, config):
            pass
        
        def run_weekly_audit(self):
            return {
                'success': False,
                'metrics': {}
            }
    
    with patch('app.infrastructure.cron_tasks.BacktestAuditor', MockBacktestAuditor):
        result = scheduler.run_weekly_audit()
    
    assert result['success'] is False
    assert result['task'] == 'weekly_audit'


def test_weekly_audit_exception(scheduler):
    """Test weekly audit handles exceptions."""
    class MockBacktestAuditor:
        def __init__(self, config):
            pass
        
        def run_weekly_audit(self):
            raise Exception("Audit error")
    
    with patch('app.infrastructure.cron_tasks.BacktestAuditor', MockBacktestAuditor):
        result = scheduler.run_weekly_audit()
    
    assert result['success'] is False
    assert 'error' in result
    assert 'Audit error' in result['error']


# --- Monthly Optimization Tests ---

def test_monthly_optimization_success(scheduler):
    """Test successful monthly optimization."""
    class MockGeneticOptimizer:
        def __init__(self, config):
            pass
        
        def run_monthly_optimization(self):
            return {
                'success': True,
                'best_fitness': 0.85,
                'generations': 100
            }
    
    with patch('app.infrastructure.cron_tasks.GeneticOptimizer', MockGeneticOptimizer):
        result = scheduler.run_monthly_optimization()
    
    assert result['success'] is True
    assert result['task'] == 'monthly_optimization'
    assert result['best_fitness'] == 0.85
    assert result['generations'] == 100
    assert 'duration_seconds' in result


def test_monthly_optimization_failure(scheduler):
    """Test monthly optimization handles failures."""
    class MockGeneticOptimizer:
        def __init__(self, config):
            pass
        
        def run_monthly_optimization(self):
            return {
                'success': False,
                'best_fitness': 0.0,
                'generations': 0
            }
    
    with patch('app.infrastructure.cron_tasks.GeneticOptimizer', MockGeneticOptimizer):
        result = scheduler.run_monthly_optimization()
    
    assert result['success'] is False
    assert result['task'] == 'monthly_optimization'


def test_monthly_optimization_exception(scheduler):
    """Test monthly optimization handles exceptions."""
    class MockGeneticOptimizer:
        def __init__(self, config):
            pass
        
        def run_monthly_optimization(self):
            raise Exception("Optimization error")
    
    with patch('app.infrastructure.cron_tasks.GeneticOptimizer', MockGeneticOptimizer):
        result = scheduler.run_monthly_optimization()
    
    assert result['success'] is False
    assert 'error' in result
    assert 'Optimization error' in result['error']


# --- Logging Tests ---

def test_log_cron_execution(scheduler, mock_config):
    """Test cron execution logging."""
    scheduler._log_cron_execution(
        task_name='test_task',
        status='success',
        duration=10.5,
        details={'test': 'data'}
    )
    
    log_file = Path(mock_config.log_dir) / 'cron_executions.jsonl'
    assert log_file.exists()
    
    with open(log_file, 'r') as f:
        log_entry = json.loads(f.readline())
    
    assert log_entry['task'] == 'test_task'
    assert log_entry['status'] == 'success'
    assert log_entry['duration_seconds'] == 10.5
    assert log_entry['details']['test'] == 'data'


def test_log_cron_execution_with_error(scheduler, mock_config):
    """Test logging with error message."""
    scheduler._log_cron_execution(
        task_name='failed_task',
        status='error',
        duration=5.2,
        error='Something went wrong'
    )
    
    log_file = Path(mock_config.log_dir) / 'cron_executions.jsonl'
    
    with open(log_file, 'r') as f:
        log_entry = json.loads(f.readline())
    
    assert log_entry['task'] == 'failed_task'
    assert log_entry['status'] == 'error'
    assert log_entry['error'] == 'Something went wrong'


def test_log_cron_execution_handles_write_error(scheduler, mock_config):
    """Test logging handles file write errors gracefully."""
    # Make log directory read-only
    mock_config.log_dir = Path('/nonexistent/path')
    
    # Should not raise exception
    scheduler._log_cron_execution(
        task_name='test_task',
        status='success',
        duration=1.0
    )


# --- Integration Tests ---

def test_multiple_daily_executions(scheduler):
    """Test multiple daily pipeline executions."""
    mock_data_manager = Mock()
    mock_data_manager.update_market_data.return_value = {
        'success': True,
        'total_records': 100
    }
    
    mock_decision_engine = Mock()
    mock_decision_engine.generate_recommendations.return_value = [
        {'symbol': 'AAPL', 'action': 'BUY'}
    ]
    
    scheduler.backup_manager.backup_database = Mock(return_value={
        'success': True,
        'size_mb': 10.5
    })
    scheduler.backup_manager.cleanup_old_backups = Mock(return_value={
        'deleted_count': 0
    })
    scheduler.log_manager.rotate_logs = Mock(return_value={
        'archived_count': 0
    })
    
    with patch('app.data_access.data_manager.DataManager', return_value=mock_data_manager), \
         patch('app.decision.decision_engine.DecisionEngine', return_value=mock_decision_engine):
        
        # Run twice
        result1 = scheduler.run_daily_pipeline()
        result2 = scheduler.run_daily_pipeline()
    
    assert result1['success'] is True
    assert result2['success'] is True


def test_task_timing(scheduler):
    """Test that task execution records accurate timing."""
    class MockBacktestAuditor:
        def __init__(self, config):
            pass
        
        def run_weekly_audit(self):
            return {
                'success': True,
                'metrics': {}
            }
    
    with patch('app.infrastructure.cron_tasks.BacktestAuditor', MockBacktestAuditor):
        result = scheduler.run_weekly_audit()
    
    # Should have non-negative duration
    assert result['duration_seconds'] >= 0
    assert result['duration_seconds'] < 10  # Should be fast with mocks


# --- CLI Tests ---

def test_cli_help():
    """Test CLI help message."""
    from app.infrastructure.cron_tasks import main
    
    with pytest.raises(SystemExit) as exc_info:
        with patch('sys.argv', ['cron_tasks.py']):
            main()
    
    assert exc_info.value.code == 1
