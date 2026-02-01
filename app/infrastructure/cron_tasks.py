"""
Cron Task Scheduler - Wrapper functions for scheduled tasks on Raspberry Pi.

This module provides entry points for daily, weekly, and monthly scheduled tasks
that are called from cron jobs. Each function handles error logging and execution
tracking.

Author: AI Assistant
Date: 2026-02-01
"""

import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.infrastructure.logger import setup_logger
from app.infrastructure.backup_manager import BackupManager
from app.infrastructure.log_manager import LogManager
from app.config.config import Config

logger = setup_logger(__name__)

# Optional imports (kept at module level for easier testing/mocking)
try:
    from app.backtesting.backtest_audit import BacktestAuditor
except Exception:
    BacktestAuditor = None

try:
    from app.optimization.fitness import GeneticOptimizer
except Exception:
    GeneticOptimizer = None


class CronTaskScheduler:
    """
    Scheduler for automated cron tasks on Raspberry Pi.
    
    Provides wrapper functions for:
    - Daily market pipeline (6:00 AM)
    - Weekly backtest audit (Monday 4:00 AM)
    - Monthly GA optimization (1st of month 1:00 AM)
    """
    
    def __init__(self):
        """Initialize scheduler with config and managers."""
        self.config = Config()
        self.backup_manager = BackupManager()
        self.log_manager = LogManager()
    
    def run_daily_pipeline(self) -> Dict[str, Any]:
        """
        Run daily market data pipeline and analysis.
        
        Tasks:
        1. Fetch latest market data
        2. Update technical indicators
        3. Run decision engine
        4. Generate recommendations
        5. Backup database
        6. Rotate logs
        
        Returns:
            dict: Execution result with success status and metrics
        """
        task_name = "daily_pipeline"
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Starting {task_name} at {start_time.isoformat()}")
            
            # Import here to avoid circular dependencies
            from app.data_access.data_manager import DataManager
            from app.decision.decision_engine import DecisionEngine
            
            results = {
                'task': task_name,
                'start_time': start_time.isoformat(),
                'success': False,
                'steps': {}
            }
            
            # Step 1: Update market data
            logger.info("Step 1/5: Updating market data...")
            data_manager = DataManager(db_path=self.config.db_path)
            update_result = data_manager.update_market_data()
            results['steps']['data_update'] = {
                'success': update_result.get('success', False),
                'records': update_result.get('total_records', 0)
            }
            
            # Step 2: Generate recommendations
            logger.info("Step 2/5: Generating recommendations...")
            decision_engine = DecisionEngine(config=self.config)
            recommendations = decision_engine.generate_recommendations()
            results['steps']['recommendations'] = {
                'success': len(recommendations) > 0,
                'count': len(recommendations)
            }
            
            # Step 3: Backup database
            logger.info("Step 3/5: Creating database backup...")
            backup_result = self.backup_manager.backup_database()
            results['steps']['backup'] = {
                'success': backup_result.get('success', False),
                'size_mb': backup_result.get('size_mb', 0)
            }
            
            # Step 4: Rotate logs
            logger.info("Step 4/5: Rotating log files...")
            rotation_result = self.log_manager.rotate_logs()
            results['steps']['log_rotation'] = {
                'success': True,
                'archived': rotation_result.get('archived_count', 0)
            }
            
            # Step 5: Cleanup old backups
            logger.info("Step 5/5: Cleaning up old backups...")
            cleanup_result = self.backup_manager.cleanup_old_backups()
            results['steps']['backup_cleanup'] = {
                'success': True,
                'deleted': cleanup_result.get('deleted_count', 0)
            }
            
            # Calculate duration
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            results['success'] = all(
                step.get('success', False) 
                for step in results['steps'].values()
            )
            results['end_time'] = end_time.isoformat()
            results['duration_seconds'] = round(duration, 2)
            
            # Log execution
            self._log_cron_execution(
                task_name=task_name,
                status='success' if results['success'] else 'partial',
                duration=duration,
                details=results['steps']
            )
            
            logger.info(f"Daily pipeline completed in {duration:.2f}s - "
                       f"Status: {'SUCCESS' if results['success'] else 'PARTIAL'}")
            
            return results
            
        except Exception as e:
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            error_msg = f"Daily pipeline failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            self._log_cron_execution(
                task_name=task_name,
                status='error',
                duration=duration,
                error=str(e)
            )
            
            return {
                'task': task_name,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': round(duration, 2),
                'success': False,
                'error': str(e)
            }
    
    def run_weekly_audit(self) -> Dict[str, Any]:
        """
        Run weekly backtesting audit (Monday 4:00 AM).
        
        Tasks:
        1. Run backtesting on recent decisions
        2. Calculate performance metrics
        3. Generate audit report
        4. Log results
        
        Returns:
            dict: Execution result with audit metrics
        """
        task_name = "weekly_audit"
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Starting {task_name} at {start_time.isoformat()}")
            
            if BacktestAuditor is None:
                raise ImportError("BacktestAuditor is not available")
            
            results = {
                'task': task_name,
                'start_time': start_time.isoformat(),
                'success': False
            }
            
            # Run audit
            logger.info("Running backtesting audit...")
            auditor = BacktestAuditor(config=self.config)
            audit_result = auditor.run_weekly_audit()
            
            results['metrics'] = audit_result.get('metrics', {})
            results['success'] = audit_result.get('success', False)
            
            # Calculate duration
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            results['end_time'] = end_time.isoformat()
            results['duration_seconds'] = round(duration, 2)
            
            # Log execution
            self._log_cron_execution(
                task_name=task_name,
                status='success' if results['success'] else 'error',
                duration=duration,
                details=results.get('metrics', {})
            )
            
            logger.info(f"Weekly audit completed in {duration:.2f}s")
            
            return results
            
        except Exception as e:
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            error_msg = f"Weekly audit failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            self._log_cron_execution(
                task_name=task_name,
                status='error',
                duration=duration,
                error=str(e)
            )
            
            return {
                'task': task_name,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': round(duration, 2),
                'success': False,
                'error': str(e)
            }
    
    def run_monthly_optimization(self) -> Dict[str, Any]:
        """
        Run monthly genetic algorithm optimization (1st of month 1:00 AM).
        
        Tasks:
        1. Run GA optimization for strategy parameters
        2. Update best configurations
        3. Generate optimization report
        4. Archive old configurations
        
        Returns:
            dict: Execution result with optimization metrics
        """
        task_name = "monthly_optimization"
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Starting {task_name} at {start_time.isoformat()}")
            
            if GeneticOptimizer is None:
                raise ImportError("GeneticOptimizer is not available")
            
            results = {
                'task': task_name,
                'start_time': start_time.isoformat(),
                'success': False
            }
            
            # Run optimization
            logger.info("Running genetic algorithm optimization...")
            optimizer = GeneticOptimizer(config=self.config)
            opt_result = optimizer.run_monthly_optimization()
            
            results['best_fitness'] = opt_result.get('best_fitness', 0.0)
            results['generations'] = opt_result.get('generations', 0)
            results['success'] = opt_result.get('success', False)
            
            # Calculate duration
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            results['end_time'] = end_time.isoformat()
            results['duration_seconds'] = round(duration, 2)
            
            # Log execution
            self._log_cron_execution(
                task_name=task_name,
                status='success' if results['success'] else 'error',
                duration=duration,
                details={
                    'best_fitness': results.get('best_fitness', 0.0),
                    'generations': results.get('generations', 0)
                }
            )
            
            logger.info(f"Monthly optimization completed in {duration:.2f}s")
            
            return results
            
        except Exception as e:
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            error_msg = f"Monthly optimization failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            self._log_cron_execution(
                task_name=task_name,
                status='error',
                duration=duration,
                error=str(e)
            )
            
            return {
                'task': task_name,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': round(duration, 2),
                'success': False,
                'error': str(e)
            }
    
    def _log_cron_execution(
        self,
        task_name: str,
        status: str,
        duration: float,
        details: Dict = None,
        error: str = None
    ) -> None:
        """
        Log cron task execution to dedicated log file.
        
        Args:
            task_name: Name of the cron task
            status: Execution status (success/error/partial)
            duration: Execution time in seconds
            details: Optional execution details/metrics
            error: Optional error message
        """
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'task': task_name,
            'status': status,
            'duration_seconds': round(duration, 2),
            'details': details or {},
            'error': error
        }
        
        # Write to cron execution log
        cron_log_path = Path(self.config.log_dir) / 'cron_executions.jsonl'
        
        try:
            with open(cron_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to write cron execution log: {e}")


def main():
    """CLI entry point for cron task scheduler."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Cron task scheduler for trading system')
    parser.add_argument('--daily', action='store_true', help='Run daily pipeline')
    parser.add_argument('--weekly', action='store_true', help='Run weekly audit')
    parser.add_argument('--monthly', action='store_true', help='Run monthly optimization')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    
    args = parser.parse_args()
    
    scheduler = CronTaskScheduler()
    
    if args.daily:
        result = scheduler.run_daily_pipeline()
    elif args.weekly:
        result = scheduler.run_weekly_audit()
    elif args.monthly:
        result = scheduler.run_monthly_optimization()
    else:
        parser.print_help()
        sys.exit(1)
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        status = 'SUCCESS' if result.get('success', False) else 'FAILED'
        duration = result.get('duration_seconds', 0)
        print(f"{result['task']} - {status} ({duration:.2f}s)")
    
    sys.exit(0 if result.get('success', False) else 1)


if __name__ == '__main__':
    main()
