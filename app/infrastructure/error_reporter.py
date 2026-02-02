"""
Error Reporter - Comprehensive error tracking and reporting system.

This module tracks application errors, aggregates error statistics, monitors
error rates, and provides alerting for critical errors. Includes error history
export and automated cleanup.

Classes:
    ErrorSeverity: Enum for error severity levels
    ErrorRecord: Dataclass for error information
    ErrorReporter: Main error tracking and reporting class
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
import sqlite3
import traceback
import logging

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""
    timestamp: datetime
    severity: ErrorSeverity
    error_type: str
    error_message: str
    stack_trace: Optional[str] = None
    context: Optional[str] = None
    module: Optional[str] = None
    function: Optional[str] = None


@dataclass
class ErrorStatistics:
    """Aggregated error statistics."""
    total_errors: int
    errors_by_severity: Dict[str, int]
    errors_by_type: Dict[str, int]
    errors_by_module: Dict[str, int]
    error_rate_per_hour: float
    most_common_error: Optional[str]
    critical_errors: int
    period_start: datetime
    period_end: datetime


class ErrorReporter:
    """
    Comprehensive error tracking and reporting system.
    
    Features:
    - Error aggregation by severity, type, module
    - Error rate monitoring
    - Critical error alerting
    - Error history export
    - Automated cleanup
    """
    
    def __init__(
        self,
        db_path: str = None,
        critical_threshold: int = 5,
        error_rate_threshold: float = 10.0
    ):
        """
        Initialize error reporter.
        
        Args:
            db_path: Path to SQLite database for persistence
            critical_threshold: Number of critical errors before alerting
            error_rate_threshold: Error rate per hour threshold for alerting
        """
        self.db_path = db_path
        self.critical_threshold = critical_threshold
        self.error_rate_threshold = error_rate_threshold
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize SQLite database for error tracking."""
        if not self.db_path:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS error_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    severity TEXT NOT NULL,
                    error_type TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    stack_trace TEXT,
                    context TEXT,
                    module TEXT,
                    function TEXT,
                    indexed_date DATE NOT NULL
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_error_date 
                ON error_log(indexed_date)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_error_severity 
                ON error_log(severity)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_error_type 
                ON error_log(error_type)
            """)
            
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
    
    def log_error(
        self,
        error: Exception,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        context: Optional[str] = None,
        module: Optional[str] = None,
        function: Optional[str] = None
    ) -> bool:
        """
        Log an error to the database.
        
        Args:
            error: Exception object
            severity: Error severity level
            context: Additional context information
            module: Module where error occurred
            function: Function where error occurred
        
        Returns:
            bool: True if successfully logged
        """
        error_record = ErrorRecord(
            timestamp=datetime.now(),
            severity=severity,
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            context=context,
            module=module,
            function=function
        )
        
        return self._persist_error(error_record)
    
    def log_error_simple(
        self,
        error_type: str,
        error_message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        **kwargs
    ) -> bool:
        """
        Log an error with simplified interface.
        
        Args:
            error_type: Type/category of error
            error_message: Error message
            severity: Error severity level
            **kwargs: Additional fields (context, module, function, stack_trace)
        
        Returns:
            bool: True if successfully logged
        """
        error_record = ErrorRecord(
            timestamp=datetime.now(),
            severity=severity,
            error_type=error_type,
            error_message=error_message,
            stack_trace=kwargs.get('stack_trace'),
            context=kwargs.get('context'),
            module=kwargs.get('module'),
            function=kwargs.get('function')
        )
        
        return self._persist_error(error_record)
    
    def _persist_error(self, error_record: ErrorRecord) -> bool:
        """Persist error record to database."""
        if not self.db_path:
            logger.warning("No database path configured, error not persisted")
            return False
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            indexed_date = error_record.timestamp.date()
            
            cursor.execute("""
                INSERT INTO error_log
                (timestamp, severity, error_type, error_message, stack_trace,
                 context, module, function, indexed_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                error_record.timestamp,
                error_record.severity.value,
                error_record.error_type,
                error_record.error_message,
                error_record.stack_trace,
                error_record.context,
                error_record.module,
                error_record.function,
                indexed_date
            ))
            
            conn.commit()
            conn.close()
            
            # Check for critical error threshold
            if error_record.severity == ErrorSeverity.CRITICAL:
                self._check_critical_threshold()
            
            return True
        except sqlite3.Error as e:
            logger.error(f"Error persisting error record: {e}")
            return False
    
    def get_error_statistics(self, hours_back: int = 24) -> ErrorStatistics:
        """
        Get aggregated error statistics.
        
        Args:
            hours_back: Number of hours to analyze
        
        Returns:
            ErrorStatistics: Aggregated error statistics
        """
        if not self.db_path:
            return ErrorStatistics(
                total_errors=0,
                errors_by_severity={},
                errors_by_type={},
                errors_by_module={},
                error_rate_per_hour=0.0,
                most_common_error=None,
                critical_errors=0,
                period_start=datetime.now(),
                period_end=datetime.now()
            )
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            # Total errors
            cursor.execute("""
                SELECT COUNT(*) FROM error_log
                WHERE timestamp >= ?
            """, (cutoff_time,))
            total_errors = cursor.fetchone()[0]
            
            # By severity
            cursor.execute("""
                SELECT severity, COUNT(*) FROM error_log
                WHERE timestamp >= ?
                GROUP BY severity
            """, (cutoff_time,))
            errors_by_severity = dict(cursor.fetchall())
            
            # By type
            cursor.execute("""
                SELECT error_type, COUNT(*) FROM error_log
                WHERE timestamp >= ?
                GROUP BY error_type
                ORDER BY COUNT(*) DESC
            """, (cutoff_time,))
            errors_by_type = dict(cursor.fetchall())
            
            # By module
            cursor.execute("""
                SELECT module, COUNT(*) FROM error_log
                WHERE timestamp >= ? AND module IS NOT NULL
                GROUP BY module
            """, (cutoff_time,))
            errors_by_module = dict(cursor.fetchall())
            
            # Error rate
            error_rate = total_errors / hours_back if hours_back > 0 else 0.0
            
            # Most common error
            most_common = list(errors_by_type.keys())[0] if errors_by_type else None
            
            # Critical errors
            critical_errors = errors_by_severity.get(ErrorSeverity.CRITICAL.value, 0)
            
            conn.close()
            
            return ErrorStatistics(
                total_errors=total_errors,
                errors_by_severity=errors_by_severity,
                errors_by_type=errors_by_type,
                errors_by_module=errors_by_module,
                error_rate_per_hour=error_rate,
                most_common_error=most_common,
                critical_errors=critical_errors,
                period_start=cutoff_time,
                period_end=datetime.now()
            )
        except sqlite3.Error as e:
            logger.error(f"Error retrieving error statistics: {e}")
            return ErrorStatistics(
                total_errors=0,
                errors_by_severity={},
                errors_by_type={},
                errors_by_module={},
                error_rate_per_hour=0.0,
                most_common_error=None,
                critical_errors=0,
                period_start=datetime.now(),
                period_end=datetime.now()
            )
    
    def get_recent_errors(
        self,
        limit: int = 50,
        severity: Optional[ErrorSeverity] = None
    ) -> List[Dict]:
        """
        Get recent error records.
        
        Args:
            limit: Maximum number of records to return
            severity: Optional severity filter
        
        Returns:
            List of error records
        """
        if not self.db_path:
            return []
        
        records = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = """
                SELECT timestamp, severity, error_type, error_message, 
                       context, module, function
                FROM error_log
            """
            params = []
            
            if severity:
                query += " WHERE severity = ?"
                params.append(severity.value)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            
            for row in cursor.fetchall():
                records.append({
                    'timestamp': row[0],
                    'severity': row[1],
                    'error_type': row[2],
                    'error_message': row[3],
                    'context': row[4],
                    'module': row[5],
                    'function': row[6]
                })
            
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Error retrieving recent errors: {e}")
        
        return records
    
    def _check_critical_threshold(self) -> None:
        """Check if critical error threshold has been exceeded."""
        stats = self.get_error_statistics(hours_back=1)
        
        if stats.critical_errors >= self.critical_threshold:
            logger.critical(
                f"ALERT: Critical error threshold exceeded! "
                f"{stats.critical_errors} critical errors in last hour"
            )
            # Here you could trigger email/SMS/Slack alerts
    
    def check_error_rate(self) -> bool:
        """
        Check if error rate exceeds threshold.
        
        Returns:
            bool: True if error rate is within acceptable limits
        """
        stats = self.get_error_statistics(hours_back=1)
        
        if stats.error_rate_per_hour > self.error_rate_threshold:
            logger.warning(
                f"High error rate detected: {stats.error_rate_per_hour:.1f} errors/hour "
                f"(threshold: {self.error_rate_threshold})"
            )
            return False
        
        return True
    
    def export_error_report(
        self,
        output_file: str,
        days_back: int = 7
    ) -> bool:
        """
        Export error report to JSON file.
        
        Args:
            output_file: Path to output JSON file
            days_back: Number of days to include
        
        Returns:
            bool: True if successfully exported
        """
        if not self.db_path:
            return False
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            cursor.execute("""
                SELECT timestamp, severity, error_type, error_message, 
                       stack_trace, context, module, function
                FROM error_log
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            """, (cutoff_date,))
            
            errors = []
            for row in cursor.fetchall():
                errors.append({
                    'timestamp': row[0],
                    'severity': row[1],
                    'error_type': row[2],
                    'error_message': row[3],
                    'stack_trace': row[4],
                    'context': row[5],
                    'module': row[6],
                    'function': row[7]
                })
            
            conn.close()
            
            report = {
                'report_date': datetime.now().isoformat(),
                'period_days': days_back,
                'total_errors': len(errors),
                'errors': errors
            }
            
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Exported {len(errors)} errors to {output_file}")
            return True
        except (sqlite3.Error, IOError) as e:
            logger.error(f"Error exporting error report: {e}")
            return False
    
    def clear_old_errors(self, days_to_keep: int = 30) -> int:
        """
        Clear error records older than specified days.
        
        Args:
            days_to_keep: Number of days to retain
        
        Returns:
            int: Number of records deleted
        """
        if not self.db_path:
            return 0
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).date()
            
            cursor.execute("""
                DELETE FROM error_log
                WHERE indexed_date < ?
            """, (cutoff_date,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            logger.info(f"Deleted {deleted_count} error records older than {days_to_keep} days")
            return deleted_count
        except sqlite3.Error as e:
            logger.error(f"Error deleting old records: {e}")
            return 0
    
    def get_error_trends(self, days: int = 7) -> Dict[str, List]:
        """
        Get error trends over time.
        
        Args:
            days: Number of days to analyze
        
        Returns:
            Dict with daily error counts and trends
        """
        if not self.db_path:
            return {'dates': [], 'counts': [], 'severity_breakdown': {}}
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Daily counts
            cursor.execute("""
                SELECT indexed_date, COUNT(*) 
                FROM error_log
                WHERE timestamp >= ?
                GROUP BY indexed_date
                ORDER BY indexed_date
            """, (cutoff_date,))
            
            daily_data = cursor.fetchall()
            dates = [row[0] for row in daily_data]
            counts = [row[1] for row in daily_data]
            
            # Severity breakdown by day
            cursor.execute("""
                SELECT indexed_date, severity, COUNT(*)
                FROM error_log
                WHERE timestamp >= ?
                GROUP BY indexed_date, severity
                ORDER BY indexed_date
            """, (cutoff_date,))
            
            severity_data = cursor.fetchall()
            severity_breakdown = {}
            for date, severity, count in severity_data:
                if severity not in severity_breakdown:
                    severity_breakdown[severity] = {}
                severity_breakdown[severity][date] = count
            
            conn.close()
            
            return {
                'dates': dates,
                'counts': counts,
                'severity_breakdown': severity_breakdown
            }
        except sqlite3.Error as e:
            logger.error(f"Error retrieving error trends: {e}")
            return {'dates': [], 'counts': [], 'severity_breakdown': {}}
