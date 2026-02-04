"""
Error Alerting System (P9 — Engineering Hardening)

Responsibility:
    - Detect critical system errors
    - Send immediate alerts to operators
    - Prevent silent failures in production
    - Integration with email notifications

Features:
    - Critical error classification
    - Alert level determination
    - Email and log notifications
    - Error recovery suggestions

Usage:
    try:
        run_daily()
    except Exception as e:
        ErrorAlerter.alert(
            "CRON_EXECUTION_FAILED",
            str(e),
            details={"ticker": "VOO", "traceback": traceback.format_exc()}
        )
        raise
"""

from typing import Dict, Optional
from datetime import datetime, timezone
import traceback
import json

from app.config.config import Config
from app.infrastructure.logger import setup_logger
from app.notifications.mailer import send_email

logger = setup_logger(__name__)


class ErrorAlerter:
    """
    System-level error detection and alerting.

    Error Severity Levels:
        - CRITICAL: System-level failures (DB, network, crash)
        - WARNING: Operational issues (missing data, retry needed)
        - INFO: Normal events requiring logging only
    """

    # Critical errors that always trigger email alerts
    CRITICAL_ERRORS = {
        "DB_CONNECTION_FAILED": "Database connection error - check DB availability",
        "DB_WRITE_FAILED": "Database write error - check disk space and permissions",
        "CRON_EXECUTION_FAILED": "Daily pipeline failed unexpectedly",
        "DATA_DOWNLOAD_ERROR": "Failed to download market data from source",
        "PIPELINE_TIMEOUT": "Pipeline execution exceeded timeout threshold",
        "AUTHENTICATION_FAILED": "Email or API authentication failed",
        "INSUFFICIENT_DATA": "Insufficient historical data for analysis",
    }

    # Warning errors that log but don't always email
    WARNING_ERRORS = {
        "SLOW_EXECUTION": "Pipeline execution slower than threshold",
        "MISSING_TICKER_DATA": "No data available for some tickers",
        "LOW_CONFIDENCE": "All recommendations below confidence threshold",
        "HIGH_DRIFT": "Model performance drift detected",
    }

    @staticmethod
    def alert(
        error_code: str, message: str, details: Dict = None, severity: str = "auto"
    ) -> bool:
        """
        Log and potentially alert on error.

        Args:
            error_code: Error classification code
            message: Human-readable error message
            details: Additional context (ticker, traceback, etc.)
            severity: "auto" (detect) | "critical" | "warning" | "info"

        Returns:
            True if alert was sent, False otherwise
        """
        # TODO: Implement

        details = details or {}

        # Auto-detect severity
        if severity == "auto":
            if error_code in ErrorAlerter.CRITICAL_ERRORS:
                severity = "critical"
            elif error_code in ErrorAlerter.WARNING_ERRORS:
                severity = "warning"
            else:
                severity = "info"

        # Log appropriately
        log_entry = f"[{severity.upper()}] {error_code}: {message}"

        if severity == "critical":
            logger.critical(log_entry)
            should_alert = True
        elif severity == "warning":
            logger.warning(log_entry)
            should_alert = ErrorAlerter._should_alert_on_warning(error_code)
        else:
            logger.info(log_entry)
            should_alert = False

        # Send alert if needed
        if should_alert:
            ErrorAlerter._send_alert_email(error_code, message, details, severity)
            return True

        return False

    @staticmethod
    def _should_alert_on_warning(error_code: str) -> bool:
        """
        Determine if warning should trigger email alert.

        Some warnings are frequent/expected and shouldn't spam email.
        """
        # TODO: Implement
        # Check frequency of similar errors in recent logs
        # Only alert if error is unusual

        high_priority_warnings = {
            "HIGH_DRIFT": True,
            "AUTHENTICATION_FAILED": True,
        }

        return high_priority_warnings.get(error_code, False)

    @staticmethod
    def _send_alert_email(error_code: str, message: str, details: Dict, severity: str):
        """
        Send alert email to operations team.

        Args:
            error_code: Error classification
            message: Error description
            details: Additional context
            severity: Alert severity level
        """
        # TODO: Implement

        description = ErrorAlerter.CRITICAL_ERRORS.get(
            error_code, ErrorAlerter.WARNING_ERRORS.get(error_code, "Unknown error")
        )

        subject = f"🚨 [{severity.upper()}] {error_code}"

        body = f"""
Tozsde Alert
============

Error Code: {error_code}
Severity: {severity.upper()}
Timestamp: {datetime.now(timezone.utc).isoformat()}

Description:
{description}

Message:
{message}

Details:
{json.dumps(details, indent=2)}

Recommended Action:
{ErrorAlerter._get_remediation_steps(error_code)}

---
Alert generated by Tozsde monitoring system
"""

        try:
            send_email(subject, body, Config.NOTIFY_EMAIL)
            logger.info(f"Alert email sent for {error_code}")
        except Exception as e:
            logger.error(f"Failed to send alert email: {e}")

    @staticmethod
    def _get_remediation_steps(error_code: str) -> str:
        """
        Get suggested remediation steps for error code.

        Args:
            error_code: Error classification

        Returns:
            Remediation suggestions
        """
        # TODO: Implement

        remediation_map = {
            "DB_CONNECTION_FAILED": (
                "1. Check database is running\n"
                "2. Verify database credentials in .env\n"
                "3. Check network connectivity to DB host\n"
                "4. Restart application"
            ),
            "DATA_DOWNLOAD_ERROR": (
                "1. Verify internet connectivity\n"
                "2. Check yfinance API status\n"
                "3. Try downloading data manually\n"
                "4. Check firewall rules"
            ),
            "CRON_EXECUTION_FAILED": (
                "1. Check cron logs for details\n"
                "2. Verify all dependencies installed\n"
                "3. Run pipeline manually to identify issue\n"
                "4. Check system resources (memory, CPU)"
            ),
            "PIPELINE_TIMEOUT": (
                "1. Check system load\n"
                "2. Increase timeout threshold if needed\n"
                "3. Optimize slow operations\n"
                "4. Run pipeline with profiling"
            ),
        }

        return remediation_map.get(
            error_code, "1. Review error logs\n2. Contact system administrator"
        )


# Utility functions


def catch_and_alert(func):
    """
    Decorator to catch exceptions and send alerts automatically.

    Usage:
        @catch_and_alert
        def my_function():
            ...
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            ErrorAlerter.alert(
                error_code=f"{func.__name__.upper()}_FAILED",
                message=str(e),
                details={
                    "function": func.__name__,
                    "args": str(args)[:100],
                    "traceback": traceback.format_exc(),
                },
                severity="critical",
            )
            raise

    return wrapper


import json
