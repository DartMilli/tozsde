"""
Health Check Module for Raspberry Pi Deployment (Sprint 5)

Responsibility:
    - Monitor system health (API, database, resources)
    - Log health metrics in JSONL format
    - Generate alerts when thresholds exceeded
    - Provide health status for monitoring and auto-restart

Features:
    - Flask API health check (port 5000)
    - SQLite database connectivity test
    - System resource monitoring (CPU, memory, disk)
    - Cron job execution tracking
    - Alert generation for critical failures
    - JSONL logging for monitoring dashboard

Usage:
    from app.infrastructure.health_check import HealthChecker

    checker = HealthChecker()
    status = checker.check_all()

    if not status['healthy']:
        print(f"Issues: {status['issues']}")
"""

import json
import os
import psutil
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional
import urllib.request
import urllib.error

from app.infrastructure import get_settings
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)


class HealthChecker:
    """Monitor system health for Raspberry Pi deployment."""

    # Thresholds
    DISK_SPACE_THRESHOLD_PCT = 10  # Alert if < 10% free
    MEMORY_THRESHOLD_PCT = 85  # Alert if > 85% used
    CPU_THRESHOLD_PCT = 90  # Alert if > 90% used
    API_TIMEOUT_SEC = 5  # API response timeout
    DB_TIMEOUT_SEC = 3  # Database query timeout
    CRON_MAX_AGE_HOURS = 25  # Alert if daily cron didn't run in 25h

    def __init__(self, settings: Optional[object] = None):
        """Initialize health checker.

        Args:
            settings: optional Settings object produced by `build_settings()`.
        """
        self.settings = settings

        def _is_pathlike(value) -> bool:
            try:
                return isinstance(value, (str, Path)) or hasattr(value, "__fspath__")
            except Exception:
                return False

        cfg = settings if settings is not None else get_settings()

        self.admin_api_key = (
            getattr(cfg, "ADMIN_API_KEY", None)
            or getattr(cfg, "admin_api_key", None)
            or os.getenv("ADMIN_API_KEY")
        )

        db_path = getattr(cfg, "DB_PATH", None)
        self.db_path = str(db_path) if _is_pathlike(db_path) else ""

        log_dir = getattr(cfg, "LOG_DIR", None)
        log_dir_explicit = _is_pathlike(log_dir)
        if not log_dir_explicit:
            log_dir = "."

        try:
            self.log_dir = Path(log_dir)
        except Exception:
            self.log_dir = Path(str(log_dir))

        self.health_log = self.log_dir / "health_check.jsonl"
        self._log_dir_explicit = log_dir_explicit

        # Ensure log directory exists (defensive)
        try:
            os.makedirs(str(self.log_dir), exist_ok=True)
        except Exception:
            try:
                os.makedirs(self.log_dir, exist_ok=True)
            except Exception:
                pass

        # If the resolved log_dir does not exist (e.g. Config lookup missed),
        # attempt sensible fallbacks based on DB path or common locations.
        if not os.path.exists(str(self.log_dir)):
            try:
                chosen = None
                if self.db_path:
                    db_p = Path(self.db_path)
                    # Walk ancestors and look for a sibling 'logs' directory
                    for ancestor in db_p.parents:
                        candidate = ancestor / "logs"
                        if candidate.exists():
                            chosen = candidate
                            break

                    # If none found, default to creating a 'logs' folder two levels up
                    if chosen is None:
                        chosen = (
                            db_p.parents[1] / "logs"
                            if len(db_p.parents) > 1
                            else db_p.parent / "logs"
                        )

                elif Path("logs").exists():
                    chosen = Path("logs")

                if chosen is not None:
                    self.log_dir = Path(chosen)
                    os.makedirs(str(self.log_dir), exist_ok=True)
                    self.health_log = self.log_dir / "health_check.jsonl"
            except Exception:
                pass

    def check_all(self) -> Dict:
        """
        Run all health checks and return comprehensive status.

        Returns:
            dict: {
                'timestamp': ISO timestamp,
                'healthy': bool,
                'issues': list of issue descriptions,
                'checks': {
                    'api': dict,
                    'database': dict,
                    'disk': dict,
                    'memory': dict,
                    'cpu': dict,
                    'cron': dict
                }
            }
        """
        logger.info("Starting comprehensive health check...")

        checks = {
            "api": self.check_api_health(),
            "database": self.check_database_health(),
            "disk": self.check_disk_space(),
            "memory": self.check_memory_usage(),
            "cpu": self.check_cpu_usage(),
            "cron": self.check_cron_execution(),
        }

        # Determine overall health
        issues = []
        for check_name, check_result in checks.items():
            if not check_result.get("healthy", False):
                issue = f"{check_name}: {check_result.get('message', 'Unknown issue')}"
                issues.append(issue)

        healthy = len(issues) == 0

        status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "healthy": healthy,
            "issues": issues,
            "checks": checks,
        }

        # Log to JSONL
        self._log_health_status(status)

        if healthy:
            logger.info("v All health checks passed")
        else:
            logger.warning(f"x Health issues detected: {issues}")

        return status

    def check_api_health(self) -> Dict:
        """
        Check if Flask API is responding on port 5000.

        Returns:
            dict: {'healthy': bool, 'status_code': int, 'response_time': float, 'message': str}
        """
        api_url = "http://localhost:5000/admin/health"

        try:
            start = time.time()
            headers = {}
            if self.admin_api_key:
                headers["X-Admin-Key"] = str(self.admin_api_key)

            req = urllib.request.Request(api_url, headers=headers, method="GET")

            with urllib.request.urlopen(req, timeout=self.API_TIMEOUT_SEC) as response:
                status_code = response.getcode()
                response_time = time.time() - start

                if status_code == 200:
                    return {
                        "healthy": True,
                        "status_code": status_code,
                        "response_time": round(response_time, 3),
                        "message": "API responding normally",
                    }
                else:
                    return {
                        "healthy": False,
                        "status_code": status_code,
                        "response_time": round(response_time, 3),
                        "message": f"API returned non-200 status: {status_code}",
                    }

        except urllib.error.URLError as e:
            return {
                "healthy": False,
                "status_code": None,
                "response_time": None,
                "message": f"API not reachable: {str(e)}",
            }
        except Exception as e:
            return {
                "healthy": False,
                "status_code": None,
                "response_time": None,
                "message": f"API check failed: {str(e)}",
            }

    def check_database_health(self) -> Dict:
        """
        Test SQLite database connectivity and basic operations.

        Returns:
            dict: {'healthy': bool, 'writable': bool, 'tables': int, 'message': str}
        """
        try:
            conn = sqlite3.connect(self.db_path, timeout=self.DB_TIMEOUT_SEC)
            cursor = conn.cursor()

            # Check if database is accessible
            cursor.execute("SELECT 1")
            result = cursor.fetchone()

            # Count tables
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]

            # Test write operation (no-op transaction)
            cursor.execute("BEGIN")
            cursor.execute("SELECT 1")
            cursor.execute("ROLLBACK")

            conn.close()

            return {
                "healthy": True,
                "writable": True,
                "tables": table_count,
                "message": f"Database healthy ({table_count} tables)",
            }

        except sqlite3.OperationalError as e:
            return {
                "healthy": False,
                "writable": False,
                "tables": 0,
                "message": f"Database locked or inaccessible: {str(e)}",
            }
        except Exception as e:
            return {
                "healthy": False,
                "writable": False,
                "tables": 0,
                "message": f"Database check failed: {str(e)}",
            }

    def check_disk_space(self) -> Dict:
        """
        Check disk space on root partition.

        Returns:
            dict: {'healthy': bool, 'free_pct': float, 'free_gb': float, 'message': str}
        """
        try:
            usage = psutil.disk_usage("/")
            free_pct = 100 - usage.percent
            free_gb = usage.free / (1024**3)  # Convert to GB

            healthy = free_pct >= self.DISK_SPACE_THRESHOLD_PCT

            return {
                "healthy": healthy,
                "free_pct": round(free_pct, 1),
                "free_gb": round(free_gb, 2),
                "total_gb": round(usage.total / (1024**3), 2),
                "message": (
                    f"{free_pct:.1f}% free ({free_gb:.2f} GB)"
                    if healthy
                    else f"Low disk space: {free_pct:.1f}% free"
                ),
            }

        except Exception as e:
            return {
                "healthy": False,
                "free_pct": None,
                "free_gb": None,
                "message": f"Disk check failed: {str(e)}",
            }

    def check_memory_usage(self) -> Dict:
        """
        Check system memory usage.

        Returns:
            dict: {'healthy': bool, 'used_pct': float, 'available_gb': float, 'message': str}
        """
        try:
            memory = psutil.virtual_memory()
            used_pct = memory.percent
            available_gb = memory.available / (1024**3)

            healthy = used_pct < self.MEMORY_THRESHOLD_PCT

            return {
                "healthy": healthy,
                "used_pct": round(used_pct, 1),
                "available_gb": round(available_gb, 2),
                "total_gb": round(memory.total / (1024**3), 2),
                "message": (
                    f"{used_pct:.1f}% used ({available_gb:.2f} GB available)"
                    if healthy
                    else f"High memory usage: {used_pct:.1f}%"
                ),
            }

        except Exception as e:
            return {
                "healthy": False,
                "used_pct": None,
                "available_gb": None,
                "message": f"Memory check failed: {str(e)}",
            }

    def check_cpu_usage(self) -> Dict:
        """
        Check CPU usage (1-second interval).

        Returns:
            dict: {'healthy': bool, 'used_pct': float, 'message': str}
        """
        try:
            # Sample CPU usage over 1 second
            cpu_pct = psutil.cpu_percent(interval=1.0)

            healthy = cpu_pct < self.CPU_THRESHOLD_PCT

            return {
                "healthy": healthy,
                "used_pct": round(cpu_pct, 1),
                "message": (
                    f"{cpu_pct:.1f}% CPU usage"
                    if healthy
                    else f"High CPU usage: {cpu_pct:.1f}%"
                ),
            }

        except Exception as e:
            return {
                "healthy": False,
                "used_pct": None,
                "message": f"CPU check failed: {str(e)}",
            }

    def check_cron_execution(self) -> Dict:
        """
        Check if daily cron job ran recently (via metrics log).

        Looks for recent pipeline execution in logs/metrics.jsonl.

        Returns:
            dict: {'healthy': bool, 'last_run': str, 'hours_ago': float, 'message': str}
        """
        metrics_log = self.log_dir / "metrics.jsonl"

        # If metrics log not present at resolved location, try to locate it
        # only when log_dir was not explicitly provided.
        if (
            not os.path.exists(str(metrics_log))
            and self.db_path
            and not getattr(self, "_log_dir_explicit", False)
        ):
            try:
                db_p = Path(self.db_path)
                for ancestor in db_p.parents:
                    for child in ancestor.iterdir():
                        if child.is_dir() and (child / "metrics.jsonl").exists():
                            self.log_dir = child
                            self.health_log = self.log_dir / "health_check.jsonl"
                            metrics_log = child / "metrics.jsonl"
                            break
                    if os.path.exists(str(metrics_log)):
                        break
            except Exception:
                pass

        try:
            if not os.path.exists(str(metrics_log)):
                return {
                    "healthy": False,
                    "last_run": None,
                    "hours_ago": None,
                    "message": "No metrics log found (pipeline never ran)",
                }

            # Read last 100 lines (reverse order)
            last_pipeline_time = None

            with open(str(metrics_log), "r") as f:
                lines = f.readlines()

                # Search backwards for most recent pipeline execution
                for line in reversed(lines[-100:]):
                    try:
                        log_entry = json.loads(line.strip())
                        if log_entry.get("event") == "pipeline_execution":
                            last_pipeline_time = log_entry.get("timestamp")
                            break
                    except json.JSONDecodeError:
                        continue

            if not last_pipeline_time:
                return {
                    "healthy": False,
                    "last_run": None,
                    "hours_ago": None,
                    "message": "No pipeline execution found in metrics",
                }

            # Calculate hours since last run
            try:
                last_run = datetime.fromisoformat(last_pipeline_time)
            except ValueError:
                try:
                    last_run = datetime.strptime(
                        last_pipeline_time, "%Y-%m-%dT%H:%M:%S.%f"
                    )
                except ValueError:
                    last_run = datetime.strptime(
                        last_pipeline_time, "%Y-%m-%dT%H:%M:%S"
                    )
            if last_run.tzinfo is None:
                last_run = last_run.replace(tzinfo=timezone.utc)
            hours_ago = (datetime.now(timezone.utc) - last_run).total_seconds() / 3600

            healthy = hours_ago < self.CRON_MAX_AGE_HOURS

            return {
                "healthy": healthy,
                "last_run": last_pipeline_time,
                "hours_ago": round(hours_ago, 1),
                "message": (
                    f"Last pipeline: {hours_ago:.1f}h ago"
                    if healthy
                    else f"Pipeline overdue: {hours_ago:.1f}h since last run"
                ),
            }

        except Exception as e:
            return {
                "healthy": False,
                "last_run": None,
                "hours_ago": None,
                "message": f"Cron check failed: {str(e)}",
            }

    def _log_health_status(self, status: Dict) -> None:
        """
        Log health status to JSONL file.

        Args:
            status: Health status dictionary from check_all()
        """
        try:
            # Ensure parent dir exists and use str path for compatibility
            try:
                os.makedirs(str(self.log_dir), exist_ok=True)
            except Exception:
                pass

            # If the target health log path is not under an existing dir, try to locate
            if not os.path.exists(str(self.log_dir)) and self.db_path:
                try:
                    db_p = Path(self.db_path)
                    for ancestor in db_p.parents:
                        for child in ancestor.iterdir():
                            if (
                                child.is_dir()
                                and (child / "health_check.jsonl").exists()
                            ):
                                self.log_dir = child
                                self.health_log = self.log_dir / "health_check.jsonl"
                                break
                        if os.path.exists(str(self.health_log)):
                            break
                except Exception:
                    pass

            with open(str(self.health_log), "a") as f:
                json_line = json.dumps(status)
                f.write(json_line + "\n")
                f.flush()
            # Also attempt to write to a discovered 'logs' sibling near DB (best-effort)
            try:
                if self.db_path:
                    db_p = Path(self.db_path)
                    candidate = (
                        db_p.parents[1] / "logs"
                        if len(db_p.parents) > 1
                        else db_p.parent / "logs"
                    )
                    candidate_file = candidate / "health_check.jsonl"
                    if candidate.exists():
                        try:
                            os.makedirs(str(candidate), exist_ok=True)
                            with open(str(candidate_file), "a") as cf:
                                cf.write(json_line + "\n")
                        except Exception:
                            pass
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Failed to write health log: {e}")

    def get_recent_health_logs(self, hours: int = 24) -> List[Dict]:
        """
        Retrieve recent health check logs.

        Args:
            hours: Number of hours to look back

        Returns:
            List of health status dictionaries
        """
        if not os.path.exists(str(self.health_log)):
            # Try to discover health log near DB if not found
            if self.db_path:
                try:
                    db_p = Path(self.db_path)
                    for ancestor in db_p.parents:
                        candidate = ancestor / "logs" / "health_check.jsonl"
                        if candidate.exists():
                            self.health_log = candidate
                            self.log_dir = candidate.parent
                            break
                except Exception:
                    pass

            if not os.path.exists(str(self.health_log)):
                return []

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_logs = []

        try:
            with open(str(self.health_log), "r") as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        timestamp_str = log_entry.get("timestamp", "")

                        try:
                            log_time = datetime.fromisoformat(timestamp_str)
                        except ValueError:
                            try:
                                log_time = datetime.strptime(
                                    timestamp_str, "%Y-%m-%dT%H:%M:%S.%f"
                                )
                            except ValueError:
                                log_time = datetime.strptime(
                                    timestamp_str, "%Y-%m-%dT%H:%M:%S"
                                )
                        if log_time.tzinfo is None:
                            log_time = log_time.replace(tzinfo=timezone.utc)

                        if log_time >= cutoff:
                            recent_logs.append(log_entry)
                    except (json.JSONDecodeError, ValueError):
                        continue
        except Exception as e:
            logger.error(f"Failed to read health logs: {e}")

        return recent_logs

    def generate_alert(self, status: Dict) -> Optional[str]:
        """
        Generate alert message if critical issues detected.

        Args:
            status: Health status from check_all()

        Returns:
            Alert message string if issues found, None otherwise
        """
        if status["healthy"]:
            return None

        issues = status["issues"]
        alert = f" HEALTH CHECK ALERT [{datetime.now(timezone.utc).isoformat()}]\n\n"
        alert += f"Detected {len(issues)} issue(s):\n"

        for i, issue in enumerate(issues, 1):
            alert += f"  {i}. {issue}\n"

        alert += "\nAction: Check system logs and restart services if needed.\n"

        return alert


def main():
    """CLI entry point for health check script."""
    checker = HealthChecker()
    status = checker.check_all()

    # Print summary
    print(f"\n{'='*60}")
    print(f"HEALTH CHECK SUMMARY - {status['timestamp']}")
    print(f"{'='*60}")
    print(f"Overall Status: {'v HEALTHY' if status['healthy'] else 'x UNHEALTHY'}")
    print(f"{'='*60}\n")

    # Print individual checks
    for check_name, check_result in status["checks"].items():
        healthy = check_result.get("healthy", False)
        message = check_result.get("message", "N/A")
        status_icon = "v" if healthy else "x"

        print(f"{status_icon} {check_name.upper()}: {message}")

    # Generate alert if needed
    if not status["healthy"]:
        alert = checker.generate_alert(status)
        print(f"\n{alert}")
        return 1  # Exit code 1 for unhealthy

    return 0  # Exit code 0 for healthy


if __name__ == "__main__":
    exit(main())
