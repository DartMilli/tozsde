"""
Database Backup Manager for Raspberry Pi Deployment (Sprint 5)

Responsibility:
    - Create nightly backups of SQLite database
    - Manage backup retention (30 days)
    - Verify backup integrity
    - Monitor backup health and disk space
    - Provide backup restoration capability

Features:
    - Timestamped database backups
    - Backup verification (file integrity check)
    - Automatic cleanup of old backups
    - Backup statistics and monitoring
    - Safe backup process (no database locks)
    - Callable from cron jobs

Usage:
    from app.infrastructure.backup_manager import BackupManager

    manager = BackupManager()
    manager.backup_database()  # Nightly cron
    manager.cleanup_old_backups()  # Weekly cron
    stats = manager.get_backup_stats()  # For monitoring
"""

import os
import shutil
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from app.infrastructure import get_settings

from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)


class BackupManager:
    """Manage SQLite database backups for Raspberry Pi deployment."""

    # Retention policy
    BACKUP_RETENTION_DAYS = 30  # Keep backups for 30 days

    # Backup directory name
    BACKUP_DIR_NAME = "backups"

    def __init__(
        self,
        db_path: Optional[Path] = None,
        backup_dir: Optional[Path] = None,
        settings: Optional[object] = None,
    ):
        """
        Initialize backup manager.

        Args:
            db_path: Path to database file (defaults to settings.database_path)
            backup_dir: Path to backup directory (defaults to <project_root>/backups)
        """
        # Allow injecting a Settings object; fall back to legacy Config via helper
        self.settings = settings

        cfg = settings or get_settings()

        if db_path:
            self.db_path = Path(db_path)
        else:
            db_from_settings = None
            if settings is not None and hasattr(settings, "DB_PATH"):
                db_from_settings = getattr(settings, "DB_PATH")
            db_from_cfg = getattr(cfg, "DB_PATH", None) if cfg is not None else None
            db_val = db_from_settings or db_from_cfg
            self.db_path = Path(db_val) if db_val is not None else Path("")

        # Default backup directory is project root / backups
        if backup_dir:
            self.backup_dir = Path(backup_dir)
        else:
            data_dir = None
            if settings is not None and hasattr(settings, "DATA_DIR"):
                data_dir = getattr(settings, "DATA_DIR")
            elif cfg is not None:
                data_dir = getattr(cfg, "DATA_DIR", None)

            if data_dir:
                project_root = Path(data_dir).parent
            else:
                project_root = Path(".")

            self.backup_dir = project_root / self.BACKUP_DIR_NAME

        # Ensure backup directory exists
        try:
            os.makedirs(self.backup_dir, exist_ok=True)
        except Exception:
            # defensive: path may be a Path object convertible to str
            os.makedirs(str(self.backup_dir), exist_ok=True)

    def backup_database(self) -> Dict:
        """
        Create timestamped backup of SQLite database.

        Uses SQLite's backup API for safe, non-locking backup.
        This should be called nightly via cron.

        Returns:
            dict: {
                'success': bool,
                'backup_path': str,
                'size_mb': float,
                'timestamp': str,
                'error': str (if failed)
            }
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S%f")
        backup_filename = f"market_data_backup_{timestamp}.db"
        backup_path = self.backup_dir / backup_filename
        if backup_path.exists():
            suffix = 1
            while True:
                candidate = self.backup_dir / (
                    f"market_data_backup_{timestamp}_{suffix}.db"
                )
                if not candidate.exists():
                    backup_path = candidate
                    backup_filename = candidate.name
                    break
                suffix += 1

        logger.info(f"Starting database backup to {backup_path}...")

        try:
            # Check if source database exists
            if not self.db_path.exists():
                error_msg = f"Source database not found: {self.db_path}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "backup_path": None,
                    "size_mb": 0.0,
                    "timestamp": timestamp,
                    "error": error_msg,
                }

            # Python 3.6 compatible: use file copy instead of backup() API
            # Close any existing connections to ensure clean copy
            source_conn = None

            try:
                # Open source briefly to ensure it's accessible
                source_conn = sqlite3.connect(str(self.db_path))
                source_conn.close()
                source_conn = None

                # Copy database file
                shutil.copy2(self.db_path, backup_path)

                # Get backup size
                backup_size = backup_path.stat().st_size
                size_mb = backup_size / (1024 * 1024)

                logger.info(
                    f"Backup created successfully: {backup_filename} ({size_mb:.2f} MB)"
                )

                # Verify backup integrity
                verification = self.verify_backup(backup_path)

                if not verification["valid"]:
                    logger.warning(
                        f"Backup verification failed: {verification['error']}"
                    )
                    return {
                        "success": False,
                        "backup_path": str(backup_path),
                        "size_mb": size_mb,
                        "timestamp": timestamp,
                        "error": f"Backup verification failed: {verification['error']}",
                    }

                return {
                    "success": True,
                    "backup_path": str(backup_path),
                    "size_mb": round(size_mb, 2),
                    "timestamp": timestamp,
                    "tables": verification.get("tables", 0),
                }

            except Exception as e:
                # Cleanup partial backup
                if backup_path.exists():
                    backup_path.unlink()
                raise e

        except Exception as e:
            error_msg = f"Backup failed: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "backup_path": None,
                "size_mb": 0.0,
                "timestamp": timestamp,
                "error": error_msg,
            }

    def verify_backup(self, backup_path: Path) -> Dict:
        """
        Verify backup file integrity.

        Args:
            backup_path: Path to backup file to verify

        Returns:
            dict: {
                'valid': bool,
                'tables': int (number of tables),
                'error': str (if invalid)
            }
        """
        conn = None
        try:
            if not backup_path.exists():
                return {"valid": False, "tables": 0, "error": "Backup file not found"}

            # Try to open and query the backup
            conn = sqlite3.connect(str(backup_path))
            cursor = conn.cursor()

            # Check database integrity
            cursor.execute("PRAGMA integrity_check")
            integrity_result = cursor.fetchone()[0]

            if integrity_result != "ok":
                return {
                    "valid": False,
                    "tables": 0,
                    "error": f"Integrity check failed: {integrity_result}",
                }

            # Count tables
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]

            return {"valid": True, "tables": table_count, "error": None}

        except Exception as e:
            return {"valid": False, "tables": 0, "error": str(e)}
        finally:
            if conn is not None:
                conn.close()

    def cleanup_old_backups(self) -> Dict:
        """
        Delete backups older than BACKUP_RETENTION_DAYS.

        This should be called weekly via cron.

        Returns:
            dict: {
                'deleted_count': int,
                'deleted_files': list,
                'errors': list,
                'bytes_freed': int
            }
        """
        logger.info(
            f"Starting backup cleanup (retention: {self.BACKUP_RETENTION_DAYS} days)..."
        )

        cutoff_date = datetime.now(timezone.utc) - timedelta(
            days=self.BACKUP_RETENTION_DAYS
        )
        deleted_files = []
        errors = []
        bytes_freed = 0

        try:
            # Find all backup files
            backup_files = self._find_backup_files()

            for backup_file in backup_files:
                try:
                    # Check file age
                    mtime = datetime.fromtimestamp(
                        backup_file.stat().st_mtime, timezone.utc
                    )

                    if mtime < cutoff_date:
                        # Delete old backup
                        file_size = backup_file.stat().st_size
                        backup_file.unlink()

                        bytes_freed += file_size
                        deleted_files.append(str(backup_file.name))
                        logger.info(
                            f"Deleted old backup: {backup_file.name} "
                            f"(freed {file_size / (1024*1024):.2f} MB)"
                        )

                except Exception as e:
                    error_msg = f"Failed to delete {backup_file.name}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)

            logger.info(
                f"Backup cleanup complete: {len(deleted_files)} files deleted, "
                f"{bytes_freed / (1024*1024):.2f} MB freed"
            )

            return {
                "deleted_count": len(deleted_files),
                "deleted_files": deleted_files,
                "errors": errors,
                "bytes_freed": bytes_freed,
            }

        except Exception as e:
            error_msg = f"Backup cleanup failed: {str(e)}"
            logger.error(error_msg)
            return {
                "deleted_count": 0,
                "deleted_files": [],
                "errors": [error_msg],
                "bytes_freed": 0,
            }

    def get_backup_stats(self) -> Dict:
        """
        Get statistics about backup directory.

        Returns:
            dict: {
                'total_backups': int,
                'total_size_mb': float,
                'oldest_backup': str,
                'newest_backup': str,
                'oldest_age_days': float,
                'directory': str
            }
        """
        try:
            backup_files = self._find_backup_files()

            if not backup_files:
                return {
                    "total_backups": 0,
                    "total_size_mb": 0.0,
                    "oldest_backup": None,
                    "newest_backup": None,
                    "oldest_age_days": None,
                    "directory": str(self.backup_dir),
                }

            # Calculate total size
            total_size = sum(f.stat().st_size for f in backup_files)

            # Find oldest and newest
            backup_files_with_mtime = [(f, f.stat().st_mtime) for f in backup_files]
            backup_files_with_mtime.sort(key=lambda x: x[1])

            oldest_file, oldest_mtime = backup_files_with_mtime[0]
            newest_file, newest_mtime = backup_files_with_mtime[-1]

            oldest_datetime = datetime.fromtimestamp(oldest_mtime, timezone.utc)
            oldest_age_days = (
                datetime.now(timezone.utc) - oldest_datetime
            ).total_seconds() / 86400

            return {
                "total_backups": len(backup_files),
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "oldest_backup": oldest_file.name,
                "newest_backup": newest_file.name,
                "oldest_age_days": round(oldest_age_days, 1),
                "directory": str(self.backup_dir),
            }

        except Exception as e:
            logger.error(f"Failed to get backup stats: {e}")
            return {
                "total_backups": 0,
                "total_size_mb": 0.0,
                "oldest_backup": None,
                "newest_backup": None,
                "oldest_age_days": None,
                "directory": str(self.backup_dir),
                "error": str(e),
            }

    def restore_backup(
        self, backup_path: Path, target_path: Optional[Path] = None
    ) -> Dict:
        """
        Restore database from backup.

        Args:
            backup_path: Path to backup file to restore
            target_path: Target path for restored database (defaults to self.db_path)

        Returns:
            dict: {
                'success': bool,
                'restored_path': str,
                'error': str (if failed)
            }
        """
        target_path = target_path or self.db_path

        logger.info(f"Restoring database from {backup_path} to {target_path}...")

        try:
            # Verify backup before restoring
            verification = self.verify_backup(backup_path)

            if not verification["valid"]:
                error_msg = f"Cannot restore: backup verification failed: {verification['error']}"
                logger.error(error_msg)
                return {"success": False, "restored_path": None, "error": error_msg}

            # Create backup of current database if it exists
            if target_path.exists():
                backup_current = target_path.with_suffix(".db.before_restore")
                shutil.copy2(target_path, backup_current)
                logger.info(f"Created backup of current database: {backup_current}")

            # Restore backup
            shutil.copy2(backup_path, target_path)

            logger.info(f"v Database restored successfully from {backup_path.name}")

            return {"success": True, "restored_path": str(target_path), "error": None}

        except Exception as e:
            error_msg = f"Restore failed: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "restored_path": None, "error": error_msg}

    def list_backups(self) -> List[Dict]:
        """
        List all available backups with metadata.

        Returns:
            List of dicts with backup information
        """
        backups = []

        try:
            backup_files = self._find_backup_files()

            for backup_file in backup_files:
                mtime = datetime.fromtimestamp(
                    backup_file.stat().st_mtime, timezone.utc
                )
                size_mb = backup_file.stat().st_size / (1024 * 1024)
                age_days = (datetime.now(timezone.utc) - mtime).total_seconds() / 86400

                backups.append(
                    {
                        "filename": backup_file.name,
                        "path": str(backup_file),
                        "size_mb": round(size_mb, 2),
                        "created": mtime.isoformat(),
                        "age_days": round(age_days, 1),
                    }
                )

            # Sort by creation time (newest first)
            backups.sort(key=lambda x: x["created"], reverse=True)

        except Exception as e:
            logger.error(f"Failed to list backups: {e}")

        return backups

    def _find_backup_files(self) -> List[Path]:
        """
        Find all backup files in backup directory.

        Returns:
            List of Path objects for backup files
        """
        if not self.backup_dir.exists():
            return []

        # Match pattern: market_data_backup_YYYYMMDD_HHMMSS.db
        return list(self.backup_dir.glob("market_data_backup_*.db"))


def main():
    """CLI entry point for backup management."""
    import argparse

    parser = argparse.ArgumentParser(description="Database backup manager")
    parser.add_argument("--backup", action="store_true", help="Create database backup")
    parser.add_argument("--cleanup", action="store_true", help="Cleanup old backups")
    parser.add_argument("--stats", action="store_true", help="Show backup statistics")
    parser.add_argument("--list", action="store_true", help="List all backups")
    parser.add_argument("--verify", type=str, help="Verify specific backup file")
    parser.add_argument("--restore", type=str, help="Restore from specific backup file")

    args = parser.parse_args()

    manager = BackupManager()

    if args.backup:
        result = manager.backup_database()
        if result["success"]:
            print(f"Backup created: {result['backup_path']} ({result['size_mb']} MB)")
        else:
            print(f"Backup failed: {result.get('error', 'Unknown error')}")
            exit(1)

    elif args.cleanup:
        result = manager.cleanup_old_backups()
        print(
            f"Deleted {result['deleted_count']} backups, "
            f"freed {result['bytes_freed'] / (1024*1024):.2f} MB"
        )
        if result["errors"]:
            print(f"Errors: {result['errors']}")

    elif args.stats:
        stats = manager.get_backup_stats()
        print(f"\nBackup Directory Statistics:")
        print(f"  Directory: {stats['directory']}")
        print(f"  Total backups: {stats['total_backups']}")
        print(f"  Total size: {stats['total_size_mb']} MB")
        print(
            f"  Oldest backup: {stats['oldest_backup']} ({stats['oldest_age_days']} days old)"
        )
        print(f"  Newest backup: {stats['newest_backup']}")

    elif args.list:
        backups = manager.list_backups()
        print(f"\nAvailable Backups ({len(backups)}):")
        for backup in backups:
            print(f"  {backup['filename']}")
            print(f"    Size: {backup['size_mb']} MB")
            print(f"    Created: {backup['created']}")
            print(f"    Age: {backup['age_days']} days")

    elif args.verify:
        backup_path = Path(args.verify)
        result = manager.verify_backup(backup_path)
        if result["valid"]:
            print(f"v Backup valid: {backup_path.name} ({result['tables']} tables)")
        else:
            print(f"x Backup invalid: {result['error']}")
            exit(1)

    elif args.restore:
        backup_path = Path(args.restore)
        result = manager.restore_backup(backup_path)
        if result["success"]:
            print(f"v Database restored from {backup_path.name}")
        else:
            print(f"x Restore failed: {result['error']}")
            exit(1)

    else:
        # Default: show stats
        stats = manager.get_backup_stats()
        print(f"Backups: {stats['total_backups']} files, {stats['total_size_mb']} MB")
        print("Use --help for more options")


if __name__ == "__main__":
    main()
