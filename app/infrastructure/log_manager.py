"""
Log Rotation & Cleanup Manager for Raspberry Pi Deployment (Sprint 5)

Responsibility:
    - Rotate and archive old log files
    - Cleanup old archives to prevent disk space issues
    - Monitor log directory size
    - Ensure Pi doesn't run out of disk space

Features:
    - Automatic log rotation after 7 days (compress to .gz)
    - Automatic archive deletion after 30 days
    - Log directory size monitoring
    - Graceful handling of missing files/directories
    - Callable from cron jobs

Usage:
    from app.infrastructure.log_manager import LogManager

    manager = LogManager()
    manager.rotate_logs()  # Daily cron
    manager.cleanup_old_archives()  # Weekly cron
    stats = manager.get_log_stats()  # For monitoring
"""

import gzip
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from app.config.config import Config
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)


class LogManager:
    """Manage log rotation, archiving, and cleanup for Raspberry Pi."""

    # Retention policies
    ROTATION_AGE_DAYS = 7  # Archive logs older than 7 days
    ARCHIVE_RETENTION_DAYS = 30  # Delete archives older than 30 days

    # Patterns to match log files
    LOG_EXTENSIONS = [".log", ".jsonl"]
    ARCHIVE_EXTENSION = ".gz"

    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize log manager.

        Args:
            log_dir: Custom log directory (defaults to Config.LOG_DIR)
        """
        self.log_dir = log_dir or Config.LOG_DIR

        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)

    def rotate_logs(self) -> Dict:
        """
        Rotate logs older than ROTATION_AGE_DAYS by compressing to .gz.

        This should be called daily via cron.

        Returns:
            dict: {
                'rotated_count': int,
                'rotated_files': list,
                'errors': list,
                'bytes_saved': int
            }
        """
        logger.info(
            f"Starting log rotation (age threshold: {self.ROTATION_AGE_DAYS} days)..."
        )

        cutoff_date = datetime.now(timezone.utc) - timedelta(
            days=self.ROTATION_AGE_DAYS
        )
        rotated_files = []
        errors = []
        bytes_saved = 0

        try:
            # Find all log files
            log_files = self._find_log_files()

            for log_file in log_files:
                try:
                    # Check file age
                    mtime = datetime.fromtimestamp(
                        log_file.stat().st_mtime, timezone.utc
                    )

                    if mtime < cutoff_date:
                        # Compress and archive
                        original_size = log_file.stat().st_size
                        archive_path = self._compress_log(log_file)

                        if archive_path:
                            compressed_size = archive_path.stat().st_size
                            bytes_saved += original_size - compressed_size
                            rotated_files.append(str(log_file.name))
                            logger.info(
                                f"Rotated: {log_file.name} -> {archive_path.name} "
                                f"(saved {original_size - compressed_size} bytes)"
                            )

                except Exception as e:
                    error_msg = f"Failed to rotate {log_file.name}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)

            logger.info(
                f"Log rotation complete: {len(rotated_files)} files rotated, "
                f"{bytes_saved} bytes saved"
            )

            return {
                "rotated_count": len(rotated_files),
                "rotated_files": rotated_files,
                "errors": errors,
                "bytes_saved": bytes_saved,
            }

        except Exception as e:
            error_msg = f"Log rotation failed: {str(e)}"
            logger.error(error_msg)
            return {
                "rotated_count": 0,
                "rotated_files": [],
                "errors": [error_msg],
                "bytes_saved": 0,
            }

    def cleanup_old_archives(self) -> Dict:
        """
        Delete archived logs older than ARCHIVE_RETENTION_DAYS.

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
            f"Starting archive cleanup (retention: {self.ARCHIVE_RETENTION_DAYS} days)..."
        )

        cutoff_date = datetime.now(timezone.utc) - timedelta(
            days=self.ARCHIVE_RETENTION_DAYS
        )
        deleted_files = []
        errors = []
        bytes_freed = 0

        try:
            # Find all archived files
            archive_files = self._find_archive_files()

            for archive_file in archive_files:
                try:
                    # Check file age
                    mtime = datetime.fromtimestamp(
                        archive_file.stat().st_mtime, timezone.utc
                    )

                    if mtime < cutoff_date:
                        # Delete old archive
                        file_size = archive_file.stat().st_size
                        archive_file.unlink()

                        bytes_freed += file_size
                        deleted_files.append(str(archive_file.name))
                        logger.info(
                            f"Deleted old archive: {archive_file.name} "
                            f"(freed {file_size} bytes)"
                        )

                except Exception as e:
                    error_msg = f"Failed to delete {archive_file.name}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)

            logger.info(
                f"Archive cleanup complete: {len(deleted_files)} files deleted, "
                f"{bytes_freed} bytes freed"
            )

            return {
                "deleted_count": len(deleted_files),
                "deleted_files": deleted_files,
                "errors": errors,
                "bytes_freed": bytes_freed,
            }

        except Exception as e:
            error_msg = f"Archive cleanup failed: {str(e)}"
            logger.error(error_msg)
            return {
                "deleted_count": 0,
                "deleted_files": [],
                "errors": [error_msg],
                "bytes_freed": 0,
            }

    def get_log_stats(self) -> Dict:
        """
        Get statistics about log directory.

        Returns:
            dict: {
                'total_files': int,
                'log_files': int,
                'archive_files': int,
                'total_size_mb': float,
                'oldest_file': str,
                'oldest_age_days': float,
                'directory': str
            }
        """
        try:
            log_files = self._find_log_files()
            archive_files = self._find_archive_files()
            all_files = log_files + archive_files

            total_size = sum(f.stat().st_size for f in all_files)

            # Find oldest file
            oldest_file = None
            oldest_mtime = None

            for f in all_files:
                mtime = f.stat().st_mtime
                if oldest_mtime is None or mtime < oldest_mtime:
                    oldest_mtime = mtime
                    oldest_file = f

            oldest_age_days = None
            if oldest_file:
                oldest_datetime = datetime.fromtimestamp(oldest_mtime, timezone.utc)
                oldest_age_days = (
                    datetime.now(timezone.utc) - oldest_datetime
                ).total_seconds() / 86400

            return {
                "total_files": len(all_files),
                "log_files": len(log_files),
                "archive_files": len(archive_files),
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "oldest_file": oldest_file.name if oldest_file else None,
                "oldest_age_days": (
                    round(oldest_age_days, 1) if oldest_age_days else None
                ),
                "directory": str(self.log_dir),
            }

        except Exception as e:
            logger.error(f"Failed to get log stats: {e}")
            return {
                "total_files": 0,
                "log_files": 0,
                "archive_files": 0,
                "total_size_mb": 0.0,
                "oldest_file": None,
                "oldest_age_days": None,
                "directory": str(self.log_dir),
                "error": str(e),
            }

    def _find_log_files(self) -> List[Path]:
        """
        Find all uncompressed log files in log directory.

        Returns:
            List of Path objects for log files
        """
        log_files = []

        if not self.log_dir.exists():
            return log_files

        for ext in self.LOG_EXTENSIONS:
            log_files.extend(self.log_dir.glob(f"*{ext}"))

        return log_files

    def _find_archive_files(self) -> List[Path]:
        """
        Find all compressed archive files in log directory.

        Returns:
            List of Path objects for archive files
        """
        if not self.log_dir.exists():
            return []

        return list(self.log_dir.glob(f"*{self.ARCHIVE_EXTENSION}"))

    def _compress_log(self, log_file: Path) -> Optional[Path]:
        """
        Compress a log file to .gz and delete original.

        Args:
            log_file: Path to log file to compress

        Returns:
            Path to compressed archive, or None if failed
        """
        try:
            archive_path = log_file.with_suffix(
                log_file.suffix + self.ARCHIVE_EXTENSION
            )

            # Compress file
            with open(log_file, "rb") as f_in:
                with gzip.open(archive_path, "wb", compresslevel=9) as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Delete original
            log_file.unlink()

            return archive_path

        except Exception as e:
            logger.error(f"Failed to compress {log_file}: {e}")
            # Clean up partial archive if exists
            if archive_path.exists():
                archive_path.unlink()
            return None

    def force_rotate_all(self) -> Dict:
        """
        Force rotation of ALL log files regardless of age.

        Useful for testing or manual cleanup.

        Returns:
            dict: Same as rotate_logs()
        """
        logger.info("Force rotating all log files...")

        rotated_files = []
        errors = []
        bytes_saved = 0

        try:
            log_files = self._find_log_files()

            for log_file in log_files:
                try:
                    original_size = log_file.stat().st_size
                    archive_path = self._compress_log(log_file)

                    if archive_path:
                        compressed_size = archive_path.stat().st_size
                        bytes_saved += original_size - compressed_size
                        rotated_files.append(str(log_file.name))

                except Exception as e:
                    error_msg = f"Failed to rotate {log_file.name}: {str(e)}"
                    errors.append(error_msg)

            return {
                "rotated_count": len(rotated_files),
                "rotated_files": rotated_files,
                "errors": errors,
                "bytes_saved": bytes_saved,
            }

        except Exception as e:
            return {
                "rotated_count": 0,
                "rotated_files": [],
                "errors": [str(e)],
                "bytes_saved": 0,
            }

    def cleanup_all_archives(self) -> Dict:
        """
        Delete ALL archived logs regardless of age.

        Useful for testing or manual cleanup.

        Returns:
            dict: Same as cleanup_old_archives()
        """
        logger.info("Force deleting all archived logs...")

        deleted_files = []
        errors = []
        bytes_freed = 0

        try:
            archive_files = self._find_archive_files()

            for archive_file in archive_files:
                try:
                    file_size = archive_file.stat().st_size
                    archive_file.unlink()

                    bytes_freed += file_size
                    deleted_files.append(str(archive_file.name))

                except Exception as e:
                    error_msg = f"Failed to delete {archive_file.name}: {str(e)}"
                    errors.append(error_msg)

            return {
                "deleted_count": len(deleted_files),
                "deleted_files": deleted_files,
                "errors": errors,
                "bytes_freed": bytes_freed,
            }

        except Exception as e:
            return {
                "deleted_count": 0,
                "deleted_files": [],
                "errors": [str(e)],
                "bytes_freed": 0,
            }


def main():
    """CLI entry point for log management."""
    import argparse

    parser = argparse.ArgumentParser(description="Log rotation and cleanup manager")
    parser.add_argument("--rotate", action="store_true", help="Rotate old logs")
    parser.add_argument("--cleanup", action="store_true", help="Cleanup old archives")
    parser.add_argument("--stats", action="store_true", help="Show log statistics")
    parser.add_argument(
        "--force-rotate", action="store_true", help="Force rotate all logs"
    )
    parser.add_argument(
        "--force-cleanup", action="store_true", help="Force cleanup all archives"
    )

    args = parser.parse_args()

    manager = LogManager()

    if args.rotate:
        result = manager.rotate_logs()
        print(
            f"Rotated {result['rotated_count']} files, saved {result['bytes_saved']} bytes"
        )
        if result["errors"]:
            print(f"Errors: {result['errors']}")

    elif args.cleanup:
        result = manager.cleanup_old_archives()
        print(
            f"Deleted {result['deleted_count']} files, freed {result['bytes_freed']} bytes"
        )
        if result["errors"]:
            print(f"Errors: {result['errors']}")

    elif args.force_rotate:
        result = manager.force_rotate_all()
        print(f"Force rotated {result['rotated_count']} files")

    elif args.force_cleanup:
        result = manager.cleanup_all_archives()
        print(f"Force deleted {result['deleted_count']} archives")

    elif args.stats:
        stats = manager.get_log_stats()
        print(f"\nLog Directory Statistics:")
        print(f"  Directory: {stats['directory']}")
        print(f"  Total files: {stats['total_files']}")
        print(f"  Log files: {stats['log_files']}")
        print(f"  Archive files: {stats['archive_files']}")
        print(f"  Total size: {stats['total_size_mb']} MB")
        print(
            f"  Oldest file: {stats['oldest_file']} ({stats['oldest_age_days']} days old)"
        )

    else:
        # Default: show stats
        stats = manager.get_log_stats()
        print(f"Log stats: {stats['total_files']} files, {stats['total_size_mb']} MB")
        print("Use --help for more options")


if __name__ == "__main__":
    main()
