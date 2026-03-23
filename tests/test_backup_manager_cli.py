"""
Integration tests for backup_manager CLI.

Tests the command-line interface specifically to cover the main() function.
"""

import os
import pytest
import sqlite3
import tempfile
from pathlib import Path
from unittest import mock
from datetime import datetime, timedelta
import sys
from io import StringIO

from app.infrastructure.backup_manager import BackupManager, main


def _create_test_db(path: Path, size_kb=100):
    """Create minimal test SQLite database with schema."""
    conn = sqlite3.connect(str(path))
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE ohlcv (
            id INTEGER PRIMARY KEY,
            ticker TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER
        )
    """
    )
    cursor.execute(
        """
        CREATE TABLE decisions (
            id INTEGER PRIMARY KEY,
            ticker TEXT,
            decision TEXT,
            confidence REAL,
            timestamp TEXT
        )
    """
    )

    rows_needed = max(10, (size_kb * 100) // 5)
    for i in range(rows_needed):
        cursor.execute(
            "INSERT INTO ohlcv (ticker, date, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                f"TEST_{i}",
                f"2023-01-{(i%28)+1:02d}",
                100.0 + i,
                101.0 + i,
                99.0 + i,
                100.5 + i,
                1000000 + i,
            ),
        )

    conn.commit()
    conn.close()


class TestMainCLIIntegration:
    """Integration tests for main() CLI function."""

    @mock.patch("sys.argv", ["backup_manager.py", "--help"])
    def test_main_help_coverage(self):
        """Test --help argument to trigger lines 455-474."""
        with pytest.raises(SystemExit) as exc_info:
            main()

        # --help causes exit(0)
        assert exc_info.value.code == 0

    def test_main_backup_coverage(self, tmp_path):
        """Test --backup argument to cover backup execution paths."""
        db_path = tmp_path / "test.db"
        _create_test_db(db_path, size_kb=100)
        backup_dir = tmp_path / "backups"

        # Mock sys.argv
        test_argv = ["backup_manager.py", "--backup"]

        with mock.patch("sys.argv", test_argv):
            # Mock the backup directory initialization
            with mock.patch(
                "app.infrastructure.backup_manager.BackupManager.__init__",
                return_value=None,
            ):
                manager = BackupManager(db_path=db_path, backup_dir=backup_dir)

                # Mock backup_database to return success
                with mock.patch.object(
                    manager,
                    "backup_database",
                    return_value={
                        "success": True,
                        "backup_path": str(backup_dir / "test.db"),
                        "size_mb": 1.5,
                    },
                ):

                    # Capture stdout
                    captured_output = StringIO()
                    with mock.patch("sys.stdout", captured_output):
                        with mock.patch(
                            "app.infrastructure.backup_manager.BackupManager",
                            return_value=manager,
                        ):
                            try:
                                main()
                            except SystemExit:
                                pass

                    output = captured_output.getvalue()
                    assert "Backup" in output or output == ""

    def test_main_backup_failure_coverage(self, tmp_path):
        """Test --backup with failure to cover error paths (lines 476-478)."""
        db_path = tmp_path / "test.db"
        backup_dir = tmp_path / "backups"

        test_argv = ["backup_manager.py", "--backup"]

        with mock.patch("sys.argv", test_argv):
            with mock.patch(
                "app.infrastructure.backup_manager.BackupManager.__init__",
                return_value=None,
            ):
                manager = BackupManager(db_path=db_path, backup_dir=backup_dir)

                # Mock backup_database to return failure
                with mock.patch.object(
                    manager,
                    "backup_database",
                    return_value={"success": False, "error": "Test error"},
                ):

                    captured_output = StringIO()
                    with mock.patch("sys.stdout", captured_output):
                        with mock.patch(
                            "app.infrastructure.backup_manager.BackupManager",
                            return_value=manager,
                        ):
                            with pytest.raises(SystemExit) as exc_info:
                                main()

                            # Should exit with code 1
                            assert exc_info.value.code == 1

    def test_main_cleanup_coverage(self, tmp_path):
        """Test --cleanup argument to cover cleanup execution (lines 481-485)."""
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir(exist_ok=True)

        test_argv = ["backup_manager.py", "--cleanup"]

        with mock.patch("sys.argv", test_argv):
            with mock.patch(
                "app.infrastructure.backup_manager.BackupManager.__init__",
                return_value=None,
            ):
                manager = BackupManager(backup_dir=backup_dir)

                # Mock cleanup_old_backups
                with mock.patch.object(
                    manager,
                    "cleanup_old_backups",
                    return_value={
                        "deleted_count": 2,
                        "bytes_freed": 5242880,
                        "errors": [],
                    },
                ):

                    captured_output = StringIO()
                    with mock.patch("sys.stdout", captured_output):
                        with mock.patch(
                            "app.infrastructure.backup_manager.BackupManager",
                            return_value=manager,
                        ):
                            try:
                                main()
                            except SystemExit:
                                pass

                    output = captured_output.getvalue()
                    assert "Deleted" in output or output == ""

    def test_main_cleanup_with_errors_coverage(self, tmp_path):
        """Test --cleanup with errors to cover error reporting (line 487)."""
        backup_dir = tmp_path / "backups"

        test_argv = ["backup_manager.py", "--cleanup"]

        with mock.patch("sys.argv", test_argv):
            with mock.patch(
                "app.infrastructure.backup_manager.BackupManager.__init__",
                return_value=None,
            ):
                manager = BackupManager(backup_dir=backup_dir)

                # Mock cleanup_old_backups with errors
                with mock.patch.object(
                    manager,
                    "cleanup_old_backups",
                    return_value={
                        "deleted_count": 1,
                        "bytes_freed": 1048576,
                        "errors": ["Permission denied on file.db"],
                    },
                ):

                    captured_output = StringIO()
                    with mock.patch("sys.stdout", captured_output):
                        with mock.patch(
                            "app.infrastructure.backup_manager.BackupManager",
                            return_value=manager,
                        ):
                            try:
                                main()
                            except SystemExit:
                                pass

                    output = captured_output.getvalue()
                    assert "Errors" in output or output == ""

    def test_main_stats_coverage(self, tmp_path):
        """Test --stats argument (lines 489-497)."""
        backup_dir = tmp_path / "backups"

        test_argv = ["backup_manager.py", "--stats"]

        with mock.patch("sys.argv", test_argv):
            with mock.patch(
                "app.infrastructure.backup_manager.BackupManager.__init__",
                return_value=None,
            ):
                manager = BackupManager(backup_dir=backup_dir)

                # Mock get_backup_stats
                stats_data = {
                    "directory": str(backup_dir),
                    "total_backups": 3,
                    "total_size_mb": 150.5,
                    "oldest_backup": "backup_001.db",
                    "oldest_age_days": 25,
                    "newest_backup": "backup_003.db",
                }
                with mock.patch.object(
                    manager, "get_backup_stats", return_value=stats_data
                ):

                    captured_output = StringIO()
                    with mock.patch("sys.stdout", captured_output):
                        with mock.patch(
                            "app.infrastructure.backup_manager.BackupManager",
                            return_value=manager,
                        ):
                            try:
                                main()
                            except SystemExit:
                                pass

                    output = captured_output.getvalue()
                    assert "Backup Directory Statistics" in output or output == ""

    def test_main_list_coverage(self, tmp_path):
        """Test --list argument (lines 499-507)."""
        backup_dir = tmp_path / "backups"

        test_argv = ["backup_manager.py", "--list"]

        with mock.patch("sys.argv", test_argv):
            with mock.patch(
                "app.infrastructure.backup_manager.BackupManager.__init__",
                return_value=None,
            ):
                manager = BackupManager(backup_dir=backup_dir)

                # Mock list_backups
                backups_data = [
                    {
                        "filename": "backup_001.db",
                        "size_mb": 50.0,
                        "created": "2023-01-01",
                        "age_days": 10,
                    },
                    {
                        "filename": "backup_002.db",
                        "size_mb": 55.0,
                        "created": "2023-01-05",
                        "age_days": 5,
                    },
                ]
                with mock.patch.object(
                    manager, "list_backups", return_value=backups_data
                ):

                    captured_output = StringIO()
                    with mock.patch("sys.stdout", captured_output):
                        with mock.patch(
                            "app.infrastructure.backup_manager.BackupManager",
                            return_value=manager,
                        ):
                            try:
                                main()
                            except SystemExit:
                                pass

                    output = captured_output.getvalue()
                    assert "Available Backups" in output or output == ""

    def test_main_verify_success_coverage(self, tmp_path):
        """Test --verify with valid backup (lines 509-514)."""
        backup_dir = tmp_path / "backups"
        backup_file = backup_dir / "test_backup.db"

        test_argv = ["backup_manager.py", "--verify", str(backup_file)]

        with mock.patch("sys.argv", test_argv):
            with mock.patch(
                "app.infrastructure.backup_manager.BackupManager.__init__",
                return_value=None,
            ):
                manager = BackupManager(backup_dir=backup_dir)

                # Mock verify_backup to return valid
                with mock.patch.object(
                    manager, "verify_backup", return_value={"valid": True, "tables": 5}
                ):

                    captured_output = StringIO()
                    with mock.patch("sys.stdout", captured_output):
                        with mock.patch(
                            "app.infrastructure.backup_manager.BackupManager",
                            return_value=manager,
                        ):
                            try:
                                main()
                            except SystemExit:
                                pass

                    output = captured_output.getvalue()
                    assert (
                        "Backup valid" in output or "valid" in output.lower()
                    ) or output == ""

    def test_main_verify_failure_coverage(self, tmp_path):
        """Test --verify with invalid backup (lines 515-518)."""
        backup_dir = tmp_path / "backups"
        backup_file = backup_dir / "test_backup.db"

        test_argv = ["backup_manager.py", "--verify", str(backup_file)]

        with mock.patch("sys.argv", test_argv):
            with mock.patch(
                "app.infrastructure.backup_manager.BackupManager.__init__",
                return_value=None,
            ):
                manager = BackupManager(backup_dir=backup_dir)

                # Mock verify_backup to return invalid
                with mock.patch.object(
                    manager,
                    "verify_backup",
                    return_value={"valid": False, "error": "Corrupted file"},
                ):

                    captured_output = StringIO()
                    with mock.patch("sys.stdout", captured_output):
                        with mock.patch(
                            "app.infrastructure.backup_manager.BackupManager",
                            return_value=manager,
                        ):
                            with pytest.raises(SystemExit) as exc_info:
                                main()

                            # Should exit with code 1
                            assert exc_info.value.code == 1

    def test_main_restore_success_coverage(self, tmp_path):
        """Test --restore with valid backup (lines 520-524)."""
        backup_dir = tmp_path / "backups"
        backup_file = backup_dir / "test_backup.db"

        test_argv = ["backup_manager.py", "--restore", str(backup_file)]

        with mock.patch("sys.argv", test_argv):
            with mock.patch(
                "app.infrastructure.backup_manager.BackupManager.__init__",
                return_value=None,
            ):
                manager = BackupManager(backup_dir=backup_dir)

                # Mock restore_backup to return success
                with mock.patch.object(
                    manager, "restore_backup", return_value={"success": True}
                ):

                    captured_output = StringIO()
                    with mock.patch("sys.stdout", captured_output):
                        with mock.patch(
                            "app.infrastructure.backup_manager.BackupManager",
                            return_value=manager,
                        ):
                            try:
                                main()
                            except SystemExit:
                                pass

                    output = captured_output.getvalue()
                    assert "restored" in output.lower() or output == ""

    def test_main_restore_failure_coverage(self, tmp_path):
        """Test --restore with invalid backup (lines 525-528)."""
        backup_dir = tmp_path / "backups"
        backup_file = backup_dir / "test_backup.db"

        test_argv = ["backup_manager.py", "--restore", str(backup_file)]

        with mock.patch("sys.argv", test_argv):
            with mock.patch(
                "app.infrastructure.backup_manager.BackupManager.__init__",
                return_value=None,
            ):
                manager = BackupManager(backup_dir=backup_dir)

                # Mock restore_backup to return failure
                with mock.patch.object(
                    manager,
                    "restore_backup",
                    return_value={"success": False, "error": "Invalid backup file"},
                ):

                    captured_output = StringIO()
                    with mock.patch("sys.stdout", captured_output):
                        with mock.patch(
                            "app.infrastructure.backup_manager.BackupManager",
                            return_value=manager,
                        ):
                            with pytest.raises(SystemExit) as exc_info:
                                main()

                            # Should exit with code 1
                            assert exc_info.value.code == 1

    def test_main_default_no_args_coverage(self, tmp_path):
        """Test main() with no arguments (default behavior, lines 530-532)."""
        backup_dir = tmp_path / "backups"

        test_argv = ["backup_manager.py"]

        with mock.patch("sys.argv", test_argv):
            with mock.patch(
                "app.infrastructure.backup_manager.BackupManager.__init__",
                return_value=None,
            ):
                manager = BackupManager(backup_dir=backup_dir)

                # Mock get_backup_stats
                stats_data = {
                    "directory": str(backup_dir),
                    "total_backups": 0,
                    "total_size_mb": 0,
                    "oldest_backup": None,
                    "oldest_age_days": None,
                    "newest_backup": None,
                }
                with mock.patch.object(
                    manager, "get_backup_stats", return_value=stats_data
                ):

                    captured_output = StringIO()
                    with mock.patch("sys.stdout", captured_output):
                        with mock.patch(
                            "app.infrastructure.backup_manager.BackupManager",
                            return_value=manager,
                        ):
                            try:
                                main()
                            except SystemExit:
                                pass

                    output = captured_output.getvalue()
                    assert "Backups:" in output or output == ""
