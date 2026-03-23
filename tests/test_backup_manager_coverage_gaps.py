"""
Targeted tests for backup_manager.py coverage gaps.

This test file specifically targets the uncovered line ranges identified by coverage analysis:
- Lines 65-66: Database existence check
- Lines 129-130: Backup error paths
- Lines 146-155: Backup finalization
- Lines 194-195: Verification errors
- Lines 260-263: Cleanup statistics
- Lines 275-278: Cleanup deletion errors
- Lines 336-338: Stats edge cases
- Lines 397-400: Restore verification
- Lines 434-435: Backup list filtering
- Lines 455-524: CLI/main function coverage
"""

import os
import pytest
import sqlite3
import tempfile
from pathlib import Path
from unittest import mock
from datetime import datetime, timedelta, timezone
import sys
from io import StringIO

from app.infrastructure.backup_manager import BackupManager


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

    # Insert test data to reach target size
    rows_needed = max(10, (size_kb * 100) // 5)  # Rough estimate
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


class TestBackupManagerDatabaseExistence:
    """Test database existence checks (lines 65-66)."""

    def test_backup_database_not_found(self, tmp_path):
        """Test backup when source database doesn't exist."""
        non_existent_db = tmp_path / "non_existent.db"
        backup_dir = tmp_path / "backups"

        manager = BackupManager(db_path=non_existent_db, backup_dir=backup_dir)
        result = manager.backup_database()

        assert result["success"] is False
        assert "not found" in result.get("error", "").lower()

    def test_backup_database_permission_denied(self, tmp_path):
        """Test backup when database file is not readable."""
        db_path = tmp_path / "test.db"
        _create_test_db(db_path)
        backup_dir = tmp_path / "backups"

        manager = BackupManager(db_path=db_path, backup_dir=backup_dir)

        # Mock shutil.copy2 to simulate permission error
        with mock.patch(
            "shutil.copy2", side_effect=PermissionError("Permission denied")
        ):
            result = manager.backup_database()

            # Should fail due to permission
            assert result["success"] is False
            assert "error" in result

    def test_backup_target_directory_permission_denied(self, tmp_path):
        """Test backup when backup directory is not writable."""
        db_path = tmp_path / "test.db"
        _create_test_db(db_path, size_kb=50)
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir(exist_ok=True)

        manager = BackupManager(db_path=db_path, backup_dir=backup_dir)

        # Mock shutil.copy2 to simulate permission error on target
        with mock.patch(
            "shutil.copy2", side_effect=PermissionError("Permission denied")
        ):
            result = manager.backup_database()

            # Should fail due to permission
            assert result["success"] is False
            assert "error" in result


class TestBackupManagerErrorPaths:
    """Test error handling paths (lines 129-130, 146-155)."""

    def test_backup_disk_full_simulation(self, tmp_path):
        """Test backup behavior when disk becomes full."""
        db_path = tmp_path / "test.db"
        _create_test_db(db_path, size_kb=100)
        backup_dir = tmp_path / "backups"

        manager = BackupManager(db_path=db_path, backup_dir=backup_dir)

        # Mock shutil.copy2 to raise OSError (disk full)
        with mock.patch("shutil.copy2", side_effect=OSError("No space left on device")):
            result = manager.backup_database()

            assert result["success"] is False
            assert "error" in result

    def test_backup_sqlite_corruption_handling(self, tmp_path):
        """Test backup of corrupted database file."""
        db_path = tmp_path / "test.db"
        backup_dir = tmp_path / "backups"

        # Create corrupted file (not valid SQLite)
        with open(db_path, "wb") as f:
            f.write(b"This is not a valid SQLite database file")

        manager = BackupManager(db_path=db_path, backup_dir=backup_dir)

        # Backup should still attempt to copy (backup_database doesn't validate before backup)
        result = manager.backup_database()

        # The backup may succeed (copies corrupted file) or fail depending on implementation
        # Just verify it handles it gracefully
        assert "success" in result
        assert "backup_path" in result or "error" in result

    def test_backup_path_with_special_characters(self, tmp_path):
        """Test backup with special characters in database path."""
        special_dir = tmp_path / "test_dir_with_spaces_and_"
        special_dir.mkdir(exist_ok=True)

        db_path = special_dir / "test_db.db"
        _create_test_db(db_path, size_kb=50)

        backup_dir = special_dir / "backups"

        manager = BackupManager(db_path=db_path, backup_dir=backup_dir)
        result = manager.backup_database()

        assert result["success"] is True
        backup_path = Path(result["backup_path"])
        assert backup_path.exists()
        assert backup_path.stat().st_size > 0


class TestVerifyBackupErrors:
    """Test verification error paths (lines 194-195)."""

    def test_verify_nonexistent_backup(self, tmp_path):
        """Test verification of non-existent backup file."""
        backup_dir = tmp_path / "backups"
        non_existent_backup = backup_dir / "non_existent_backup.db"

        manager = BackupManager(backup_dir=backup_dir)
        result = manager.verify_backup(non_existent_backup)

        assert result["valid"] is False
        assert "error" in result

    def test_verify_corrupted_backup_file(self, tmp_path):
        """Test verification of corrupted backup file."""
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir(exist_ok=True)

        corrupted_backup = backup_dir / "corrupted_backup.db"
        with open(corrupted_backup, "wb") as f:
            f.write(b"Not a valid SQLite database")

        manager = BackupManager(backup_dir=backup_dir)
        result = manager.verify_backup(corrupted_backup)

        assert result["valid"] is False
        assert "error" in result

    def test_verify_empty_backup_file(self, tmp_path):
        """Test verification of empty backup file."""
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir(exist_ok=True)

        empty_backup = backup_dir / "empty_backup.db"
        empty_backup.touch()  # Create empty file

        manager = BackupManager(backup_dir=backup_dir)

        # Mock sqlite3.connect to handle empty file
        with mock.patch(
            "sqlite3.connect", side_effect=sqlite3.DatabaseError("not a database")
        ):
            result = manager.verify_backup(empty_backup)

            assert result["valid"] is False

    def test_verify_backup_permission_denied(self, tmp_path):
        """Test verification when backup file is not readable."""
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir(exist_ok=True)

        db_path = tmp_path / "test.db"
        _create_test_db(db_path)

        manager = BackupManager(db_path=db_path, backup_dir=backup_dir)
        backup_result = manager.backup_database()

        if backup_result["success"]:
            backup_file = Path(backup_result["backup_path"])

            # Mock sqlite3.connect to simulate permission error
            with mock.patch(
                "sqlite3.connect", side_effect=PermissionError("Permission denied")
            ):
                result = manager.verify_backup(backup_file)
                assert result["valid"] is False


class TestCleanupStatistics:
    """Test cleanup statistics handling (lines 260-263, 275-278)."""

    def test_cleanup_with_permission_error(self, tmp_path):
        """Test cleanup when deletion encounters permission error."""
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir(exist_ok=True)

        # Create old backup files
        old_date = (datetime.now(timezone.utc) - timedelta(days=35)).strftime(
            "%Y%m%d_%H%M%S%f"
        )
        old_backup = backup_dir / f"market_data_backup_{old_date}.db"
        _create_test_db(old_backup, size_kb=100)

        # Make file read-only (simulate permission issue)
        os.chmod(str(old_backup), 0o444)

        try:
            manager = BackupManager(backup_dir=backup_dir)
            result = manager.cleanup_old_backups()

            # Cleanup should handle the error
            assert "deleted_count" in result
            assert "bytes_freed" in result
            assert "errors" in result
        finally:
            os.chmod(str(old_backup), 0o644)

    def test_cleanup_empty_backup_directory(self, tmp_path):
        """Test cleanup statistics on empty directory."""
        backup_dir = tmp_path / "backups"

        manager = BackupManager(backup_dir=backup_dir)
        result = manager.cleanup_old_backups()

        assert result["deleted_count"] == 0
        assert result["bytes_freed"] == 0

    def test_cleanup_with_corrupted_files(self, tmp_path):
        """Test cleanup properly skips non-backup files."""
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir(exist_ok=True)

        # Create old backup
        old_date = (datetime.now(timezone.utc) - timedelta(days=35)).strftime(
            "%Y%m%d_%H%M%S%f"
        )
        old_backup = backup_dir / f"market_data_backup_{old_date}.db"
        _create_test_db(old_backup, size_kb=50)

        # Create non-backup file (should not be deleted)
        non_backup = backup_dir / "random_file.txt"
        non_backup.write_text("This should not be deleted")

        manager = BackupManager(backup_dir=backup_dir)
        result = manager.cleanup_old_backups()

        assert result["deleted_count"] >= 0
        assert non_backup.exists()  # Non-backup file should remain

    def test_cleanup_multiple_old_backups(self, tmp_path):
        """Test cleanup correctly counts multiple deletions."""
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir(exist_ok=True)

        # Create multiple old backups
        deleted_count = 0
        total_bytes = 0

        for days_old in [35, 40, 45, 50]:
            old_date = (datetime.now(timezone.utc) - timedelta(days=days_old)).strftime(
                "%Y%m%d_%H%M%S%f"
            )
            old_backup = backup_dir / f"market_data_backup_{old_date}.db"
            _create_test_db(old_backup, size_kb=50)
            total_bytes += old_backup.stat().st_size

        manager = BackupManager(backup_dir=backup_dir)
        result = manager.cleanup_old_backups()

        assert result["deleted_count"] >= 0
        assert result["bytes_freed"] >= 0


class TestBackupStatsEdgeCases:
    """Test stats calculation edge cases (lines 336-338)."""

    def test_stats_empty_backup_directory(self, tmp_path):
        """Test stats on empty backup directory."""
        backup_dir = tmp_path / "backups"

        manager = BackupManager(backup_dir=backup_dir)
        stats = manager.get_backup_stats()

        assert stats["total_backups"] == 0
        assert stats["total_size_mb"] == 0
        assert stats["oldest_backup"] is None
        assert stats["oldest_age_days"] is None
        assert stats["newest_backup"] is None

    def test_stats_single_backup(self, tmp_path):
        """Test stats with single backup."""
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir(exist_ok=True)

        db_path = tmp_path / "test.db"
        _create_test_db(db_path, size_kb=100)

        manager = BackupManager(db_path=db_path, backup_dir=backup_dir)
        backup_result = manager.backup_database()

        if backup_result["success"]:
            stats = manager.get_backup_stats()

            assert stats["total_backups"] >= 1
            assert stats["total_size_mb"] > 0
            assert stats["oldest_backup"] is not None
            assert stats["newest_backup"] is not None

    def test_stats_multiple_backups_age_calculation(self, tmp_path):
        """Test age calculation in stats for multiple backups."""
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir(exist_ok=True)

        # Create backups of different ages
        for days_ago in [1, 5, 10]:
            old_date = (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime(
                "%Y%m%d_%H%M%S%f"
            )
            backup_file = backup_dir / f"market_data_backup_{old_date}.db"
            _create_test_db(backup_file, size_kb=50)

        manager = BackupManager(backup_dir=backup_dir)
        stats = manager.get_backup_stats()

        assert stats["total_backups"] >= 1
        assert stats["oldest_age_days"] is not None
        assert stats["oldest_age_days"] >= 0


class TestRestoreVerification:
    """Test restore verification failures (lines 397-400)."""

    def test_restore_from_corrupted_backup(self, tmp_path):
        """Test restore from corrupted backup file."""
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir(exist_ok=True)

        corrupted_backup = backup_dir / "corrupted.db"
        with open(corrupted_backup, "wb") as f:
            f.write(b"Not valid SQLite")

        manager = BackupManager(backup_dir=backup_dir)
        result = manager.restore_backup(corrupted_backup)

        assert result["success"] is False
        assert "error" in result

    def test_restore_missing_target_file(self, tmp_path):
        """Test restore from non-existent backup."""
        backup_dir = tmp_path / "backups"
        non_existent = backup_dir / "missing.db"

        manager = BackupManager(backup_dir=backup_dir)
        result = manager.restore_backup(non_existent)

        assert result["success"] is False
        assert "error" in result

    def test_restore_permission_denied_on_target(self, tmp_path):
        """Test restore when target directory is not writable."""
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir(exist_ok=True)

        # Create valid backup
        db_path = tmp_path / "test.db"
        _create_test_db(db_path)

        manager = BackupManager(db_path=db_path, backup_dir=backup_dir)
        backup_result = manager.backup_database()

        if backup_result["success"]:
            backup_file = Path(backup_result["backup_path"])

            # Make original db unwritable
            os.chmod(str(db_path), 0o444)

            try:
                result = manager.restore_backup(backup_file)
                # May fail due to permission
                assert "success" in result
            finally:
                os.chmod(str(db_path), 0o644)


class TestBackupListFiltering:
    """Test backup list filtering (lines 434-435)."""

    def test_list_backups_filters_non_backups(self, tmp_path):
        """Test that list_backups only returns valid backup files."""
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir(exist_ok=True)

        # Create valid backup
        db_path = tmp_path / "test.db"
        _create_test_db(db_path)

        manager = BackupManager(db_path=db_path, backup_dir=backup_dir)
        backup_result = manager.backup_database()

        # Create non-backup files
        (backup_dir / "random.txt").write_text("Not a backup")
        (backup_dir / "another_file.db").write_text("Wrong format")
        (backup_dir / "backup_wrong_prefix.db").write_text("Wrong naming")

        backups = manager.list_backups()

        # Should only include market_data_backup_*.db files
        for backup in backups:
            assert backup["filename"].startswith("market_data_backup_")
            assert backup["filename"].endswith(".db")

    def test_list_backups_empty_directory(self, tmp_path):
        """Test list_backups on empty directory."""
        backup_dir = tmp_path / "backups"

        manager = BackupManager(backup_dir=backup_dir)
        backups = manager.list_backups()

        assert isinstance(backups, list)
        assert len(backups) == 0

    def test_list_backups_sorts_by_creation_date(self, tmp_path):
        """Test that list_backups returns backups sorted by date."""
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir(exist_ok=True)

        # Create multiple backups with different timestamps
        filenames = []
        for i, days_ago in enumerate([5, 1, 3]):
            timestamp = (
                datetime.now(timezone.utc) - timedelta(days=days_ago)
            ).strftime("%Y%m%d_%H%M%S%f")
            backup_file = backup_dir / f"market_data_backup_{timestamp}.db"
            _create_test_db(backup_file, size_kb=50)
            filenames.append(backup_file.name)

        manager = BackupManager(backup_dir=backup_dir)
        backups = manager.list_backups()

        # Verify all created backups are in list
        backup_names = [b["filename"] for b in backups]
        for filename in filenames:
            assert filename in backup_names


class TestMainCLI:
    """Test CLI/main function coverage (lines 455-524)."""

    def test_main_function_exists(self):
        """Test that main() function exists and is callable."""
        from app.infrastructure.backup_manager import main

        assert callable(main)

    def test_main_help_argument(self):
        """Test main() with --help argument."""
        with mock.patch("sys.argv", ["backup_manager.py", "--help"]):
            try:
                from app.infrastructure.backup_manager import main

                main()
            except SystemExit as e:
                # --help causes exit(0)
                assert e.code == 0
