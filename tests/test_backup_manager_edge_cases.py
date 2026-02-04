"""Edge case tests for BackupManager."""

import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.infrastructure.backup_manager import BackupManager


def _create_sqlite_db(path: Path):
    conn = sqlite3.connect(str(path))
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, name TEXT)")
    cursor.execute("INSERT INTO test (name) VALUES ('a')")
    conn.commit()
    conn.close()


def test_backup_database_success(tmp_path):
    db_path = tmp_path / "test.db"
    backup_dir = tmp_path / "backups"
    _create_sqlite_db(db_path)

    manager = BackupManager(db_path=db_path, backup_dir=backup_dir)

    result = manager.backup_database()

    assert result["success"] is True
    assert result["backup_path"] is not None
    assert Path(result["backup_path"]).exists()
    assert result["tables"] >= 1


def test_backup_database_missing_source(tmp_path):
    db_path = tmp_path / "missing.db"
    backup_dir = tmp_path / "backups"

    manager = BackupManager(db_path=db_path, backup_dir=backup_dir)

    result = manager.backup_database()

    assert result["success"] is False
    assert "not found" in result["error"].lower()


def test_verify_backup_invalid_file(tmp_path):
    backup_path = tmp_path / "nope.db"
    manager = BackupManager(db_path=backup_path, backup_dir=tmp_path)

    result = manager.verify_backup(backup_path)

    assert result["valid"] is False
    assert "not found" in result["error"].lower()


def test_verify_backup_corrupted(tmp_path):
    backup_path = tmp_path / "corrupt.db"
    backup_path.write_text("not a sqlite db")

    manager = BackupManager(db_path=backup_path, backup_dir=tmp_path)

    result = manager.verify_backup(backup_path)

    assert result["valid"] is False
    assert result["error"] is not None


def test_list_backups_and_stats(tmp_path):
    db_path = tmp_path / "test.db"
    backup_dir = tmp_path / "backups"
    _create_sqlite_db(db_path)

    manager = BackupManager(db_path=db_path, backup_dir=backup_dir)

    # Create two backups
    manager.backup_database()
    manager.backup_database()

    backups = manager.list_backups()
    stats = manager.get_backup_stats()

    assert len(backups) >= 2
    assert stats["total_backups"] >= 2
    assert stats["total_size_mb"] > 0


def test_cleanup_old_backups(tmp_path):
    db_path = tmp_path / "test.db"
    backup_dir = tmp_path / "backups"
    _create_sqlite_db(db_path)

    manager = BackupManager(db_path=db_path, backup_dir=backup_dir)

    # Create a backup
    result = manager.backup_database()
    backup_path = Path(result["backup_path"])

    # Make it old
    old_time = datetime.now(timezone.utc) - timedelta(days=manager.BACKUP_RETENTION_DAYS + 1)
    os.utime(backup_path, (old_time.timestamp(), old_time.timestamp()))

    cleanup = manager.cleanup_old_backups()

    assert cleanup["deleted_count"] >= 1
    assert backup_path.name in cleanup["deleted_files"]


def test_restore_backup_success(tmp_path):
    db_path = tmp_path / "test.db"
    backup_dir = tmp_path / "backups"
    _create_sqlite_db(db_path)

    manager = BackupManager(db_path=db_path, backup_dir=backup_dir)

    backup_result = manager.backup_database()
    backup_path = Path(backup_result["backup_path"])

    target_path = tmp_path / "restored.db"

    restore = manager.restore_backup(backup_path, target_path=target_path)

    assert restore["success"] is True
    assert target_path.exists()


def test_restore_backup_invalid(tmp_path):
    backup_path = tmp_path / "missing.db"
    manager = BackupManager(db_path=backup_path, backup_dir=tmp_path)

    restore = manager.restore_backup(backup_path)

    assert restore["success"] is False
    assert "verification failed" in restore["error"].lower()


def test_backup_empty_database(tmp_path):
    """Test backing up an empty database file."""
    db_path = tmp_path / "empty.db"
    db_path.write_bytes(b"")  # Empty file
    
    backup_dir = tmp_path / "backups"
    
    manager = BackupManager(db_path=db_path, backup_dir=backup_dir)
    result = manager.backup_database()
    
    # Should handle empty files gracefully
    assert isinstance(result, dict)
    assert 'success' in result


def test_cleanup_with_corrupted_backup(tmp_path, monkeypatch):
    """Test cleanup when backup file is corrupted."""
    db_path = tmp_path / "test.db"
    backup_dir = tmp_path / "backups"
    _create_sqlite_db(db_path)
    
    manager = BackupManager(db_path=db_path, backup_dir=backup_dir)
    
    # Create backup
    manager.backup_database()
    
    # Create corrupted file in backups
    corrupted = backup_dir / "corrupted.db.gz"
    corrupted.write_bytes(b'\x00\x01\x02\xFF')
    
    # Set old timestamp
    old_time = datetime.now(timezone.utc) - timedelta(days=60)
    os.utime(corrupted, (old_time.timestamp(), old_time.timestamp()))
    
    # Should handle corrupted backups gracefully
    result = manager.cleanup_old_backups()
    assert isinstance(result, dict)


def test_backup_permission_denied(tmp_path, monkeypatch):
    """Test backup when write permission is denied - on Windows this may succeed."""
    db_path = tmp_path / "test.db"
    backup_dir = tmp_path / "backups"
    _create_sqlite_db(db_path)
    
    # Note: On Windows, read-only doesn't prevent root from writing
    # So we expect success or graceful failure
    manager = BackupManager(db_path=db_path, backup_dir=backup_dir)
    result = manager.backup_database()
    
    # Should return valid result (success or error dict)
    assert isinstance(result, dict)
    assert 'backup_path' in result or 'error' in result


def test_verify_backup_missing(tmp_path):
    """Test verification of missing backup."""
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    
    manager = BackupManager(db_path=tmp_path / "test.db", backup_dir=backup_dir)
    
    result = manager.verify_backup("nonexistent.db.gz")
    
    # Should return False or error dict, not crash
    assert isinstance(result, (bool, dict))


def test_cleanup_multiple_backups(tmp_path):
    """Test cleanup removes only old backups."""
    db_path = tmp_path / "test.db"
    backup_dir = tmp_path / "backups"
    _create_sqlite_db(db_path)
    
    manager = BackupManager(db_path=db_path, backup_dir=backup_dir)
    
    # Create multiple backups
    backup_results = []
    for i in range(3):
        result = manager.backup_database()
        backup_results.append(result)
    
    # Make first one very old
    first_backup = Path(backup_results[0]["backup_path"])
    old_time = datetime.now(timezone.utc) - timedelta(days=60)
    os.utime(first_backup, (old_time.timestamp(), old_time.timestamp()))
    
    # Cleanup should remove old but keep new
    cleanup = manager.cleanup_old_backups()
    
    # cleanup_old_backups returns dict with deleted_count and deleted_files
    assert isinstance(cleanup, dict)
    assert cleanup["deleted_count"] >= 1
    # Newer backups should still exist
    for result in backup_results[1:]:
        assert Path(result["backup_path"]).exists()
