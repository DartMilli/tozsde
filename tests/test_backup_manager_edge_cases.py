"""Edge case tests for BackupManager."""

import os
import sqlite3
from datetime import datetime, timedelta
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
    old_time = datetime.utcnow() - timedelta(days=manager.BACKUP_RETENTION_DAYS + 1)
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
