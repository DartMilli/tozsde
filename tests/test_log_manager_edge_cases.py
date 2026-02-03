"""Edge case tests for LogManager."""

import os
from datetime import datetime, timedelta
from pathlib import Path

from app.infrastructure.log_manager import LogManager


def _create_log_file(path: Path, content: bytes = b"test"):
    path.write_bytes(content)


def test_rotate_logs_and_cleanup(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    old_log = log_dir / "old.log"
    new_log = log_dir / "new.log"
    _create_log_file(old_log)
    _create_log_file(new_log)

    manager = LogManager(log_dir=log_dir)

    # Make old_log older than rotation threshold
    old_time = datetime.utcnow() - timedelta(days=manager.ROTATION_AGE_DAYS + 1)
    os.utime(old_log, (old_time.timestamp(), old_time.timestamp()))

    rotated = manager.rotate_logs()

    assert rotated["rotated_count"] >= 1
    assert "old.log" in rotated["rotated_files"]
    assert not old_log.exists()

    # Ensure archive exists
    archives = manager._find_archive_files()
    assert any(a.name.startswith("old.log") for a in archives)

    # Force cleanup of archives
    cleanup = manager.cleanup_all_archives()
    assert cleanup["deleted_count"] >= 1


def test_force_rotate_all_and_stats(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    log_file = log_dir / "app.jsonl"
    _create_log_file(log_file, content=b"logline")

    manager = LogManager(log_dir=log_dir)

    rotated = manager.force_rotate_all()
    assert rotated["rotated_count"] == 1

    stats = manager.get_log_stats()
    assert stats["total_files"] >= 1
    assert stats["archive_files"] >= 1


def test_cleanup_old_archives(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    manager = LogManager(log_dir=log_dir)

    # Create a fake archive file
    archive = log_dir / "test.log.gz"
    _create_log_file(archive)

    old_time = datetime.utcnow() - timedelta(days=manager.ARCHIVE_RETENTION_DAYS + 1)
    os.utime(archive, (old_time.timestamp(), old_time.timestamp()))

    result = manager.cleanup_old_archives()

    assert result["deleted_count"] == 1
    assert archive.name in result["deleted_files"]
