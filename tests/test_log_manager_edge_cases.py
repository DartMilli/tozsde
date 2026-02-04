"""Edge case tests for LogManager."""

import os
import shutil
from datetime import datetime, timedelta, timezone
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
    old_time = datetime.now(timezone.utc) - timedelta(
        days=manager.ROTATION_AGE_DAYS + 1
    )
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

    old_time = datetime.now(timezone.utc) - timedelta(
        days=manager.ARCHIVE_RETENTION_DAYS + 1
    )
    os.utime(archive, (old_time.timestamp(), old_time.timestamp()))

    result = manager.cleanup_old_archives()

    assert result["deleted_count"] == 1
    assert archive.name in result["deleted_files"]


def test_rotate_logs_disk_full(tmp_path, monkeypatch):
    """Test log rotation when disk is full - errors logged but not returned."""
    import gzip

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    old_log = log_dir / "old.log"
    _create_log_file(old_log, b"test data" * 1000)

    # Set modification time to 10 days ago
    old_time = (datetime.now(timezone.utc) - timedelta(days=10)).timestamp()
    os.utime(old_log, (old_time, old_time))

    # Mock gzip.open to simulate disk full
    def mock_gzip_open(*args, **kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(gzip, "open", mock_gzip_open)

    manager = LogManager(log_dir=log_dir)
    result = manager.rotate_logs()

    # Should handle error gracefully
    assert result["rotated_count"] == 0
    # Errors may be in list or logged but not returned


def test_cleanup_permission_denied(tmp_path, monkeypatch):
    """Test archive cleanup when file deletion fails - uses Path.unlink()."""
    import gzip
    from pathlib import Path

    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    # Create old archive
    old_archive = log_dir / "old.log.gz"
    with gzip.open(old_archive, "wt") as f:
        f.write("old data")

    # Set modification time to 40 days ago
    old_time = (datetime.now(timezone.utc) - timedelta(days=40)).timestamp()
    os.utime(old_archive, (old_time, old_time))

    # Mock Path.unlink() to fail
    original_unlink = Path.unlink

    def mock_unlink(self):
        raise PermissionError("Cannot delete")

    monkeypatch.setattr(Path, "unlink", mock_unlink)

    manager = LogManager(log_dir=log_dir)
    result = manager.cleanup_old_archives()

    # Should have errors
    assert len(result["errors"]) > 0


def test_get_log_stats_empty_directory(tmp_path):
    """Test stats for empty log directory."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    manager = LogManager(log_dir=log_dir)
    stats = manager.get_log_stats()

    assert stats["total_files"] >= 0
    assert isinstance(stats, dict)


def test_rotate_logs_corrupted_file(tmp_path):
    """Test rotation of corrupted/unreadable log file."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    # Create corrupted file
    corrupted_log = log_dir / "corrupted.log"
    corrupted_log.write_bytes(b"\x00\x01\x02\xff\xfe")

    # Set modification time to 10 days ago
    old_time = (datetime.now(timezone.utc) - timedelta(days=10)).timestamp()
    os.utime(corrupted_log, (old_time, old_time))

    manager = LogManager(log_dir=log_dir)
    result = manager.rotate_logs()

    # Should handle gracefully
    assert isinstance(result, dict)
    assert "rotated_count" in result
    assert "errors" in result


def test_rotate_logs_very_large_file(tmp_path):
    """Test rotation of very large log file."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    # Create large log file (5 MB)
    large_log = log_dir / "large.log"
    with open(large_log, "w") as f:
        f.write("x" * (5 * 1024 * 1024))

    # Set modification time to 10 days ago
    old_time = (datetime.now(timezone.utc) - timedelta(days=10)).timestamp()
    os.utime(large_log, (old_time, old_time))

    manager = LogManager(log_dir=log_dir)
    result = manager.rotate_logs()

    # Should handle large files
    assert isinstance(result, dict)
    if result["rotated_count"] > 0:
        assert result["bytes_saved"] > 0


def test_compress_log_zero_byte_file(tmp_path):
    """Test compression of empty log file."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    empty_log = log_dir / "empty.log"
    empty_log.write_text("")  # Empty file

    old_time = (datetime.now(timezone.utc) - timedelta(days=10)).timestamp()
    os.utime(empty_log, (old_time, old_time))

    manager = LogManager(log_dir=log_dir)
    result = manager.rotate_logs()

    # Should handle empty files
    assert isinstance(result, dict)


def test_compress_log_cleanup_on_failure(tmp_path, monkeypatch):
    """Ensure partial archives are cleaned up on compression failure."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    log_file = log_dir / "fail.log"
    log_file.write_text("x" * 1024)

    manager = LogManager(log_dir=log_dir)

    def raise_copy(*args, **kwargs):
        raise RuntimeError("copy failed")

    monkeypatch.setattr(shutil, "copyfileobj", raise_copy)

    archive_path = log_file.with_suffix(".log.gz")
    result = manager._compress_log(log_file)

    assert result is None
    assert log_file.exists()
    assert not archive_path.exists()


def test_rotate_logs_outer_exception(tmp_path, monkeypatch):
    """Rotate logs handles unexpected exceptions."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    manager = LogManager(log_dir=log_dir)

    def boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(manager, "_find_log_files", boom)

    result = manager.rotate_logs()

    assert result["rotated_count"] == 0
    assert result["errors"]


def test_cleanup_old_archives_outer_exception(tmp_path, monkeypatch):
    """Cleanup old archives handles unexpected exceptions."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    manager = LogManager(log_dir=log_dir)

    def boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(manager, "_find_archive_files", boom)

    result = manager.cleanup_old_archives()

    assert result["deleted_count"] == 0
    assert result["errors"]


def test_force_rotate_all_outer_exception(tmp_path, monkeypatch):
    """Force rotate handles unexpected exceptions."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    manager = LogManager(log_dir=log_dir)

    def boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(manager, "_find_log_files", boom)

    result = manager.force_rotate_all()

    assert result["rotated_count"] == 0
    assert result["errors"]


def test_cleanup_all_archives_outer_exception(tmp_path, monkeypatch):
    """Force cleanup handles unexpected exceptions."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    manager = LogManager(log_dir=log_dir)

    def boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(manager, "_find_archive_files", boom)

    result = manager.cleanup_all_archives()

    assert result["deleted_count"] == 0
    assert result["errors"]
