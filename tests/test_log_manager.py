"""
Unit Tests for Log Manager Module (Sprint 5)

Tests log rotation and cleanup functionality for Raspberry Pi deployment:
- Log file rotation (compression to .gz)
- Old archive cleanup
- Log directory statistics
- File age handling
"""

import gzip
import os
import pytest
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from app.infrastructure.log_manager import LogManager


@pytest.fixture
def temp_log_dir():
    """Create temporary log directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / "logs"
        log_dir.mkdir()
        yield log_dir


@pytest.fixture
def log_manager(temp_log_dir):
    """Create LogManager instance with temporary directory."""
    return LogManager(log_dir=temp_log_dir)


def create_test_log(log_dir: Path, filename: str, content: str = "test log content\n", 
                    age_days: int = 0) -> Path:
    """
    Create a test log file with specified age.
    
    Args:
        log_dir: Directory to create log in
        filename: Name of log file
        content: Content to write
        age_days: How many days old to make the file
        
    Returns:
        Path to created log file
    """
    log_path = log_dir / filename
    
    # Write content
    with open(log_path, 'w') as f:
        f.write(content)
    
    # Set modification time if age specified
    if age_days > 0:
        old_time = time.time() - (age_days * 86400)
        os.utime(log_path, (old_time, old_time))
    
    return log_path


# --- Log Rotation Tests ---

def test_rotate_logs_no_files(log_manager):
    """Test rotation when no log files exist."""
    result = log_manager.rotate_logs()
    
    assert result['rotated_count'] == 0
    assert len(result['rotated_files']) == 0
    assert result['bytes_saved'] == 0


def test_rotate_logs_recent_files(log_manager, temp_log_dir):
    """Test rotation doesn't affect recent files."""
    # Create recent log (2 days old - below 7 day threshold)
    create_test_log(temp_log_dir, "recent.log", age_days=2)
    
    result = log_manager.rotate_logs()
    
    assert result['rotated_count'] == 0
    assert (temp_log_dir / "recent.log").exists()
    assert not (temp_log_dir / "recent.log.gz").exists()


def test_rotate_logs_old_files(log_manager, temp_log_dir):
    """Test rotation of old log files."""
    # Create old log (10 days old - above 7 day threshold)
    log_content = "test log content\n" * 100  # Make it compressible
    create_test_log(temp_log_dir, "old.log", content=log_content, age_days=10)
    
    original_size = (temp_log_dir / "old.log").stat().st_size
    
    result = log_manager.rotate_logs()
    
    assert result['rotated_count'] == 1
    assert 'old.log' in result['rotated_files']
    assert not (temp_log_dir / "old.log").exists()
    assert (temp_log_dir / "old.log.gz").exists()
    assert result['bytes_saved'] > 0  # Compression should save space


def test_rotate_logs_multiple_files(log_manager, temp_log_dir):
    """Test rotation of multiple log files."""
    # Create mix of old and recent files
    create_test_log(temp_log_dir, "old1.log", age_days=10)
    create_test_log(temp_log_dir, "old2.jsonl", age_days=15)
    create_test_log(temp_log_dir, "recent.log", age_days=2)
    
    result = log_manager.rotate_logs()
    
    assert result['rotated_count'] == 2
    assert 'old1.log' in result['rotated_files']
    assert 'old2.jsonl' in result['rotated_files']
    assert (temp_log_dir / "recent.log").exists()


def test_rotate_logs_compression_quality(log_manager, temp_log_dir):
    """Test that compression actually reduces file size."""
    # Create log with repetitive content (highly compressible)
    content = "This is a test log line\n" * 1000
    create_test_log(temp_log_dir, "compressible.log", content=content, age_days=10)
    
    original_size = (temp_log_dir / "compressible.log").stat().st_size
    
    result = log_manager.rotate_logs()
    
    compressed_size = (temp_log_dir / "compressible.log.gz").stat().st_size
    
    assert compressed_size < original_size * 0.5  # Should compress to < 50%
    assert result['bytes_saved'] > 0


def test_rotate_logs_preserves_content(log_manager, temp_log_dir):
    """Test that rotated logs preserve content when decompressed."""
    original_content = "Important log data\nLine 2\nLine 3\n"
    create_test_log(temp_log_dir, "preserve.log", content=original_content, age_days=10)
    
    log_manager.rotate_logs()
    
    # Decompress and verify content
    with gzip.open(temp_log_dir / "preserve.log.gz", 'rt') as f:
        decompressed_content = f.read()
    
    assert decompressed_content == original_content


# --- Archive Cleanup Tests ---

def test_cleanup_old_archives_no_files(log_manager):
    """Test cleanup when no archive files exist."""
    result = log_manager.cleanup_old_archives()
    
    assert result['deleted_count'] == 0
    assert len(result['deleted_files']) == 0
    assert result['bytes_freed'] == 0


def test_cleanup_old_archives_recent_files(log_manager, temp_log_dir):
    """Test cleanup doesn't affect recent archives."""
    # Create recent archive (5 days old - below 30 day threshold)
    archive_path = temp_log_dir / "recent.log.gz"
    with gzip.open(archive_path, 'wb') as f:
        f.write(b"test content")
    
    # Set recent mtime
    recent_time = time.time() - (5 * 86400)
    os.utime(archive_path, (recent_time, recent_time))
    
    result = log_manager.cleanup_old_archives()
    
    assert result['deleted_count'] == 0
    assert archive_path.exists()


def test_cleanup_old_archives_old_files(log_manager, temp_log_dir):
    """Test cleanup of old archive files."""
    # Create old archive (40 days old - above 30 day threshold)
    archive_path = temp_log_dir / "old.log.gz"
    test_content = b"test content" * 100
    with gzip.open(archive_path, 'wb') as f:
        f.write(test_content)
    
    # Set old mtime
    old_time = time.time() - (40 * 86400)
    os.utime(archive_path, (old_time, old_time))
    
    file_size = archive_path.stat().st_size
    
    result = log_manager.cleanup_old_archives()
    
    assert result['deleted_count'] == 1
    assert 'old.log.gz' in result['deleted_files']
    assert not archive_path.exists()
    assert result['bytes_freed'] == file_size


def test_cleanup_old_archives_multiple_files(log_manager, temp_log_dir):
    """Test cleanup of multiple old archives."""
    # Create mix of old and recent archives
    for i in range(3):
        archive_path = temp_log_dir / f"old{i}.log.gz"
        with gzip.open(archive_path, 'wb') as f:
            f.write(b"content")
        old_time = time.time() - (40 * 86400)
        os.utime(archive_path, (old_time, old_time))
    
    # Recent archive
    recent_path = temp_log_dir / "recent.log.gz"
    with gzip.open(recent_path, 'wb') as f:
        f.write(b"content")
    
    result = log_manager.cleanup_old_archives()
    
    assert result['deleted_count'] == 3
    assert recent_path.exists()


# --- Log Statistics Tests ---

def test_get_log_stats_empty_directory(log_manager):
    """Test statistics for empty log directory."""
    stats = log_manager.get_log_stats()
    
    assert stats['total_files'] == 0
    assert stats['log_files'] == 0
    assert stats['archive_files'] == 0
    assert stats['total_size_mb'] == 0.0
    assert stats['oldest_file'] is None


def test_get_log_stats_with_files(log_manager, temp_log_dir):
    """Test statistics with various log files."""
    # Create test files
    create_test_log(temp_log_dir, "test1.log", age_days=5)
    create_test_log(temp_log_dir, "test2.jsonl", age_days=10)
    
    # Create archive
    archive_path = temp_log_dir / "old.log.gz"
    with gzip.open(archive_path, 'wb') as f:
        f.write(b"archived content")
    
    stats = log_manager.get_log_stats()
    
    assert stats['total_files'] == 3
    assert stats['log_files'] == 2
    assert stats['archive_files'] == 1
    assert stats['total_size_mb'] >= 0  # Can be 0 for tiny files
    assert stats['oldest_file'] is not None
    assert stats['oldest_age_days'] >= 10


def test_get_log_stats_calculates_size(log_manager, temp_log_dir):
    """Test that statistics correctly calculate total size."""
    # Create files with known sizes
    content = "x" * 1024 * 512  # 512 KB
    create_test_log(temp_log_dir, "size_test.log", content=content)
    
    stats = log_manager.get_log_stats()
    
    assert stats['total_size_mb'] >= 0.5  # At least 0.5 MB


# --- Force Operations Tests ---

def test_force_rotate_all(log_manager, temp_log_dir):
    """Test force rotation of all logs regardless of age."""
    # Create recent logs (normally wouldn't be rotated)
    create_test_log(temp_log_dir, "recent1.log", age_days=1)
    create_test_log(temp_log_dir, "recent2.log", age_days=2)
    
    result = log_manager.force_rotate_all()
    
    assert result['rotated_count'] == 2
    assert not (temp_log_dir / "recent1.log").exists()
    assert not (temp_log_dir / "recent2.log").exists()
    assert (temp_log_dir / "recent1.log.gz").exists()
    assert (temp_log_dir / "recent2.log.gz").exists()


def test_cleanup_all_archives(log_manager, temp_log_dir):
    """Test force cleanup of all archives regardless of age."""
    # Create recent archives (normally wouldn't be deleted)
    for i in range(2):
        archive_path = temp_log_dir / f"recent{i}.log.gz"
        with gzip.open(archive_path, 'wb') as f:
            f.write(b"content")
    
    result = log_manager.cleanup_all_archives()
    
    assert result['deleted_count'] == 2
    assert not (temp_log_dir / "recent0.log.gz").exists()
    assert not (temp_log_dir / "recent1.log.gz").exists()


# --- Error Handling Tests ---

def test_rotate_logs_handles_permission_error(log_manager, temp_log_dir):
    """Test rotation handles permission errors gracefully."""
    create_test_log(temp_log_dir, "readonly.log", age_days=10)
    
    # Make file read-only on Windows by removing write permission
    log_path = temp_log_dir / "readonly.log"
    
    # Mock compression failure
    with patch.object(log_manager, '_compress_log', return_value=None):
        result = log_manager.rotate_logs()
    
    # Should not crash, but report error
    assert result['rotated_count'] == 0
    # Original file should still exist since compression failed
    assert log_path.exists()


def test_get_log_stats_missing_directory(temp_log_dir):
    """Test statistics when log directory doesn't exist."""
    non_existent = temp_log_dir / "missing"
    manager = LogManager(log_dir=non_existent)
    
    stats = manager.get_log_stats()
    
    assert stats['total_files'] == 0
    assert stats['directory'] == str(non_existent)


# --- Integration Tests ---

def test_full_rotation_cleanup_cycle(log_manager, temp_log_dir):
    """Test complete rotation and cleanup cycle."""
    # Day 0: Create logs
    create_test_log(temp_log_dir, "day0.log", age_days=0)
    
    # Day 8: Rotate (7+ days old)
    create_test_log(temp_log_dir, "day8.log", age_days=8)
    result = log_manager.rotate_logs()
    assert result['rotated_count'] == 1
    assert (temp_log_dir / "day8.log.gz").exists()
    
    # Day 40: Cleanup (30+ days old)
    old_archive = temp_log_dir / "day40.log.gz"
    with gzip.open(old_archive, 'wb') as f:
        f.write(b"old content")
    old_time = time.time() - (40 * 86400)
    os.utime(old_archive, (old_time, old_time))
    
    result = log_manager.cleanup_old_archives()
    assert result['deleted_count'] == 1
    assert not old_archive.exists()
    
    # Recent files should remain
    assert (temp_log_dir / "day0.log").exists()
    assert (temp_log_dir / "day8.log.gz").exists()
