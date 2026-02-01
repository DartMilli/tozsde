"""
Unit Tests for Database Backup Manager (Sprint 5)

Tests database backup functionality for Raspberry Pi deployment:
- Database backup creation
- Backup verification
- Old backup cleanup
- Backup restoration
- Statistics and monitoring
"""

import os
import pytest
import shutil
import sqlite3
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, Mock

from app.infrastructure.backup_manager import BackupManager


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        backup_dir = Path(tmpdir) / "backups"
        data_dir.mkdir()
        backup_dir.mkdir()
        
        yield {
            'data_dir': data_dir,
            'backup_dir': backup_dir,
            'db_path': data_dir / "test.db"
        }


@pytest.fixture
def test_database(temp_dirs):
    """Create a test database with sample data."""
    db_path = temp_dirs['db_path']
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Create sample tables
    cursor.execute("""
        CREATE TABLE test_table (
            id INTEGER PRIMARY KEY,
            name TEXT,
            value REAL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE ohlcv (
            ticker TEXT,
            date TEXT,
            close REAL,
            PRIMARY KEY (ticker, date)
        )
    """)
    
    # Insert sample data
    cursor.execute("INSERT INTO test_table VALUES (1, 'test', 123.45)")
    cursor.execute("INSERT INTO ohlcv VALUES ('VOO', '2026-01-15', 450.0)")
    
    conn.commit()
    conn.close()
    
    return db_path


@pytest.fixture
def backup_manager(temp_dirs, test_database):
    """Create BackupManager instance with test paths."""
    return BackupManager(
        db_path=test_database,
        backup_dir=temp_dirs['backup_dir']
    )


# --- Backup Creation Tests ---

def test_backup_database_success(backup_manager, temp_dirs):
    """Test successful database backup."""
    result = backup_manager.backup_database()
    
    assert result['success'] is True
    assert result['backup_path'] is not None
    assert result['size_mb'] > 0
    assert result['timestamp'] is not None
    assert 'tables' in result
    
    # Verify backup file exists
    backup_path = Path(result['backup_path'])
    assert backup_path.exists()
    assert backup_path.parent == temp_dirs['backup_dir']


def test_backup_database_creates_valid_backup(backup_manager):
    """Test that backup is a valid SQLite database."""
    result = backup_manager.backup_database()
    
    assert result['success'] is True
    
    # Open and query backup
    backup_path = Path(result['backup_path'])
    conn = sqlite3.connect(str(backup_path))
    cursor = conn.cursor()
    
    # Check tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    assert 'test_table' in tables
    assert 'ohlcv' in tables
    
    # Check data exists
    cursor.execute("SELECT * FROM test_table")
    data = cursor.fetchall()
    assert len(data) == 1
    assert data[0][1] == 'test'
    
    conn.close()


def test_backup_database_timestamped_filename(backup_manager):
    """Test backup filename contains timestamp."""
    result = backup_manager.backup_database()
    
    backup_path = Path(result['backup_path'])
    filename = backup_path.name
    
    assert filename.startswith('market_data_backup_')
    assert filename.endswith('.db')
    # Format: market_data_backup_YYYYMMDD_HHMMSSffffff.db (with microseconds and underscore)
    assert len(filename) == len('market_data_backup_20260201_064455123456.db')


def test_backup_database_missing_source(temp_dirs):
    """Test backup when source database doesn't exist."""
    non_existent_db = temp_dirs['data_dir'] / "missing.db"
    manager = BackupManager(
        db_path=non_existent_db,
        backup_dir=temp_dirs['backup_dir']
    )
    
    result = manager.backup_database()
    
    assert result['success'] is False
    assert 'not found' in result['error'].lower()


def test_backup_database_multiple_backups(backup_manager):
    """Test creating multiple backups doesn't overwrite."""
    result1 = backup_manager.backup_database()
    time.sleep(1)  # Ensure different timestamp
    result2 = backup_manager.backup_database()
    
    assert result1['success'] is True
    assert result2['success'] is True
    assert result1['backup_path'] != result2['backup_path']
    
    # Both files should exist
    assert Path(result1['backup_path']).exists()
    assert Path(result2['backup_path']).exists()


# --- Backup Verification Tests ---

def test_verify_backup_valid(backup_manager):
    """Test verification of valid backup."""
    # Create backup
    result = backup_manager.backup_database()
    backup_path = Path(result['backup_path'])
    
    # Verify it
    verification = backup_manager.verify_backup(backup_path)
    
    assert verification['valid'] is True
    assert verification['tables'] >= 2
    assert verification['error'] is None


def test_verify_backup_missing_file(backup_manager, temp_dirs):
    """Test verification of non-existent backup."""
    non_existent = temp_dirs['backup_dir'] / "missing.db"
    
    verification = backup_manager.verify_backup(non_existent)
    
    assert verification['valid'] is False
    assert 'not found' in verification['error'].lower()


def test_verify_backup_corrupted_file(backup_manager, temp_dirs):
    """Test verification of corrupted backup."""
    # Create corrupted file
    corrupted_path = temp_dirs['backup_dir'] / "corrupted.db"
    with open(corrupted_path, 'wb') as f:
        f.write(b"This is not a valid SQLite database")
    
    verification = backup_manager.verify_backup(corrupted_path)
    
    assert verification['valid'] is False
    assert verification['error'] is not None


# --- Backup Cleanup Tests ---

def test_cleanup_old_backups_no_files(backup_manager):
    """Test cleanup when no backups exist."""
    result = backup_manager.cleanup_old_backups()
    
    assert result['deleted_count'] == 0
    assert len(result['deleted_files']) == 0
    assert result['bytes_freed'] == 0


def test_cleanup_old_backups_recent_files(backup_manager, temp_dirs):
    """Test cleanup doesn't affect recent backups."""
    # Create recent backup
    result = backup_manager.backup_database()
    backup_path = Path(result['backup_path'])
    
    # Try to cleanup (should not delete recent backup)
    cleanup_result = backup_manager.cleanup_old_backups()
    
    assert cleanup_result['deleted_count'] == 0
    assert backup_path.exists()


def test_cleanup_old_backups_old_files(backup_manager, temp_dirs):
    """Test cleanup of old backups."""
    # Create old backup
    result = backup_manager.backup_database()
    backup_path = Path(result['backup_path'])
    
    # Make it old (40 days)
    old_time = time.time() - (40 * 86400)
    os.utime(backup_path, (old_time, old_time))
    
    file_size = backup_path.stat().st_size
    
    # Cleanup
    cleanup_result = backup_manager.cleanup_old_backups()
    
    assert cleanup_result['deleted_count'] == 1
    assert backup_path.name in cleanup_result['deleted_files']
    assert not backup_path.exists()
    assert cleanup_result['bytes_freed'] == file_size


def test_cleanup_old_backups_mixed_ages(backup_manager, temp_dirs):
    """Test cleanup with mix of old and recent backups."""
    # Create 3 backups
    backups = []
    for i in range(3):
        result = backup_manager.backup_database()
        backups.append(Path(result['backup_path']))
        time.sleep(0.1)
    
    # Make first two old
    old_time = time.time() - (40 * 86400)
    for i in range(2):
        os.utime(backups[i], (old_time, old_time))
    
    # Cleanup
    cleanup_result = backup_manager.cleanup_old_backups()
    
    assert cleanup_result['deleted_count'] == 2
    assert not backups[0].exists()
    assert not backups[1].exists()
    assert backups[2].exists()  # Recent one remains


# --- Backup Statistics Tests ---

def test_get_backup_stats_empty(backup_manager):
    """Test statistics when no backups exist."""
    stats = backup_manager.get_backup_stats()
    
    assert stats['total_backups'] == 0
    assert stats['total_size_mb'] == 0.0
    assert stats['oldest_backup'] is None
    assert stats['newest_backup'] is None


def test_get_backup_stats_with_backups(backup_manager):
    """Test statistics with multiple backups."""
    # Create 3 backups
    for i in range(3):
        backup_manager.backup_database()
        time.sleep(0.1)
    
    stats = backup_manager.get_backup_stats()
    
    assert stats['total_backups'] == 3
    assert stats['total_size_mb'] > 0
    assert stats['oldest_backup'] is not None
    assert stats['newest_backup'] is not None
    assert stats['oldest_age_days'] is not None


def test_get_backup_stats_calculates_size(backup_manager):
    """Test that statistics correctly calculate total size."""
    # Create backup
    backup_manager.backup_database()
    
    stats = backup_manager.get_backup_stats()
    
    # Size should be reasonable for small test database
    assert stats['total_size_mb'] > 0
    assert stats['total_size_mb'] < 10  # Should be small


# --- Backup Restoration Tests ---

def test_restore_backup_success(backup_manager, temp_dirs):
    """Test successful backup restoration."""
    # Create backup
    backup_result = backup_manager.backup_database()
    backup_path = Path(backup_result['backup_path'])
    
    # Modify original database
    conn = sqlite3.connect(str(backup_manager.db_path))
    conn.execute("INSERT INTO test_table VALUES (2, 'new', 999.0)")
    conn.commit()
    conn.close()
    
    # Create new target for restoration
    restore_target = temp_dirs['data_dir'] / "restored.db"
    
    # Restore
    restore_result = backup_manager.restore_backup(backup_path, restore_target)
    
    assert restore_result['success'] is True
    assert restore_target.exists()
    
    # Verify restored data (should not have the new row)
    conn = sqlite3.connect(str(restore_target))
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM test_table")
    count = cursor.fetchone()[0]
    conn.close()
    
    assert count == 1  # Only original row


def test_restore_backup_invalid_backup(backup_manager, temp_dirs):
    """Test restoration fails for invalid backup."""
    # Create corrupted backup
    corrupted_path = temp_dirs['backup_dir'] / "corrupted.db"
    with open(corrupted_path, 'wb') as f:
        f.write(b"not a database")
    
    restore_result = backup_manager.restore_backup(corrupted_path)
    
    assert restore_result['success'] is False
    assert 'verification failed' in restore_result['error'].lower()


def test_restore_backup_creates_safety_backup(backup_manager, temp_dirs):
    """Test that restore creates backup of current database."""
    # Create backup
    backup_result = backup_manager.backup_database()
    backup_path = Path(backup_result['backup_path'])
    
    # Restore to same location (should create .before_restore)
    restore_result = backup_manager.restore_backup(backup_path)
    
    assert restore_result['success'] is True
    
    # Check safety backup exists
    safety_backup = backup_manager.db_path.with_suffix('.db.before_restore')
    assert safety_backup.exists()


# --- List Backups Tests ---

def test_list_backups_empty(backup_manager):
    """Test listing when no backups exist."""
    backups = backup_manager.list_backups()
    
    assert len(backups) == 0


def test_list_backups_with_files(backup_manager):
    """Test listing multiple backups."""
    # Create 3 backups
    for i in range(3):
        backup_manager.backup_database()
        time.sleep(0.1)
    
    backups = backup_manager.list_backups()
    
    assert len(backups) == 3
    
    # Check each backup has required fields
    for backup in backups:
        assert 'filename' in backup
        assert 'path' in backup
        assert 'size_mb' in backup
        assert 'created' in backup
        assert 'age_days' in backup


def test_list_backups_sorted_newest_first(backup_manager):
    """Test backups list is sorted consistently."""
    # Create 3 backups with delays
    for i in range(3):
        backup_manager.backup_database()
        time.sleep(0.2)
    
    backups = backup_manager.list_backups()
    
    # Check that we get all 3 backups and they're sorted consistently
    assert len(backups) == 3
    # Verify sort is consistent (by checking timestamps are in order)
    timestamps = [b['created'] for b in backups]
    assert timestamps == sorted(timestamps, reverse=True)


# --- Edge Cases and Error Handling ---

def test_backup_manager_creates_backup_directory(temp_dirs):
    """Test that backup manager creates directory if missing."""
    non_existent_dir = temp_dirs['backup_dir'] / "new_backups"
    
    # Should create directory automatically
    manager = BackupManager(
        db_path=temp_dirs['db_path'],
        backup_dir=non_existent_dir
    )
    
    assert non_existent_dir.exists()


def test_find_backup_files_pattern_matching(backup_manager, temp_dirs):
    """Test that only correct backup files are found."""
    # Create correct backup
    backup_manager.backup_database()
    
    # Create files that shouldn't match
    (temp_dirs['backup_dir'] / "other_file.db").touch()
    (temp_dirs['backup_dir'] / "backup_wrong_format.db").touch()
    
    backups = backup_manager._find_backup_files()
    
    # Should only find the correctly named backup
    assert len(backups) == 1
    assert 'market_data_backup_' in backups[0].name
