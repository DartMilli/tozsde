"""CLI coverage tests for LogManager main() entry point."""

import sys

import app.infrastructure.log_manager as log_manager


class DummyManager:
    """Minimal LogManager stub for CLI testing."""

    calls = []

    def __init__(self, *args, **kwargs):
        pass

    def rotate_logs(self):
        type(self).calls.append("rotate_logs")
        return {
            "rotated_count": 1,
            "rotated_files": ["a.log"],
            "errors": [],
            "bytes_saved": 10,
        }

    def cleanup_old_archives(self):
        type(self).calls.append("cleanup_old_archives")
        return {
            "deleted_count": 1,
            "deleted_files": ["a.log.gz"],
            "errors": [],
            "bytes_freed": 10,
        }

    def force_rotate_all(self):
        type(self).calls.append("force_rotate_all")
        return {
            "rotated_count": 1,
            "rotated_files": ["a.log"],
            "errors": [],
            "bytes_saved": 0,
        }

    def cleanup_all_archives(self):
        type(self).calls.append("cleanup_all_archives")
        return {
            "deleted_count": 1,
            "deleted_files": ["a.log.gz"],
            "errors": [],
            "bytes_freed": 0,
        }

    def get_log_stats(self):
        type(self).calls.append("get_log_stats")
        return {
            "total_files": 2,
            "log_files": 1,
            "archive_files": 1,
            "total_size_mb": 0.1,
            "oldest_file": "a.log",
            "oldest_age_days": 1.0,
            "directory": "C:/tmp/logs",
        }


def _run_cli(args, monkeypatch):
    DummyManager.calls = []
    monkeypatch.setattr(log_manager, "LogManager", DummyManager)
    monkeypatch.setattr(sys, "argv", ["log_manager.py"] + args)
    log_manager.main()


def test_cli_rotate(monkeypatch):
    _run_cli(["--rotate"], monkeypatch)
    assert "rotate_logs" in DummyManager.calls


def test_cli_cleanup(monkeypatch):
    _run_cli(["--cleanup"], monkeypatch)
    assert "cleanup_old_archives" in DummyManager.calls


def test_cli_force_rotate(monkeypatch):
    _run_cli(["--force-rotate"], monkeypatch)
    assert "force_rotate_all" in DummyManager.calls


def test_cli_force_cleanup(monkeypatch):
    _run_cli(["--force-cleanup"], monkeypatch)
    assert "cleanup_all_archives" in DummyManager.calls


def test_cli_stats(monkeypatch):
    _run_cli(["--stats"], monkeypatch)
    assert "get_log_stats" in DummyManager.calls


def test_cli_default(monkeypatch):
    _run_cli([], monkeypatch)
    assert "get_log_stats" in DummyManager.calls
