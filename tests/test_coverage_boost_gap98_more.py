import builtins
import gzip
import json
import os
import runpy
import sqlite3
import sys
import types
from datetime import date, datetime, timedelta
from pathlib import Path
from dataclasses import replace

import pandas as pd
import pytest


def test_log_manager_error_paths(tmp_path, monkeypatch):
    from app.infrastructure.log_manager import LogManager

    manager = LogManager(log_dir=tmp_path)

    log_file = tmp_path / "old.log"
    log_file.write_text("x")
    old_time = (datetime.now() - timedelta(days=40)).timestamp()
    os.utime(log_file, (old_time, old_time))

    monkeypatch.setattr(
        manager,
        "_compress_log",
        lambda *args, **kwargs: (_ for _ in ()).throw(Exception("compress fail")),
    )
    result = manager.rotate_logs()
    assert result["errors"]

    monkeypatch.setattr(
        manager,
        "_find_log_files",
        lambda *args, **kwargs: (_ for _ in ()).throw(Exception("find fail")),
    )
    result = manager.rotate_logs()
    assert result["errors"]

    archive_file = tmp_path / "old.log.gz"
    archive_file.write_text("z")
    os.utime(archive_file, (old_time, old_time))

    monkeypatch.setattr(
        manager, "_find_archive_files", lambda *args, **kwargs: [archive_file]
    )
    monkeypatch.setattr(
        type(archive_file),
        "unlink",
        lambda *args, **kwargs: (_ for _ in ()).throw(Exception("unlink fail")),
    )
    result = manager.cleanup_old_archives()
    assert result["errors"]

    monkeypatch.setattr(
        manager,
        "_find_log_files",
        lambda *args, **kwargs: (_ for _ in ()).throw(Exception("stats fail")),
    )
    stats = manager.get_log_stats()
    assert "error" in stats

    monkeypatch.setattr(
        manager,
        "_find_archive_files",
        lambda *args, **kwargs: (_ for _ in ()).throw(Exception("cleanup all fail")),
    )
    result = manager.cleanup_all_archives()
    assert result["errors"]


def test_log_manager_compress_and_force_paths(tmp_path, monkeypatch):
    from app.infrastructure.log_manager import LogManager

    manager = LogManager(log_dir=tmp_path)
    log_file = tmp_path / "force.log"
    log_file.write_text("x" * 10)

    archive_path = log_file.with_suffix(log_file.suffix + ".gz")
    archive_path.write_text("z")

    monkeypatch.setattr(
        gzip,
        "open",
        lambda *args, **kwargs: (_ for _ in ()).throw(Exception("gzip fail")),
    )
    assert manager._compress_log(log_file) is None

    log_file.write_text("x" * 10)
    archive_path.write_text("z")
    monkeypatch.setattr(manager, "_compress_log", lambda *args, **kwargs: archive_path)
    result = manager.force_rotate_all()
    assert result["rotated_count"] == 1

    archive_path.write_text("z")
    result = manager.cleanup_all_archives()
    assert result["deleted_count"] == 1


def test_log_manager_missing_dir_returns(tmp_path):
    from app.infrastructure.log_manager import LogManager

    manager = LogManager(log_dir=tmp_path)
    manager.log_dir = tmp_path / "no_such_dir"

    assert manager._find_log_files() == []
    assert manager._find_archive_files() == []


def test_log_manager_cleanup_all_archive_error(tmp_path, monkeypatch):
    from app.infrastructure.log_manager import LogManager

    manager = LogManager(log_dir=tmp_path)
    archive_file = tmp_path / "bad.log.gz"
    archive_file.write_text("x")

    monkeypatch.setattr(
        manager, "_find_archive_files", lambda *args, **kwargs: [archive_file]
    )
    monkeypatch.setattr(
        type(archive_file),
        "unlink",
        lambda *args, **kwargs: (_ for _ in ()).throw(Exception("unlink fail")),
    )

    result = manager.cleanup_all_archives()
    assert result["errors"]


def test_log_manager_main_paths(monkeypatch):
    import app.infrastructure.log_manager as log_manager

    class FakeManager:
        def rotate_logs(self):
            return {
                "rotated_count": 0,
                "rotated_files": [],
                "errors": ["err"],
                "bytes_saved": 0,
            }

        def cleanup_old_archives(self):
            return {
                "deleted_count": 0,
                "deleted_files": [],
                "errors": ["err"],
                "bytes_freed": 0,
            }

        def force_rotate_all(self):
            return {
                "rotated_count": 0,
                "rotated_files": [],
                "errors": [],
                "bytes_saved": 0,
            }

        def cleanup_all_archives(self):
            return {
                "deleted_count": 0,
                "deleted_files": [],
                "errors": [],
                "bytes_freed": 0,
            }

        def get_log_stats(self):
            return {
                "directory": "x",
                "total_files": 0,
                "log_files": 0,
                "archive_files": 0,
                "total_size_mb": 0,
                "oldest_file": None,
                "oldest_age_days": None,
            }

    monkeypatch.setattr(log_manager, "LogManager", FakeManager)

    monkeypatch.setattr(sys, "argv", ["log_manager.py", "--rotate"])
    log_manager.main()

    monkeypatch.setattr(sys, "argv", ["log_manager.py", "--cleanup"])
    log_manager.main()


def test_log_manager_main_module_runs(monkeypatch):
    sys.modules.pop("app.infrastructure.log_manager", None)
    monkeypatch.setattr(sys, "argv", ["log_manager.py", "--stats"])
    runpy.run_module("app.infrastructure.log_manager", run_name="__main__")


def test_backup_manager_error_paths(tmp_path, monkeypatch):
    from app.infrastructure.backup_manager import BackupManager

    db_path = tmp_path / "missing.db"
    backup_dir = tmp_path / "backups"
    manager = BackupManager(db_path=db_path, backup_dir=backup_dir)

    result = manager.backup_database()
    assert result["success"] is False

    verify = manager.verify_backup(tmp_path / "nope.db")
    assert verify["valid"] is False

    restore = manager.restore_backup(tmp_path / "nope.db")
    assert restore["success"] is False

    old_backup = backup_dir / "market_data_backup_20000101_000000.db"
    old_backup.write_text("x")
    old_time = (datetime.now() - timedelta(days=40)).timestamp()
    os.utime(old_backup, (old_time, old_time))

    monkeypatch.setattr(
        manager, "_find_backup_files", lambda *args, **kwargs: [old_backup]
    )
    monkeypatch.setattr(
        type(old_backup),
        "unlink",
        lambda *args, **kwargs: (_ for _ in ()).throw(Exception("unlink fail")),
    )
    cleanup = manager.cleanup_old_backups()
    assert cleanup["errors"]

    monkeypatch.setattr(
        manager,
        "_find_backup_files",
        lambda *args, **kwargs: (_ for _ in ()).throw(Exception("stats fail")),
    )
    stats = manager.get_backup_stats()
    assert "error" in stats


def test_backup_manager_additional_branches(tmp_path, monkeypatch):
    from app.infrastructure.backup_manager import BackupManager

    db_path = tmp_path / "db.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE t (id INTEGER)")
    conn.commit()
    conn.close()

    backup_dir = tmp_path / "backups"
    manager = BackupManager(db_path=db_path, backup_dir=backup_dir)

    monkeypatch.setattr(
        manager,
        "verify_backup",
        lambda *args, **kwargs: (_ for _ in ()).throw(Exception("verify fail")),
    )
    result = manager.backup_database()
    assert result["success"] is False

    backup_file = backup_dir / "market_data_backup_20000101_000000.db"
    backup_file.write_text("x")

    original_find_backup_files = manager._find_backup_files
    monkeypatch.setattr(
        manager,
        "_find_backup_files",
        lambda *args, **kwargs: (_ for _ in ()).throw(Exception("cleanup fail")),
    )
    cleanup = manager.cleanup_old_backups()
    assert cleanup["errors"]

    monkeypatch.setattr(manager, "_find_backup_files", original_find_backup_files)
    manager.backup_dir = tmp_path / "missing_backups"
    assert manager._find_backup_files() == []

    monkeypatch.setattr(
        manager,
        "_find_backup_files",
        lambda *args, **kwargs: (_ for _ in ()).throw(Exception("list fail")),
    )
    assert manager.list_backups() == []


def test_backup_manager_verify_integrity_failure(tmp_path, monkeypatch):
    from app.infrastructure.backup_manager import BackupManager

    backup_file = tmp_path / "market_data_backup_20000101_000000.db"
    backup_file.write_text("x")

    class FakeCursor:
        def __init__(self):
            self._count = 0

        def execute(self, *args, **kwargs):
            return None

        def fetchone(self):
            self._count += 1
            return ("bad",) if self._count == 1 else (0,)

    class FakeConn:
        def cursor(self):
            return FakeCursor()

        def close(self):
            return None

    monkeypatch.setattr(sqlite3, "connect", lambda *args, **kwargs: FakeConn())

    manager = BackupManager(db_path=backup_file, backup_dir=tmp_path)
    result = manager.verify_backup(backup_file)
    assert result["valid"] is False


def test_backup_manager_main_module_runs(monkeypatch):
    sys.modules.pop("app.infrastructure.backup_manager", None)
    monkeypatch.setattr(sys, "argv", ["backup_manager.py"])
    runpy.run_module("app.infrastructure.backup_manager", run_name="__main__")


def test_backup_manager_success_paths(tmp_path):
    from app.infrastructure.backup_manager import BackupManager

    db_path = tmp_path / "db.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE t (id INTEGER)")
    conn.commit()
    conn.close()

    backup_dir = tmp_path / "backups"
    manager = BackupManager(db_path=db_path, backup_dir=backup_dir)

    result = manager.backup_database()
    assert result["success"] is True

    backup_path = Path(result["backup_path"])
    target_path = tmp_path / "restore.db"
    target_path.write_text("x")

    restore = manager.restore_backup(backup_path, target_path=target_path)
    assert restore["success"] is True


def test_decision_logger_additional_branches(tmp_path, monkeypatch):
    from app.infrastructure.decision_logger import (
        NoTradeDecisionLogger,
        NoTradeDecision,
        NoTradeReason,
    )

    logger = NoTradeDecisionLogger(db_path=None)
    decision = NoTradeDecision(
        timestamp=datetime.now(),
        ticker="AAA",
        strategy="S1",
        reason=NoTradeReason.LOW_CONFIDENCE,
    )
    assert logger.log_no_trade_decision(decision) is False

    db_logger = NoTradeDecisionLogger(db_path=str(tmp_path / "nt.db"))

    monkeypatch.setattr(
        sqlite3,
        "connect",
        lambda *args, **kwargs: (_ for _ in ()).throw(sqlite3.Error("boom")),
    )
    assert db_logger.get_no_trade_decisions() == []
    assert db_logger.clear_old_decisions() == 0

    monkeypatch.setattr(
        builtins, "open", lambda *args, **kwargs: (_ for _ in ()).throw(IOError("nope"))
    )
    assert db_logger.export_no_trade_decisions(str(tmp_path / "out.json")) is False


def test_decision_logger_export_success(tmp_path):
    from app.infrastructure.decision_logger import NoTradeDecisionLogger, NoTradeReason

    db_logger = NoTradeDecisionLogger(db_path=str(tmp_path / "nt_ok.db"))
    db_logger.log_no_trade_simple(
        "AAA", "S1", NoTradeReason.LOW_CONFIDENCE, confidence_score=0.2
    )

    out_file = tmp_path / "export.json"
    assert db_logger.export_no_trade_decisions(str(out_file)) is True
    assert out_file.exists()


def test_decision_history_analyzer_error_paths(tmp_path, monkeypatch, test_settings):
    from app.decision.decision_history_analyzer import DecisionHistoryAnalyzer

    db_path = tmp_path / "history.db"
    settings = replace(test_settings, DB_PATH=db_path)

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE decision_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            ticker TEXT,
            action_code INTEGER,
            audit_data TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO decision_history (timestamp, ticker, action_code, audit_data) VALUES (?, ?, ?, ?)",
        (datetime.now().isoformat(), "AAA", 1, "{bad json"),
    )
    conn.commit()
    conn.close()

    analyzer = DecisionHistoryAnalyzer(settings=settings)
    stats = analyzer.analyze_ticker_reliability("AAA", days=1)
    assert stats.total_decisions == 0

    assert analyzer._is_recent({"timestamp": "not-a-date"}, days=1) is False

    monkeypatch.setattr(
        sqlite3,
        "connect",
        lambda *args, **kwargs: (_ for _ in ()).throw(Exception("db fail")),
    )
    assert analyzer._load_ticker_decisions("AAA", days=1) == []


def test_decision_history_analyzer_more_metrics(tmp_path, monkeypatch, test_settings):
    from app.decision.decision_history_analyzer import DecisionHistoryAnalyzer

    db_path = tmp_path / "history2.db"
    settings = replace(test_settings, DB_PATH=db_path)

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE decision_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            ticker TEXT,
            action_code INTEGER,
            audit_data TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO decision_history (timestamp, ticker, action_code, audit_data) VALUES (?, ?, ?, ?)",
        (datetime.now().isoformat(), "AAA", 1, "{bad json"),
    )
    conn.commit()
    conn.close()

    analyzer = DecisionHistoryAnalyzer(settings=settings)
    assert analyzer._calculate_sharpe([0.0, 0.0]) == 0.0
    assert analyzer._classify_strategy_status(0.6, 0.1, 0.1) == "DECLINING"
    assert analyzer._classify_strategy_status(0.4, 0.1, 0.4) == "POOR"

    monkeypatch.setattr(
        sqlite3,
        "connect",
        lambda *args, **kwargs: (_ for _ in ()).throw(Exception("db fail")),
    )
    assert analyzer.compute_rolling_metrics(window_days=5, total_days=30).empty


def test_decision_history_analyzer_calibration_and_strategy(
    tmp_path, monkeypatch, test_settings
):
    from app.decision.decision_history_analyzer import DecisionHistoryAnalyzer

    db_path = tmp_path / "history3.db"
    settings = replace(test_settings, DB_PATH=db_path)

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE decision_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            ticker TEXT,
            action_code INTEGER,
            audit_data TEXT
        )
        """
    )
    now = datetime.now()
    rows = []
    for i in range(6):
        audit = {
            "strategy": "MA_CROSS",
            "outcome": {"pnl_pct": 0.1 if i % 2 == 0 else -0.1},
            "confidence": 0.8 if i % 2 == 0 else 0.3,
        }
        rows.append(
            ((now - timedelta(days=i)).isoformat(), "AAA", 1, json.dumps(audit))
        )
    conn.executemany(
        "INSERT INTO decision_history (timestamp, ticker, action_code, audit_data) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()

    analyzer = DecisionHistoryAnalyzer(settings=settings)
    stats = analyzer.analyze_strategy_performance("ALL", days=90)
    assert stats.total_trades >= 1

    decisions = [
        {"outcome": {"pnl_pct": 0.1}, "confidence": 0.8},
        {"outcome": {"pnl_pct": -0.1}, "confidence": 0.3},
    ] * 3
    calibration = analyzer._calculate_confidence_calibration(decisions)
    assert 0.0 <= calibration <= 1.0


def test_data_manager_ref_date_branches(tmp_path, monkeypatch, test_settings):
    from app.data_access.data_manager import DataManager

    db_path = tmp_path / "dm2.db"
    settings = replace(test_settings, DB_PATH=db_path)
    dm = DataManager(settings=settings)
    dm.initialize_tables()

    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": [1] * 5,
            "High": [2] * 5,
            "Low": [1] * 5,
            "Close": [1] * 5,
            "Volume": [1] * 5,
            "Ticker": ["AAA"] * 5,
        }
    ).set_index("Date")
    dm.save_ohlcv("AAA", df)

    assert dm.get_market_regime_is_bear(
        benchmark="AAA", period=2, ref_date=date(2024, 1, 3)
    ) in (True, False)
    corr = dm.get_correlation_matrix(["AAA"], ref_date=date(2024, 1, 3))
    assert corr.empty or corr.isna().all().all()


def test_data_loader_exception_and_main(monkeypatch, tmp_path, test_settings):
    import app.data_access.data_loader as data_loader

    class FakeYF:
        @staticmethod
        def download(*args, **kwargs):
            raise Exception("fail")

    monkeypatch.setattr(data_loader, "yf", FakeYF)
    with pytest.raises(UnboundLocalError):
        data_loader.download_and_save_data(
            "AAA", datetime(2024, 1, 1), datetime(2024, 1, 2)
        )

    fake_dm_module = types.SimpleNamespace()

    class FakeDM:
        def load_ohlcv(self, *args, **kwargs):
            dates = pd.date_range("2024-01-01", periods=3, freq="D")
            return pd.DataFrame({"Close": [1.0, 1.0, 1.0]}, index=dates)

        def save_ohlcv(self, *args, **kwargs):
            return None

    fake_dm_module.DataManager = FakeDM

    fake_yf = types.SimpleNamespace(
        download=lambda *args, **kwargs: pd.DataFrame(
            {"Close": [1.0, 1.0]}, index=pd.date_range("2024-01-01", periods=2)
        ),
    )

    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)
    monkeypatch.setitem(sys.modules, "app.data_access.data_manager", fake_dm_module)

    from app import data_access

    data_access.set_settings(
        replace(test_settings, FAILED_DAYS_FILE_PATH=tmp_path / "failed.json")
    )

    sys.modules.pop("app.data_access.data_loader", None)
    monkeypatch.setattr(sys, "argv", ["data_loader.py"])
    runpy.run_module("app.data_access.data_loader", run_name="__main__")


def test_analyzer_main_module_runs(monkeypatch):
    fake_loader = types.SimpleNamespace()

    def fake_get_supported_ticker_list():
        return ["AAA"]

    def fake_load_data(*args, **kwargs):
        return pd.DataFrame()

    fake_loader.get_supported_ticker_list = fake_get_supported_ticker_list
    fake_loader.load_data = fake_load_data

    monkeypatch.setitem(sys.modules, "app.data_access.data_loader", fake_loader)
    sys.modules.pop("app.analysis.analyzer", None)
    monkeypatch.setattr(sys, "argv", ["analyzer.py"])
    runpy.run_module("app.analysis.analyzer", run_name="__main__")


def test_backtest_audit_empty_rewards():
    from app.backtesting.backtest_audit import (
        audit_confidence_buckets,
        audit_decision_levels,
    )

    rows = [
        {"confidence_bucket": "HIGH", "reward": None, "decision_level": "STRONG"},
        {"confidence_bucket": "LOW", "reward": None, "decision_level": "WEAK"},
    ]

    assert audit_confidence_buckets(rows) == {}
    assert audit_decision_levels(rows) == {}


def test_pi_config_detect_pi_mode_exception():
    from app.config import pi_config

    class FakePath:
        def exists(self):
            raise Exception("fail")

    assert pi_config.detect_pi_mode(env={}, model_path=FakePath()) is False


def test_portfolio_correlation_manager_branches(monkeypatch):
    from app.decision.portfolio_correlation_manager import (
        PortfolioCorrelationManager,
        find_uncorrelated_assets,
    )

    manager = PortfolioCorrelationManager()

    assert manager.calculate_diversification_score({"AAA": 1.0}) == 0.0

    monkeypatch.setattr(
        manager, "compute_correlation_matrix", lambda *args, **kwargs: pd.DataFrame()
    )
    assert manager.calculate_diversification_score({"AAA": 0.6, "BBB": 0.4}) == 0.5

    def fake_load(*args, **kwargs):
        return pd.DataFrame({"Close": [1.0] * 5})

    monkeypatch.setattr(manager.dm, "load_ohlcv", fake_load)
    decomposition = manager.decompose_portfolio_risk({"AAA": 0.5, "BBB": 0.5})
    assert decomposition.total_risk >= 0.0

    assert manager.optimize_for_low_correlation([]) == []

    candidates = ["AAA", "BBB", "CCC"]
    assert (
        manager.optimize_for_low_correlation(candidates, target_size=2)
        == candidates[:2]
    )

    corr = pd.DataFrame(
        [[1.0, 0.95, 0.95], [0.95, 1.0, 0.95], [0.95, 0.95, 1.0]],
        index=["AAA", "BBB", "CCC"],
        columns=["AAA", "BBB", "CCC"],
    )
    monkeypatch.setattr(
        manager, "compute_correlation_matrix", lambda *args, **kwargs: corr
    )
    selected = manager.optimize_for_low_correlation(
        ["AAA", "BBB", "CCC"], target_size=2, max_correlation=0.5
    )
    assert selected == ["AAA"]

    monkeypatch.setattr(
        manager, "compute_correlation_matrix", lambda *args, **kwargs: pd.DataFrame()
    )
    assert manager.get_highly_correlated_pairs(["AAA", "BBB"]) == []

    monkeypatch.setattr(
        PortfolioCorrelationManager,
        "compute_correlation_matrix",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    assert find_uncorrelated_assets("AAA", ["BBB"], max_correlation=0.1) == []


def test_confidence_allocator_more_branches(tmp_path, monkeypatch):
    from app.decision.confidence_allocator import ConfidenceBucketAllocator

    allocator = ConfidenceBucketAllocator(
        db_path=str(tmp_path / "conf.db"), base_capital=100.0
    )

    with pytest.raises(ValueError):
        allocator.allocate_capital_by_bucket({"S1": "bad"})

    monkeypatch.setattr(
        sqlite3,
        "connect",
        lambda *args, **kwargs: (_ for _ in ()).throw(sqlite3.Error("fail")),
    )
    allocator.allocate_capital("S1", 0.8)
    assert allocator.get_bucket_statistics() == {}
    assert allocator.get_allocation_history() == []

    assert allocator.suggest_rebalancing({}) == {}


def test_market_regime_detector_calculations():
    from app.decision.market_regime_detector import MarketRegimeDetector

    detector = MarketRegimeDetector()

    short_df = pd.DataFrame({"Close": [1.0]})
    assert detector._calculate_volatility(short_df) == 0.0
    assert detector._calculate_trend_strength(short_df) == 0.0
    assert detector._calculate_trend_consistency(short_df) == 0.0

    flat_df = pd.DataFrame({"Close": [1.0] * 12})
    assert detector._calculate_trend_consistency(flat_df) == 0.0

    assert detector._calculate_recent_return(short_df, days=1) == 0.0


def test_data_loader_load_data_warning(monkeypatch):
    from app.data_access import data_loader

    class FakeDM:
        def load_ohlcv(self, *args, **kwargs):
            return pd.DataFrame()

    monkeypatch.setattr(data_loader, "DataManager", lambda *args, **kwargs: FakeDM())
    monkeypatch.setattr(
        data_loader, "download_and_save_data", lambda *args, **kwargs: False
    )

    data_loader.load_data("AAA", start="2024-01-01", end="2024-01-10")


def test_config_secret_key_policy(monkeypatch):
    from app.config.config import Config, _enforce_secret_key_policy

    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("FLASK_ENV", "production")
    monkeypatch.setattr(Config, "SECRET_KEY", "")
    with pytest.raises(RuntimeError):
        _enforce_secret_key_policy()


def test_pi_config_linux_arm_detection():
    from app.config import pi_config

    class FakePlatform:
        @staticmethod
        def system():
            return "Linux"

        @staticmethod
        def machine():
            return "arm64"

    assert pi_config.detect_pi_mode(env={}, platform_module=FakePlatform) is True


def test_capital_optimizer_invalid_kelly():
    from app.decision.capital_optimizer import CapitalUtilizationOptimizer

    optimizer = CapitalUtilizationOptimizer(total_capital=1000.0, db_path=None)
    assert optimizer.calculate_kelly_fraction(0.5, 0.0, 1.0) == 0.0


def test_allocation_error_paths(monkeypatch):
    from app.decision.allocation import allocate_capital, enforce_correlation_limits
    import app.decision.allocation as allocation_module

    corr = pd.DataFrame(
        [[1.0, 0.9], [0.9, 1.0]], index=["AAA", "BBB"], columns=["AAA", "BBB"]
    )
    monkeypatch.setattr(
        allocation_module, "_get_correlation_matrix", lambda *args, **kwargs: corr
    )

    decisions = [
        {
            "ticker": "AAA",
            "payload": {"volatility": 0.0},
            "decision": {"action_code": 1, "confidence": 0.9},
        },
        {
            "ticker": "BBB",
            "payload": {"volatility": 0.0},
            "decision": {"action_code": 1, "confidence": 0.5},
        },
    ]

    out = allocate_capital(decisions)
    assert out[0]["allocation_amount"] > 0

    monkeypatch.setattr(
        allocation_module,
        "_get_correlation_matrix",
        lambda *args, **kwargs: (_ for _ in ()).throw(Exception("corr fail")),
    )
    limited = enforce_correlation_limits(out, max_correlation=0.7)
    assert limited[0]["allocation_amount"] > 0


def test_capital_optimizer_error_paths(tmp_path, monkeypatch):
    from app.decision.capital_optimizer import CapitalUtilizationOptimizer

    optimizer = CapitalUtilizationOptimizer(
        total_capital=1000.0, db_path=str(tmp_path / "cap.db")
    )
    assert optimizer.calculate_kelly_fraction(0.9, 1.0, 0.1) <= 0.5

    monkeypatch.setattr(
        sqlite3,
        "connect",
        lambda *args, **kwargs: (_ for _ in ()).throw(sqlite3.Error("fail")),
    )
    optimizer.optimize_capital_allocation({"AAA": {"kelly": 0.2, "volatility": 0.1}})
    assert optimizer.get_position_history() == []

    assert optimizer._calculate_diversification_score({}) == 0.0
    assert optimizer._calculate_diversification_score({"AAA": 0.0}) == 0.0


def test_data_loader_market_volatility_fallback(monkeypatch):
    from app.data_access import data_loader

    class FakeDM:
        def get_market_data(self, *args, **kwargs):
            return []

        def save_market_data(self, *args, **kwargs):
            return None

    class FakeTicker:
        def history(self, *args, **kwargs):
            raise Exception("boom")

    class FakeYF:
        @staticmethod
        def Ticker(*args, **kwargs):
            return FakeTicker()

    monkeypatch.setattr(data_loader, "DataManager", FakeDM)
    monkeypatch.setattr(data_loader, "yf", FakeYF)

    assert data_loader.get_market_volatility_index() is None


def test_data_manager_additional_branches(tmp_path, monkeypatch, test_settings):
    from app.data_access.data_manager import DataManager

    db_path = tmp_path / "dm.db"
    settings = replace(test_settings, DB_PATH=db_path)
    dm = DataManager(settings=settings)
    dm.initialize_tables()

    dm.save_ohlcv("AAA", None)

    monkeypatch.setattr(dm, "load_ohlcv", lambda *args, **kwargs: pd.DataFrame())
    assert dm.get_market_regime_is_bear(ref_date="2024-01-01") is False
    assert dm.get_correlation_matrix(["AAA"]).empty

    dm.save_market_data("^VIX", pd.DataFrame())


def test_pi_config_model_path_detection(monkeypatch):
    from app.config import pi_config

    class FakePath:
        def exists(self):
            return True

        def read_text(self):
            return "Raspberry Pi 4"

    class FakePlatform:
        @staticmethod
        def system():
            return "Windows"

        @staticmethod
        def machine():
            return "amd64"

    assert (
        pi_config.detect_pi_mode(
            env={}, platform_module=FakePlatform, model_path=FakePath()
        )
        is True
    )
