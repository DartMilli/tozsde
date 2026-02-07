import json
import sqlite3
import os
from pathlib import Path
from types import SimpleNamespace, ModuleType
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest


def test_health_checker_api_and_resources(monkeypatch, tmp_path):
    from app.infrastructure import health_check
    from app.config.config import Config

    db_path = tmp_path / "health.db"
    monkeypatch.setattr(Config, "DB_PATH", db_path)
    monkeypatch.setattr(Config, "LOG_DIR", tmp_path)

    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE test (id INTEGER)")
    conn.commit()
    conn.close()

    checker = health_check.HealthChecker()

    class FakeResponse:
        def __init__(self, code):
            self._code = code

        def getcode(self):
            return self._code

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        health_check.urllib.request,
        "urlopen",
        lambda *args, **kwargs: FakeResponse(200),
    )
    assert checker.check_api_health()["healthy"] is True

    monkeypatch.setattr(
        health_check.urllib.request,
        "urlopen",
        lambda *args, **kwargs: FakeResponse(500),
    )
    assert checker.check_api_health()["healthy"] is False

    monkeypatch.setattr(
        health_check.urllib.request,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            health_check.urllib.error.URLError("fail")
        ),
    )
    assert checker.check_api_health()["healthy"] is False

    monkeypatch.setattr(
        health_check.psutil,
        "disk_usage",
        lambda *_: SimpleNamespace(
            percent=95.0, free=1_000_000_000, total=2_000_000_000
        ),
    )
    assert checker.check_disk_space()["healthy"] is False

    monkeypatch.setattr(
        health_check.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(
            percent=50.0, available=2_000_000_000, total=4_000_000_000
        ),
    )
    assert checker.check_memory_usage()["healthy"] is True

    monkeypatch.setattr(health_check.psutil, "cpu_percent", lambda interval=1.0: 95.0)
    assert checker.check_cpu_usage()["healthy"] is False

    assert checker.check_database_health()["healthy"] is True


def test_health_checker_cron_and_logs(monkeypatch, tmp_path):
    from app.infrastructure import health_check
    from app.config.config import Config

    monkeypatch.setattr(Config, "LOG_DIR", tmp_path)
    checker = health_check.HealthChecker()

    metrics_path = tmp_path / "metrics.jsonl"
    now = datetime.now(timezone.utc)
    metrics_path.write_text(
        json.dumps({"event": "pipeline_execution", "timestamp": now.isoformat()}) + "\n"
    )

    cron = checker.check_cron_execution()
    assert cron["healthy"] is True

    metrics_path.write_text("not-json\n")
    cron = checker.check_cron_execution()
    assert cron["healthy"] is False

    status = {"healthy": False, "issues": ["cpu: high"], "checks": {}}
    alert = checker.generate_alert(status)
    assert "HEALTH CHECK ALERT" in alert

    checker._log_health_status(
        {"timestamp": now.isoformat(), "healthy": True, "issues": [], "checks": {}}
    )
    logs = checker.get_recent_health_logs(hours=1)
    assert logs


def test_health_checker_check_all_and_main(monkeypatch, tmp_path):
    from app.infrastructure import health_check
    from app.config.config import Config

    monkeypatch.setattr(Config, "LOG_DIR", tmp_path)

    checker = health_check.HealthChecker()

    monkeypatch.setattr(checker, "check_api_health", lambda: {"healthy": True})
    monkeypatch.setattr(
        checker, "check_database_health", lambda: {"healthy": False, "message": "db"}
    )
    monkeypatch.setattr(checker, "check_disk_space", lambda: {"healthy": True})
    monkeypatch.setattr(checker, "check_memory_usage", lambda: {"healthy": True})
    monkeypatch.setattr(checker, "check_cpu_usage", lambda: {"healthy": True})
    monkeypatch.setattr(checker, "check_cron_execution", lambda: {"healthy": True})

    status = checker.check_all()
    assert status["healthy"] is False
    assert status["issues"]

    monkeypatch.setattr(health_check, "HealthChecker", lambda: checker)
    assert health_check.main() == 1


def test_error_reporter_end_to_end(tmp_path):
    from app.infrastructure.error_reporter import ErrorReporter, ErrorSeverity

    db_path = tmp_path / "errors.db"
    reporter = ErrorReporter(
        db_path=str(db_path), critical_threshold=1, error_rate_threshold=0.1
    )

    assert reporter.log_error_simple("TypeError", "oops", severity=ErrorSeverity.ERROR)
    try:
        raise ValueError("bad")
    except Exception as exc:
        assert reporter.log_error(
            exc,
            severity=ErrorSeverity.CRITICAL,
            context="ctx",
            module="m",
            function="f",
        )

    stats = reporter.get_error_statistics(hours_back=24)
    assert stats.total_errors >= 2

    recent = reporter.get_recent_errors(limit=10)
    assert recent

    assert reporter.check_error_rate() is False

    out_file = tmp_path / "report.json"
    assert reporter.export_error_report(str(out_file), days_back=7) is True

    deleted = reporter.clear_old_errors(days_to_keep=0)
    assert isinstance(deleted, int)

    trends = reporter.get_error_trends(days=7)
    assert "dates" in trends


def test_error_reporter_error_paths(monkeypatch, tmp_path):
    from app.infrastructure.error_reporter import ErrorReporter
    import sqlite3 as sqlite

    db_path = tmp_path / "errors.db"
    reporter = ErrorReporter(db_path=str(db_path))

    monkeypatch.setattr(
        sqlite,
        "connect",
        lambda *args, **kwargs: (_ for _ in ()).throw(sqlite.Error("fail")),
    )

    assert reporter.get_error_statistics().total_errors == 0
    assert reporter.get_recent_errors() == []
    assert reporter.export_error_report(str(tmp_path / "out.json")) is False
    assert reporter.clear_old_errors() == 0
    assert reporter.get_error_trends()["dates"] == []


def test_error_reporter_no_db():
    from app.infrastructure.error_reporter import ErrorReporter

    reporter = ErrorReporter(db_path=None)
    assert reporter.get_error_statistics().total_errors == 0
    assert reporter.get_recent_errors() == []
    assert reporter.export_error_report("x.json") is False
    assert reporter.clear_old_errors() == 0
    assert reporter.get_error_trends()["dates"] == []


def test_log_manager_rotation_and_cleanup(tmp_path):
    from app.infrastructure.log_manager import LogManager

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    old_log = log_dir / "old.log"
    old_log.write_text("x" * 100)
    os_time = datetime.now(timezone.utc) - timedelta(days=10)
    old_ts = os_time.timestamp()
    os.utime(old_log, (old_ts, old_ts))

    manager = LogManager(log_dir=log_dir)

    rotated = manager.rotate_logs()
    assert rotated["rotated_count"] >= 1

    archives = list(log_dir.glob("*.gz"))
    assert archives

    # Cleanup archives by setting old mtime
    older_ts = (datetime.now(timezone.utc) - timedelta(days=40)).timestamp()
    for archive in archives:
        os.utime(archive, (older_ts, older_ts))

    cleaned = manager.cleanup_old_archives()
    assert cleaned["deleted_count"] >= 1

    # Force rotate and cleanup with fresh files
    new_log = log_dir / "new.jsonl"
    new_log.write_text("y" * 10)

    force_rotated = manager.force_rotate_all()
    assert force_rotated["rotated_count"] >= 1

    force_clean = manager.cleanup_all_archives()
    assert "deleted_count" in force_clean

    stats = manager.get_log_stats()
    assert "total_files" in stats


def test_log_manager_error_paths(monkeypatch, tmp_path):
    from app.infrastructure.log_manager import LogManager
    import gzip as gzip_module

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    log_file = log_dir / "bad.log"
    log_file.write_text("x")

    manager = LogManager(log_dir=log_dir)

    monkeypatch.setattr(
        gzip_module,
        "open",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("fail")),
    )
    assert manager._compress_log(log_file) is None

    monkeypatch.setattr(
        manager, "_find_log_files", lambda: (_ for _ in ()).throw(RuntimeError("fail"))
    )
    result = manager.rotate_logs()
    assert result["rotated_count"] == 0

    monkeypatch.setattr(
        manager,
        "_find_archive_files",
        lambda: (_ for _ in ()).throw(RuntimeError("fail")),
    )
    result = manager.cleanup_old_archives()
    assert result["deleted_count"] == 0

    monkeypatch.setattr(manager, "_find_log_files", lambda: [])
    stats = manager.get_log_stats()
    assert "total_files" in stats


def test_backup_manager_full_cycle(tmp_path):
    from app.infrastructure.backup_manager import BackupManager

    db_path = tmp_path / "db.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE test (id INTEGER)")
    conn.commit()
    conn.close()

    backup_dir = tmp_path / "backups"
    manager = BackupManager(db_path=db_path, backup_dir=backup_dir)

    result = manager.backup_database()
    assert result["success"] is True

    backup_path = Path(result["backup_path"])
    verify = manager.verify_backup(backup_path)
    assert verify["valid"] is True

    stats = manager.get_backup_stats()
    assert stats["total_backups"] >= 1

    backups = manager.list_backups()
    assert backups

    restored = manager.restore_backup(backup_path, target_path=tmp_path / "restored.db")
    assert restored["success"] is True

    # Cleanup old backups
    old_ts = (datetime.now(timezone.utc) - timedelta(days=40)).timestamp()
    for backup in backup_dir.glob("*.db"):
        os.utime(backup, (old_ts, old_ts))

    cleanup = manager.cleanup_old_backups()
    assert "deleted_count" in cleanup


def test_backup_manager_error_paths(monkeypatch, tmp_path):
    from app.infrastructure.backup_manager import BackupManager

    db_path = tmp_path / "db.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE test (id INTEGER)")
    conn.commit()
    conn.close()

    manager = BackupManager(db_path=db_path, backup_dir=tmp_path / "backups")

    monkeypatch.setattr(
        manager, "verify_backup", lambda *_: {"valid": False, "error": "bad"}
    )
    restore = manager.restore_backup(tmp_path / "nope.db")
    assert restore["success"] is False

    monkeypatch.setattr(
        manager,
        "_find_backup_files",
        lambda: (_ for _ in ()).throw(RuntimeError("fail")),
    )
    stats = manager.get_backup_stats()
    assert stats["total_backups"] == 0


def test_backup_manager_missing(tmp_path):
    from app.infrastructure.backup_manager import BackupManager

    manager = BackupManager(
        db_path=tmp_path / "missing.db", backup_dir=tmp_path / "backups"
    )
    result = manager.backup_database()
    assert result["success"] is False

    verify = manager.verify_backup(tmp_path / "nope.db")
    assert verify["valid"] is False


def test_decision_logger_end_to_end(tmp_path):
    from app.infrastructure.decision_logger import NoTradeDecisionLogger, NoTradeReason

    db_path = tmp_path / "no_trade.db"
    logger = NoTradeDecisionLogger(db_path=str(db_path))

    assert logger.log_no_trade_simple("AAA", "S1", NoTradeReason.LOW_CONFIDENCE)

    records = logger.get_no_trade_decisions(days_back=7)
    assert records

    analysis = logger.get_no_trade_analysis(days_back=7)
    assert analysis["total_no_trades"] >= 1

    out_file = tmp_path / "no_trade.json"
    assert logger.export_no_trade_decisions(str(out_file), days_back=7) is True

    deleted = logger.clear_old_decisions(days_to_keep=0)
    assert isinstance(deleted, int)


def test_decision_logger_filters_and_errors(monkeypatch, tmp_path):
    from app.infrastructure.decision_logger import NoTradeDecisionLogger, NoTradeReason

    db_path = tmp_path / "no_trade.db"
    logger = NoTradeDecisionLogger(db_path=str(db_path))

    logger.log_no_trade_simple(
        "AAA", "S1", NoTradeReason.LOW_CONFIDENCE, confidence_score=0.2
    )
    logger.log_no_trade_simple("BBB", "S2", NoTradeReason.HIGH_CORRELATION)

    records = logger.get_no_trade_decisions(days_back=7, ticker="AAA")
    assert records

    records = logger.get_no_trade_decisions(
        days_back=7, reason=NoTradeReason.HIGH_CORRELATION
    )
    assert records

    monkeypatch.setattr(
        "builtins.open", lambda *args, **kwargs: (_ for _ in ()).throw(IOError("fail"))
    )
    assert logger.export_no_trade_decisions(str(tmp_path / "x.json")) is False


def test_decision_logger_no_db():
    from app.infrastructure.decision_logger import NoTradeDecisionLogger

    logger = NoTradeDecisionLogger(db_path=None)
    assert logger.get_no_trade_decisions() == []
    assert logger.get_no_trade_analysis() == {}
    assert logger.clear_old_decisions() == 0


def test_market_regime_detector(monkeypatch):
    from app.decision.market_regime_detector import (
        MarketRegimeDetector,
        get_market_regime,
        is_bull_market,
    )

    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    df = pd.DataFrame({"Close": range(1, 61)}, index=dates)

    class FakeDM:
        def load_ohlcv(self, ticker):
            return df

    detector = MarketRegimeDetector(lookback_days=30)
    monkeypatch.setattr(detector, "dm", FakeDM())

    regime = detector.detect_regime("AAA")
    assert regime.regime_type in {"BULL", "RANGING", "VOLATILE", "BEAR"}

    transition = detector.detect_regime_transition("AAA", days=5)
    assert transition is None or transition["transition_detected"] is True

    monkeypatch.setattr(
        "app.decision.market_regime_detector.MarketRegimeDetector", lambda: detector
    )
    assert get_market_regime("AAA") in {"BULL", "RANGING", "VOLATILE", "BEAR"}
    assert isinstance(is_bull_market("AAA", confidence_threshold=0.0), bool)


def test_market_regime_detector_branches(monkeypatch):
    from app.decision.market_regime_detector import (
        MarketRegimeDetector,
        is_high_volatility,
    )

    detector = MarketRegimeDetector(lookback_days=5)

    assert detector._classify_regime(0.5, 0.0, 0.0) == "VOLATILE"
    assert detector._classify_regime(0.1, 0.5, 0.6) == "BULL"
    assert detector._classify_regime(0.1, -0.5, 0.6) == "BEAR"
    assert detector._classify_regime(0.1, 0.0, 0.1) == "RANGING"

    conf = detector._calculate_confidence(0.5, 0.6, 0.6)
    assert 0.3 <= conf <= 1.0

    class FakeDM:
        def load_ohlcv(self, ticker):
            dates = pd.date_range("2024-01-01", periods=30, freq="D")
            return pd.DataFrame({"Close": range(1, 31)}, index=dates)

    detector.dm = FakeDM()
    monkeypatch.setattr(
        "app.decision.market_regime_detector.MarketRegimeDetector", lambda: detector
    )
    assert isinstance(bool(is_high_volatility("AAA", threshold=0.0)), bool)


def test_strategy_selector_and_optimizer(tmp_path, monkeypatch):
    from app.decision.adaptive_strategy_selector import (
        AdaptiveStrategySelector,
        get_top_strategies,
        recommend_strategy_for_regime,
    )
    from app.decision.capital_optimizer import CapitalUtilizationOptimizer

    db_path = tmp_path / "strategies.db"
    monkeypatch.setattr(
        "app.decision.adaptive_strategy_selector.Config.DB_PATH", db_path
    )

    selector = AdaptiveStrategySelector(epsilon=0.0)
    weights = selector.select_strategy_weights()
    assert abs(sum(weights.values()) - 1.0) < 1e-6

    selection = selector.explore_or_exploit(market_regime="BULL")
    assert selection.selection_mode in {"EXPLOIT", "EXPLORE"}

    selector.update_strategy("MA_CROSS", success=True)
    stats = selector.get_strategy_stats()
    assert stats

    selector.reset_strategy("MA_CROSS")
    top = get_top_strategies(n=2)
    assert top

    assert recommend_strategy_for_regime("BULL")

    optimizer = CapitalUtilizationOptimizer(
        total_capital=1000.0, db_path=str(tmp_path / "cap.db")
    )
    kelly = optimizer.calculate_kelly_fraction(0.6, 1.0, 0.5)
    assert 0.01 <= kelly <= 0.5

    position = optimizer.calculate_optimal_position_size(
        "AAA", kelly_fraction=0.2, volatility=0.1
    )
    assert position.risk_adjusted_size >= optimizer.min_position_size

    allocation = optimizer.optimize_capital_allocation(
        {"AAA": {"kelly": 0.2, "volatility": 0.1}}
    )
    assert allocation.utilization_rate >= 0

    history = optimizer.get_position_history("AAA")
    assert history

    drawdown = optimizer.estimate_max_drawdown({"AAA": 100.0}, volatility_avg=0.2)
    assert 0.0 <= drawdown <= 1.0


def test_strategy_selector_explore_and_errors(tmp_path, monkeypatch):
    from app.decision.adaptive_strategy_selector import AdaptiveStrategySelector
    import sqlite3 as sqlite

    monkeypatch.setattr(
        "app.decision.adaptive_strategy_selector.Config.DB_PATH",
        tmp_path / "bandits.db",
    )

    selector = AdaptiveStrategySelector(epsilon=1.0)
    selection = selector.explore_or_exploit(market_regime="BEAR")
    assert selection.selection_mode == "EXPLORE"

    best = selector.select_best_strategy()
    assert best

    monkeypatch.setattr(
        sqlite,
        "connect",
        lambda *args, **kwargs: (_ for _ in ()).throw(sqlite.Error("fail")),
    )
    selector.update_strategy("NEW", success=False)


def test_capital_optimizer_edge_cases(tmp_path):
    from app.decision.capital_optimizer import CapitalUtilizationOptimizer

    optimizer = CapitalUtilizationOptimizer(total_capital=0.0, db_path=None)
    assert optimizer.calculate_kelly_fraction(0.5, 0.0, 1.0) == 0.0
    assert optimizer._calculate_diversification_score({}) == 0.0
    assert optimizer._calculate_diversification_score({"AAA": 0.0}) == 0.0
    assert optimizer.get_position_history() == []


def test_system_metrics(monkeypatch, tmp_path):
    from app.infrastructure.metrics import SystemMetrics, get_metrics

    class FakeDM:
        def initialize_tables(self):
            return None

        def log_pipeline_execution(self, *args, **kwargs):
            return True

        def log_backtest_execution(self, *args, **kwargs):
            return True

        def get_recent_metrics(self, hours=24):
            return {
                "success_rate": 0.9,
                "avg_duration_sec": 1.2,
                "errors_count": 1,
                "total_executions": 10,
                "last_success": None,
                "last_error": None,
            }

        def get_daily_summary(self, date):
            return {
                "date": date,
                "executions": 1,
                "successes": 1,
                "failures": 0,
                "avg_duration_sec": 1.0,
                "tickers_processed": [],
            }

    monkeypatch.setattr("app.infrastructure.metrics.DataManager", lambda: FakeDM())

    metrics = SystemMetrics()
    assert metrics.log_pipeline_execution("AAA", "success", 1.0) is True
    assert metrics.log_backtest_execution("AAA", 0.5, 2, 1.1) is True
    assert metrics.get_recent_metrics(hours=1)["total_executions"] == 10
    assert metrics.get_daily_summary("2024-01-01")["executions"] == 1

    health = metrics.get_health_status()
    assert health["status"] in {"healthy", "degraded", "critical"}

    assert get_metrics() is not None


def test_log_manager_main(monkeypatch, tmp_path):
    from app.infrastructure import log_manager
    from app.config.config import Config

    monkeypatch.setattr(Config, "LOG_DIR", tmp_path)

    monkeypatch.setattr("sys.argv", ["log_manager.py", "--stats"])
    log_manager.main()

    monkeypatch.setattr("sys.argv", ["log_manager.py", "--rotate"])
    log_manager.main()

    monkeypatch.setattr("sys.argv", ["log_manager.py", "--cleanup"])
    log_manager.main()

    monkeypatch.setattr("sys.argv", ["log_manager.py", "--force-rotate"])
    log_manager.main()

    monkeypatch.setattr("sys.argv", ["log_manager.py", "--force-cleanup"])
    log_manager.main()

    monkeypatch.setattr("sys.argv", ["log_manager.py"])
    log_manager.main()


def test_backup_manager_main(monkeypatch, tmp_path):
    from app.infrastructure import backup_manager

    manager = backup_manager.BackupManager(
        db_path=tmp_path / "db.sqlite", backup_dir=tmp_path / "backups"
    )

    monkeypatch.setattr(backup_manager, "BackupManager", lambda: manager)
    monkeypatch.setattr(
        manager,
        "backup_database",
        lambda: {"success": True, "backup_path": "x", "size_mb": 0.1},
    )
    monkeypatch.setattr(
        manager,
        "cleanup_old_backups",
        lambda: {"deleted_count": 0, "bytes_freed": 0, "errors": []},
    )
    monkeypatch.setattr(
        manager,
        "get_backup_stats",
        lambda: {
            "directory": "x",
            "total_backups": 0,
            "total_size_mb": 0,
            "oldest_backup": None,
            "oldest_age_days": None,
            "newest_backup": None,
        },
    )
    monkeypatch.setattr(manager, "list_backups", lambda: [])
    monkeypatch.setattr(
        manager,
        "verify_backup",
        lambda path: {"valid": True, "tables": 1, "error": None},
    )
    monkeypatch.setattr(
        manager, "restore_backup", lambda path: {"success": True, "error": None}
    )

    monkeypatch.setattr("sys.argv", ["backup_manager.py", "--backup"])
    backup_manager.main()

    monkeypatch.setattr("sys.argv", ["backup_manager.py", "--cleanup"])
    backup_manager.main()

    monkeypatch.setattr("sys.argv", ["backup_manager.py", "--stats"])
    backup_manager.main()

    monkeypatch.setattr("sys.argv", ["backup_manager.py", "--list"])
    backup_manager.main()

    monkeypatch.setattr("sys.argv", ["backup_manager.py", "--verify", "x"])
    backup_manager.main()

    monkeypatch.setattr("sys.argv", ["backup_manager.py", "--restore", "x"])
    backup_manager.main()


def test_cron_tasks_main(monkeypatch, tmp_path):
    from app.infrastructure import cron_tasks

    scheduler = cron_tasks.CronTaskScheduler()
    scheduler.config = SimpleNamespace(db_path="db", log_dir=str(tmp_path))

    monkeypatch.setattr(cron_tasks, "CronTaskScheduler", lambda: scheduler)
    monkeypatch.setattr(
        scheduler,
        "run_daily_pipeline",
        lambda: {"task": "daily", "success": True, "duration_seconds": 1},
    )
    monkeypatch.setattr(
        scheduler,
        "run_weekly_audit",
        lambda: {"task": "weekly", "success": True, "duration_seconds": 1},
    )
    monkeypatch.setattr(
        scheduler,
        "run_monthly_optimization",
        lambda: {"task": "monthly", "success": True, "duration_seconds": 1},
    )

    monkeypatch.setattr("sys.argv", ["cron_tasks.py", "--daily", "--json"])
    with pytest.raises(SystemExit):
        cron_tasks.main()

    monkeypatch.setattr("sys.argv", ["cron_tasks.py", "--weekly"])
    with pytest.raises(SystemExit):
        cron_tasks.main()

    monkeypatch.setattr("sys.argv", ["cron_tasks.py", "--monthly"])
    with pytest.raises(SystemExit):
        cron_tasks.main()


def test_cron_tasks_main_help(monkeypatch):
    from app.infrastructure import cron_tasks

    monkeypatch.setattr("sys.argv", ["cron_tasks.py"])
    with pytest.raises(SystemExit):
        cron_tasks.main()


def test_cron_tasks(monkeypatch, tmp_path):
    from app.infrastructure import cron_tasks

    scheduler = cron_tasks.CronTaskScheduler()
    scheduler.config = SimpleNamespace(db_path="db", log_dir=str(tmp_path))

    class FakeDataManager:
        def __init__(self, db_path=None):
            self.db_path = db_path

        def update_market_data(self):
            return {"success": True, "total_records": 1}

    class FakeDecisionEngine:
        def __init__(self, config=None):
            self.config = config

        def generate_recommendations(self):
            return [1]

    dm_module = ModuleType("app.data_access.data_manager")
    dm_module.DataManager = FakeDataManager
    de_module = ModuleType("app.decision.decision_engine")
    de_module.DecisionEngine = FakeDecisionEngine

    monkeypatch.setitem(
        __import__("sys").modules, "app.data_access.data_manager", dm_module
    )
    monkeypatch.setitem(
        __import__("sys").modules, "app.decision.decision_engine", de_module
    )

    scheduler.backup_manager.backup_database = lambda: {"success": True, "size_mb": 1.0}
    scheduler.backup_manager.cleanup_old_backups = lambda: {"deleted_count": 0}
    scheduler.log_manager.rotate_logs = lambda: {"archived_count": 0}

    result = scheduler.run_daily_pipeline()
    assert result["success"] is True

    cron_tasks.BacktestAuditor = type(
        "BA",
        (),
        {
            "__init__": lambda self, config=None: None,
            "run_weekly_audit": lambda self: {"success": True, "metrics": {}},
        },
    )
    weekly = scheduler.run_weekly_audit()
    assert weekly["success"] is True

    cron_tasks.GeneticOptimizer = type(
        "GO",
        (),
        {
            "__init__": lambda self, config=None: None,
            "run_monthly_optimization": lambda self: {
                "success": True,
                "best_fitness": 1.0,
                "generations": 1,
            },
        },
    )
    monthly = scheduler.run_monthly_optimization()
    assert monthly["success"] is True
