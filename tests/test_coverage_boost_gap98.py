import json
import os
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest


def test_backtest_audit_helpers():
    from app.backtesting.backtest_audit import (
        audit_confidence_buckets,
        audit_decision_levels,
        detect_overconfidence,
    )

    rows = [
        {
            "confidence_bucket": "HIGH",
            "reward": 1.0,
            "decision_level": "STRONG",
            "timestamp": "t",
            "confidence": 0.8,
        },
        {
            "confidence_bucket": "HIGH",
            "reward": -1.0,
            "decision_level": "STRONG",
            "timestamp": "t2",
            "confidence": 0.9,
        },
        {
            "confidence_bucket": "LOW",
            "reward": None,
            "decision_level": "WEAK",
            "timestamp": "t3",
            "confidence": None,
        },
    ]

    buckets = audit_confidence_buckets(rows)
    levels = audit_decision_levels(rows)
    flags = detect_overconfidence(rows)

    assert "HIGH" in buckets
    assert "STRONG" in levels
    assert flags


def test_backtester_series_tolist(monkeypatch):
    from app.backtesting.backtester import Backtester
    import app.backtesting.backtester as backtester_module

    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "Open": [1, 1, 1, 1, 1],
            "High": [2, 2, 2, 2, 2],
            "Low": [1, 1, 1, 1, 1],
            "Close": [1, 1, 1, 1, 1],
        },
        index=dates,
    )

    signals = ["BUY"] + ["HOLD"] * (len(df) - 1)
    monkeypatch.setattr(
        backtester_module,
        "compute_signals",
        lambda *args, **kwargs: (pd.Series(signals), {}),
    )

    bt = Backtester(df, "AAA")
    params = {
        "sma_period": 1,
        "ema_period": 1,
        "rsi_period": 1,
        "macd_slow": 1,
        "macd_signal": 1,
        "bbands_period": 1,
        "atr_period": 1,
        "adx_period": 1,
        "stoch_k": 1,
        "stoch_d": 1,
    }
    report = bt.run(params)
    assert report.metrics["trade_count"] == 0


def test_history_store_save_decision():
    from app.backtesting.history_store import HistoryStore

    class FakeDM:
        def __init__(self):
            self.saved = []

        def save_history_record(self, **kwargs):
            self.saved.append(kwargs)

    hs = HistoryStore()
    hs.dm = FakeDM()

    hs.save_decision(
        payload={"ticker": "AAA"},
        decision={
            "action_code": 1,
            "action": "BUY",
            "confidence": 0.8,
            "wf_score": 0.7,
        },
        explanation={"hu": "ok"},
        audit={"x": 1},
    )

    assert hs.dm.saved


def test_config_tickers_and_secret_policy(monkeypatch):
    from app.config.config import Config, _enforce_secret_key_policy

    monkeypatch.setattr(Config, "_TICKERS", None)
    monkeypatch.setattr(Config, "_SUPPORTED_TICKERS_LOADED", False)
    assert Config().TICKERS

    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("FLASK_ENV", "production")
    monkeypatch.setattr(Config, "SECRET_KEY", "dev_key_do_not_use_in_prod")
    with pytest.raises(RuntimeError):
        _enforce_secret_key_policy()


def test_pi_config_branches(monkeypatch, tmp_path):
    from app.config import pi_config

    class FakePlatform:
        @staticmethod
        def system():
            return "Linux"

        @staticmethod
        def machine():
            return "armv7l"

    assert pi_config.detect_pi_mode(env={}, platform_module=FakePlatform) is True

    monkeypatch.setattr(
        Path, "mkdir", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("fail"))
    )
    result = pi_config.apply_pi_config(
        config_cls=type("C", (), {"OPTIMIZER_POPULATION": 50}),
        env={"PI_MODE": "true", "PI_BASE_DIR": str(tmp_path)},
        ensure_dirs=True,
    )
    assert result["applied"] is True


def test_data_loader_and_manager_branches(monkeypatch, tmp_path):
    from app.data_access import data_loader
    from app.data_access.data_manager import DataManager
    from app.config.config import Config

    class FakeDM:
        def save_ohlcv(self, ticker, df):
            self.saved = True

    class FakeYF:
        @staticmethod
        def download(*args, **kwargs):
            return pd.DataFrame()

    monkeypatch.setattr(data_loader, "DataManager", FakeDM)
    monkeypatch.setattr(data_loader, "yf", FakeYF)
    assert (
        data_loader.download_and_save_data("AAA", date.today(), date.today()) is False
    )

    db_path = tmp_path / "dm.db"
    monkeypatch.setattr(Config, "DB_PATH", db_path)
    dm = DataManager()
    dm.initialize_tables()

    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": [1, 1, 1, 1, 1],
            "High": [2, 2, 2, 2, 2],
            "Low": [1, 1, 1, 1, 1],
            "Close": [1, 1, 1, 1, 1],
            "Volume": [1, 1, 1, 1, 1],
            "Ticker": ["AAA"] * 5,
        }
    ).set_index("Date")

    dm.save_ohlcv("AAA", df)
    loaded = dm.load_ohlcv("AAA", start_date=date(2024, 1, 1))
    assert not loaded.empty

    dm.save_market_data(
        "^VIX", pd.DataFrame({"Date": dates[:1], "Close": [1.0]}).set_index("Date")
    )
    assert dm.get_market_data("^VIX", days=1)

    assert dm.get_market_regime_is_bear(ref_date=date(2024, 1, 1)) is False
    assert dm.get_correlation_matrix(["ZZZ"]).empty


def test_allocation_and_limits(monkeypatch):
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
            "payload": {"volatility": 0},
            "decision": {"action_code": 1, "confidence": 0.9},
        },
        {
            "ticker": "BBB",
            "payload": {"volatility": 0},
            "decision": {"action_code": 1, "confidence": 0.5},
        },
    ]

    out = allocate_capital(decisions)
    assert out[0]["allocation_amount"] > 0
    assert out[1]["decision"].get("allocation_note") == "Correlation adjusted"

    limited = enforce_correlation_limits(out, max_correlation=0.7)
    assert limited[1]["decision"].get("correlation_adjustment") is True


def test_confidence_allocator_branches(tmp_path):
    from app.decision.confidence_allocator import (
        ConfidenceBucketAllocator,
        ConfidenceBucket,
    )

    db_path = tmp_path / "conf.db"
    allocator = ConfidenceBucketAllocator(db_path=str(db_path), base_capital=100.0)

    with pytest.raises(ValueError):
        allocator.classify_confidence_bucket(1.5)

    allocation = allocator.allocate_capital("S1", 0.8)
    assert allocation.bucket == ConfidenceBucket.STRONG

    stats = allocator.get_bucket_statistics()
    assert stats

    history = allocator.get_allocation_history()
    assert history

    history_one = allocator.get_allocation_history("S1")
    assert history_one

    suggestions = allocator.suggest_rebalancing({"S1": 200.0, "S2": 100.0})
    assert suggestions


def test_capital_optimizer_branches(tmp_path, monkeypatch):
    from app.decision.capital_optimizer import CapitalUtilizationOptimizer
    import sqlite3 as sqlite

    optimizer = CapitalUtilizationOptimizer(
        total_capital=1000.0, db_path=str(tmp_path / "cap.db")
    )
    assert optimizer.calculate_kelly_fraction(0.5, 1.0, 0.0) == 0.0

    position = optimizer.calculate_optimal_position_size(
        "AAA", kelly_fraction=0.2, volatility=0.1
    )
    assert position.ticker == "AAA"

    assert optimizer.get_position_history("AAA")
    assert optimizer.get_position_history()

    monkeypatch.setattr(
        sqlite,
        "connect",
        lambda *args, **kwargs: (_ for _ in ()).throw(sqlite.Error("fail")),
    )
    assert optimizer.get_position_history("AAA") == []

    optimizer_zero = CapitalUtilizationOptimizer(total_capital=0.0, db_path=None)
    assert optimizer_zero.estimate_max_drawdown({"AAA": 1.0}, volatility_avg=0.2) == 0.0


def test_decision_history_analyzer_branches(tmp_path, monkeypatch):
    from app.decision.decision_history_analyzer import DecisionHistoryAnalyzer
    from app.config.config import Config

    db_path = tmp_path / "history.db"
    monkeypatch.setattr(Config, "DB_PATH", db_path)

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
        outcome = {"pnl_pct": 0.1 if i % 2 == 0 else -0.05}
        audit = {
            "strategy": "MA_CROSS",
            "outcome": outcome,
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

    analyzer = DecisionHistoryAnalyzer()
    stats = analyzer.analyze_strategy_performance("MA_CROSS", days=90)
    assert stats.total_trades >= 1

    reliability = analyzer.analyze_ticker_reliability("AAA", days=90)
    assert reliability.total_decisions >= 1

    rolling = analyzer.compute_rolling_metrics(window_days=5, total_days=90)
    assert isinstance(rolling, pd.DataFrame)

    assert analyzer.get_best_strategy(days=90, min_trades=1) is not None
    assert analyzer.get_worst_strategy(days=90, min_trades=1) is not None


def test_decision_logger_details(tmp_path):
    from app.infrastructure.decision_logger import NoTradeDecisionLogger, NoTradeReason

    db_path = tmp_path / "no_trade.db"
    logger = NoTradeDecisionLogger(db_path=str(db_path))

    logger.log_no_trade_simple(
        "AAA", "S1", NoTradeReason.LOW_CONFIDENCE, confidence_score=0.2
    )
    logger.log_no_trade_simple(
        "AAA", "S1", NoTradeReason.HIGH_CORRELATION, confidence_score=0.7
    )

    analysis = logger.get_no_trade_analysis(days_back=7)
    assert analysis["total_no_trades"] >= 2


def test_health_check_parsing_variants(monkeypatch, tmp_path):
    from app.infrastructure import health_check
    from app.config.config import Config

    monkeypatch.setattr(Config, "LOG_DIR", tmp_path)
    checker = health_check.HealthChecker()

    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"event": "pipeline_execution", "timestamp": "2024-01-01T00:00:00"})
        + "\n"
    )

    cron = checker.check_cron_execution()
    assert "hours_ago" in cron


def test_logger_levels(monkeypatch, tmp_path):
    from app.infrastructure.logger import setup_logger
    from app.config.config import Config

    monkeypatch.setattr(Config, "LOG_DIR", tmp_path)
    monkeypatch.setattr(Config, "LOGGING_LEVEL", "WARNING")
    logger = setup_logger("lvl_warning")
    assert logger.level

    monkeypatch.setattr(Config, "LOGGING_LEVEL", "ERROR")
    logger2 = setup_logger("lvl_error")
    assert logger2.level
