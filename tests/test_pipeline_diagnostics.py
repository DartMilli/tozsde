import json
import tempfile
from pathlib import Path

import pandas as pd

from app.analysis.analyzer import compute_signals
from app.backtesting.backtester import Backtester
from app.config.config import Config
from app.data_access.data_manager import DataManager
from app.validation.data_integrity_check import run_data_integrity_checks
from app.validation.pipeline_audit import _audit_fold
from app.validation.sanity_strategy import run_sanity_backtest
from app.validation.wf_analysis import run_walk_forward_analysis


def _make_df(rows: int = 200) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=rows, freq="D")
    close = []
    for idx in range(rows):
        if idx < rows // 3:
            close.append(100 - idx * 0.2)
        elif idx < 2 * rows // 3:
            close.append(80 + (idx - rows // 3) * 0.6)
        else:
            close.append(200 - (idx - 2 * rows // 3) * 0.5)
    close = pd.Series(close, index=dates)
    df = pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.001,
            "Low": close * 0.998,
            "Close": close,
            "Volume": 1000,
        },
        index=dates,
    )
    return df


def _params():
    return {
        "sma_period": 5,
        "ema_period": 3,
        "rsi_period": 14,
        "macd_fast": 5,
        "macd_slow": 10,
        "macd_signal": 3,
        "bbands_period": 10,
        "bbands_stddev": 2.0,
        "atr_period": 5,
        "adx_period": 5,
        "stoch_k": 5,
        "stoch_d": 3,
        "use_sma": True,
        "use_ema": True,
        "use_rsi": False,
        "use_macd": False,
        "use_bbands": False,
        "use_atr": False,
        "use_adx": False,
        "use_stoch": False,
    }


def test_signal_generates_trades_when_conditions_simple():
    df = _make_df(180)
    params = _params()
    signals, _ = compute_signals(df, "TEST", params, return_series=True)
    assert any(s in {"BUY", "SELL"} for s in signals)

    report = Backtester(df, "TEST").run(params, execution_policy="next_open")
    assert report.metrics.get("trade_count", 0) > 0


def test_sanity_strategy_executes_trades():
    df = _make_df(160)
    result = run_sanity_backtest(df, every_n=30, hold_bars=5)
    assert result["expected_trades"] > 0
    assert result["executed_trades"] > 0


def test_wf_latest_only_aggregation():
    original_db = Config.DB_PATH
    original_mode = Config.AGGREGATION_MODE
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_db = Path(tmp_dir) / "test.db"
            Config.DB_PATH = temp_db
            Config.AGGREGATION_MODE = "latest_only"
            dm = DataManager()
            dm.initialize_tables()

            older = {
                "best_params": {"sma_period": 5},
                "raw_fitness": 1.0,
                "wf_summary": {"avg_return": 0.1, "avg_drawdown": -0.1},
                "wf_run_id": "run-old",
            }
            newer = {
                "best_params": {"sma_period": 7},
                "raw_fitness": 2.0,
                "wf_summary": {"avg_return": 0.2, "avg_drawdown": -0.2},
                "wf_run_id": "run-new",
            }

            with dm.connection() as conn:
                conn.execute(
                    "INSERT INTO walk_forward_results (ticker, computed_at, result_json) VALUES (?, ?, ?)",
                    ("VOO", "2026-01-01T00:00:00Z", json.dumps(older)),
                )
                conn.execute(
                    "INSERT INTO walk_forward_results (ticker, computed_at, result_json) VALUES (?, ?, ?)",
                    ("VOO", "2026-02-01T00:00:00Z", json.dumps(newer)),
                )
                conn.commit()

            result = run_walk_forward_analysis()
            assert result["runs"] == 1
            assert result["mean_oos_sharpe"] == 0.0
    finally:
        Config.DB_PATH = original_db
        Config.AGGREGATION_MODE = original_mode


def test_pipeline_audit_output_structure():
    df = _make_df(120)
    params = _params()
    audit = _audit_fold(df, "TEST", params, fold_id=0)
    required_keys = {
        "fold_id",
        "raw_rows",
        "usable_rows",
        "nan_rows_removed",
        "lookback",
        "raw_signal_count",
        "post_dropout_signal_count",
        "post_edge_filter_signal_count",
        "position_attempts",
        "orders_created",
        "executed_trades",
        "capital_start",
        "capital_end",
        "size_zero_count",
        "rejected_orders",
        "slice_start",
        "slice_end",
    }
    assert required_keys.issubset(audit.keys())


def test_data_integrity_detects_duplicate_index():
    df = _make_df(50)
    duplicated = pd.concat([df.iloc[:5], df.iloc[:5], df.iloc[5:]])
    result = run_data_integrity_checks(duplicated, lookback=5)
    assert result["duplicate_index_count"] > 0
