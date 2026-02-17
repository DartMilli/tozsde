"""Sanity strategy to validate execution pipeline integrity."""

from __future__ import annotations

from app.backtesting.backtester import Backtester
from app.config.config import Config
from app.infrastructure.logger import setup_logger
from app.validation.errors import ExecutionPipelineError

logger = setup_logger(__name__)


def run_sanity_backtest(df, every_n: int = 50, hold_bars: int = 10) -> dict:
    if df is None or df.empty:
        return {
            "expected_trades": 0,
            "executed_trades": 0,
            "capital_start": float(Config.INITIAL_CAPITAL),
            "capital_end": float(Config.INITIAL_CAPITAL),
        }

    signals = ["HOLD"] * len(df)
    expected_trades = 0
    idx = 0
    while idx + hold_bars < len(df):
        signals[idx] = "BUY"
        exit_idx = idx + hold_bars
        if exit_idx < len(signals):
            signals[exit_idx] = "SELL"
            expected_trades += 1
        idx += every_n

    params = {
        "sma_period": 1,
        "ema_period": 1,
        "rsi_period": 1,
        "macd_fast": 1,
        "macd_slow": 2,
        "macd_signal": 1,
        "bbands_period": 2,
        "bbands_stddev": 1.0,
        "atr_period": 1,
        "adx_period": 1,
        "stoch_k": 1,
        "stoch_d": 1,
        "use_sma": True,
        "use_ema": True,
        "use_rsi": False,
        "use_macd": False,
        "use_bbands": False,
        "use_atr": False,
        "use_adx": False,
        "use_stoch": False,
    }

    audit = {}
    report = Backtester(df, "SANITY").run(
        params,
        execution_policy="next_open",
        signals_override=signals,
        audit=audit,
    )

    executed_trades = int(report.metrics.get("trade_count", 0))
    if expected_trades > 0 and executed_trades == 0:
        raise ExecutionPipelineError(
            "Execution engine failure: signals generated but no trades executed."
        )

    return {
        "expected_trades": expected_trades,
        "executed_trades": executed_trades,
        "capital_start": audit.get("capital_start", float(Config.INITIAL_CAPITAL)),
        "capital_end": audit.get("capital_end", float(Config.INITIAL_CAPITAL)),
    }
