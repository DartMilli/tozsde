"""
bias_tests.py
Lookahead és leakage detektálás.

Helye:
app/validation/bias_tests.py
"""

import os
import statistics

from app.backtesting.backtester import Backtester
from app.analysis.analyzer import get_params, compute_signals
from app.data_access.data_loader import load_data
from app.validation.bias_metrics import compare_execution_modes
from app.backtesting.execution_utils import normalize_action
from app.validation.errors import ValidationError, EngineLogicError
from app.validation.utils import get_validation_ticker, get_validation_window


def run_bias_tests() -> dict:
    """
    Alignment + lookahead checks for execution policies.
    """

    alignment = run_alignment_checks()
    return alignment


# --- Stub ---
# Ezt a projekted backtesterére kell rákötni


def run_backtest(execution_mode: str, fixed_position_pct: float | None = None) -> dict:
    ticker = get_validation_ticker()
    start, end = get_validation_window()

    df = load_data(ticker, start=start.isoformat(), end=end.isoformat())
    if df is None or df.empty:
        return {"total_return": 0.0}

    df = df[(df.index.date >= start) & (df.index.date <= end)]
    if df.empty:
        return {"total_return": 0.0}

    params = get_params(ticker)
    report = Backtester(df, ticker).run(
        params,
        execution_policy=execution_mode,
        debug_trace=True,
        fixed_position_pct=fixed_position_pct,
    )
    return {
        "total_return": float(report.metrics.get("net_profit", 0.0)),
        "max_drawdown": float(report.metrics.get("max_drawdown", 0.0)),
        "trade_count": int(report.metrics.get("trade_count", 0)),
        "sharpe": float(report.diagnostics.get("sharpe", 0.0)),
        "trade_indices": report.meta.get("trade_indices", []),
        "trade_executions": report.meta.get("trade_executions", []),
    }


def run_alignment_diagnostics() -> dict:
    ticker = get_validation_ticker()
    start, end = get_validation_window()

    df = load_data(ticker, start=start.isoformat(), end=end.isoformat())
    if df is None or df.empty:
        return {"status": "no_data"}

    df = df[(df.index.date >= start) & (df.index.date <= end)]
    if df.empty:
        return {"status": "no_data"}

    params = get_params(ticker)
    signals, _ = compute_signals(df, ticker, params, return_series=True)
    if hasattr(signals, "tolist"):
        signals = signals.tolist()

    actions = [normalize_action(s) for s in signals]
    trade_indices = Backtester(df, ticker)._generate_trade_indices(actions)

    close_returns = []
    open_returns = []
    entry_gaps = []
    exit_gaps = []
    sample = []
    closes = df["Close"].values
    opens = df["Open"].values if "Open" in df.columns else closes

    for trade in trade_indices[:20]:
        entry_idx = trade.entry_idx
        exit_idx = trade.exit_idx
        if entry_idx + 1 >= len(opens) or exit_idx + 1 >= len(opens):
            continue
        close_entry = float(closes[entry_idx])
        close_exit = float(closes[exit_idx])
        open_entry = float(opens[entry_idx + 1])
        open_exit = float(opens[exit_idx + 1])
        close_ret = close_exit / close_entry - 1
        open_ret = open_exit / open_entry - 1
        entry_gap = open_entry / close_entry - 1
        exit_gap = open_exit / close_exit - 1
        close_returns.append(close_ret)
        open_returns.append(open_ret)
        entry_gaps.append(entry_gap)
        exit_gaps.append(exit_gap)
        sample.append(
            {
                "entry_idx": entry_idx,
                "exit_idx": exit_idx,
                "close_entry": close_entry,
                "close_exit": close_exit,
                "open_entry": open_entry,
                "open_exit": open_exit,
                "close_return": close_ret,
                "open_return": open_ret,
                "entry_gap": entry_gap,
                "exit_gap": exit_gap,
                "diff": open_ret - close_ret,
            }
        )

    sum_close = float(sum(close_returns)) if close_returns else 0.0
    sum_open = float(sum(open_returns)) if open_returns else 0.0

    return {
        "status": "ok",
        "trade_count": len(trade_indices),
        "first_trades": sample,
        "sum_close_returns": sum_close,
        "sum_open_returns": sum_open,
        "avg_entry_gap": (
            float(sum(entry_gaps) / len(entry_gaps)) if entry_gaps else 0.0
        ),
        "avg_exit_gap": float(sum(exit_gaps) / len(exit_gaps)) if exit_gaps else 0.0,
        "std_gap": (
            float(statistics.pstdev(entry_gaps + exit_gaps))
            if entry_gaps or exit_gaps
            else 0.0
        ),
    }


def run_alignment_checks() -> dict:
    close_fixed = run_backtest("close_to_close", fixed_position_pct=0.1)
    next_fixed = run_backtest("next_open", fixed_position_pct=0.1)

    close_comp = run_backtest("close_to_close", fixed_position_pct=None)
    next_comp = run_backtest("next_open", fixed_position_pct=None)

    close_trades = close_fixed.get("trade_indices", [])
    next_trades = next_fixed.get("trade_indices", [])
    if len(close_trades) != len(next_trades):
        raise EngineLogicError("Trade count mismatch between execution policies")
    for idx, (left, right) in enumerate(zip(close_trades, next_trades)):
        if left != right:
            raise EngineLogicError(f"Trade index mismatch at {idx}: {left} vs {right}")

    fixed_results = compare_execution_modes(
        close_return=close_fixed.get("total_return", 0.0),
        next_return=next_fixed.get("total_return", 0.0),
    )
    comp_results = compare_execution_modes(
        close_return=close_comp.get("total_return", 0.0),
        next_return=next_comp.get("total_return", 0.0),
    )

    diagnostics = run_alignment_diagnostics()

    return {
        **fixed_results,
        "engine_integrity": {
            "trade_count_match": True,
            "trade_index_match": True,
        },
        "fixed_size": {
            "close_return": close_fixed.get("total_return", 0.0),
            "next_open_return": next_fixed.get("total_return", 0.0),
            "relative_gap": fixed_results.get("relative_gap"),
            "alignment_ok": fixed_results.get("alignment_ok"),
            "alignment_error": fixed_results.get("alignment_error"),
        },
        "compounding": {
            "close_return": close_comp.get("total_return", 0.0),
            "next_open_return": next_comp.get("total_return", 0.0),
            "relative_gap": comp_results.get("relative_gap"),
            "alignment_ok": comp_results.get("alignment_ok"),
            "alignment_error": comp_results.get("alignment_error"),
        },
        "diagnostics": diagnostics,
    }
