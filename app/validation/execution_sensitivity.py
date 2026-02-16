"""Execution sensitivity analysis for trade-level timing and pricing effects."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from app.analysis.analyzer import compute_signals, get_params
from app.backtesting.execution_utils import normalize_action
from app.backtesting.equity_engine import EquityEngine
from app.backtesting.backtester import Backtester
from app.data_access.data_loader import load_data
from app.validation.utils import get_validation_ticker, get_validation_window


@dataclass(frozen=True)
class _TradeReturn:
    trade_return: float


class ExecutionVariant:
    CLOSE_REFERENCE = "close_reference"
    SAME_BAR_OPEN = "same_bar_open"
    NEXT_OPEN_SHIFT = "next_open_shift"
    HYBRID_TEST = "hybrid_test"


def _paired_t_test(values_a: List[float], values_b: List[float]) -> Dict:
    pairs = [
        (a, b) for a, b in zip(values_a, values_b) if a is not None and b is not None
    ]
    if len(pairs) < 2:
        return {"n": len(pairs), "t_stat": None, "p_value": None, "approx": True}

    diffs = [a - b for a, b in pairs]
    mean_diff = sum(diffs) / len(diffs)
    var = sum((d - mean_diff) ** 2 for d in diffs) / (len(diffs) - 1)
    std = math.sqrt(var)
    if std == 0:
        return {"n": len(pairs), "t_stat": 0.0, "p_value": 1.0, "approx": True}

    t_stat = mean_diff / (std / math.sqrt(len(diffs)))
    p_value = math.erfc(abs(t_stat) / math.sqrt(2))
    return {"n": len(pairs), "t_stat": t_stat, "p_value": p_value, "approx": True}


def _trade_indices(df) -> List[dict]:
    params = get_params(get_validation_ticker())
    signals, _ = compute_signals(
        df, get_validation_ticker(), params, return_series=True
    )
    if hasattr(signals, "tolist"):
        signals = signals.tolist()
    actions = [normalize_action(s) for s in signals]
    return Backtester(df, get_validation_ticker())._generate_trade_indices(actions)


def _variant_prices(
    trade, closes, opens, variant: str
) -> Optional[Tuple[float, float]]:
    entry_idx = trade.entry_idx
    exit_idx = trade.exit_idx

    if variant == ExecutionVariant.CLOSE_REFERENCE:
        return float(closes[entry_idx]), float(closes[exit_idx])

    if variant == ExecutionVariant.SAME_BAR_OPEN:
        return float(opens[entry_idx]), float(opens[exit_idx])

    if variant == ExecutionVariant.NEXT_OPEN_SHIFT:
        entry_at = entry_idx + 1
        exit_at = exit_idx + 1
        if entry_at >= len(opens) or exit_at >= len(opens):
            return None
        return float(opens[entry_at]), float(opens[exit_at])

    if variant == ExecutionVariant.HYBRID_TEST:
        exit_at = exit_idx + 1
        if exit_at >= len(opens):
            return None
        return float(closes[entry_idx]), float(opens[exit_at])

    return None


def _execution_returns(
    trades, closes, opens, slippage, spread, fee_pct, variant: str
) -> List[Optional[float]]:
    returns = []
    for trade in trades:
        prices = _variant_prices(trade, closes, opens, variant)
        if prices is None:
            returns.append(None)
            continue
        entry_price_raw, exit_price_raw = prices
        buy_price = entry_price_raw * (1 + slippage + spread / 2 + fee_pct)
        sell_price = exit_price_raw * (1 - slippage - spread / 2 - fee_pct)
        trade_return = sell_price / buy_price - 1
        returns.append(trade_return)
    return returns


def _equity_metrics(
    trade_returns: List[Optional[float]],
    initial_capital: float,
    position_size_pct: Optional[float],
) -> Dict:
    usable = [r for r in trade_returns if r is not None]
    executions = [_TradeReturn(trade_return=r) for r in usable]
    equity_engine = EquityEngine(initial_capital)
    result = equity_engine.apply(executions, position_size_pct=position_size_pct)
    equity_curve = result.equity_curve
    if not equity_curve:
        return {
            "total_return": 0.0,
            "sum_trade_returns": 0.0,
            "trade_count": 0,
            "avg_trade_return": 0.0,
            "std_trade_return": 0.0,
            "max_drawdown": 0.0,
        }
    total_return = (equity_curve[-1] / initial_capital) - 1
    sum_trade_returns = sum(usable)
    avg_trade_return = sum_trade_returns / len(usable) if usable else 0.0
    if len(usable) > 1:
        mean_val = avg_trade_return
        var = sum((r - mean_val) ** 2 for r in usable) / (len(usable) - 1)
        std_trade_return = math.sqrt(var)
    else:
        std_trade_return = 0.0

    peak = equity_curve[0]
    max_dd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = (val - peak) / peak if peak else 0.0
        if dd < max_dd:
            max_dd = dd

    return {
        "total_return": total_return,
        "sum_trade_returns": sum_trade_returns,
        "trade_count": len(usable),
        "avg_trade_return": avg_trade_return,
        "std_trade_return": std_trade_return,
        "max_drawdown": max_dd,
    }


def _gap_stats(trades, closes, opens) -> Dict:
    entry_gaps = []
    exit_gaps = []
    for trade in trades:
        entry_idx = trade.entry_idx
        exit_idx = trade.exit_idx
        if entry_idx + 1 >= len(opens) or exit_idx + 1 >= len(opens):
            continue
        close_entry = float(closes[entry_idx])
        close_exit = float(closes[exit_idx])
        open_entry = float(opens[entry_idx + 1])
        open_exit = float(opens[exit_idx + 1])
        entry_gaps.append(open_entry / close_entry - 1)
        exit_gaps.append(open_exit / close_exit - 1)

    combined = entry_gaps + exit_gaps
    pos_ratio = (
        (len([g for g in combined if g > 0]) / len(combined)) if combined else 0.0
    )
    neg_ratio = (
        (len([g for g in combined if g < 0]) / len(combined)) if combined else 0.0
    )

    def _mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    def _std(vals):
        if len(vals) < 2:
            return 0.0
        m = _mean(vals)
        return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))

    return {
        "mean_entry_gap": _mean(entry_gaps),
        "mean_exit_gap": _mean(exit_gaps),
        "std_entry_gap": _std(entry_gaps),
        "std_exit_gap": _std(exit_gaps),
        "positive_gap_ratio": pos_ratio,
        "negative_gap_ratio": neg_ratio,
    }


def run_execution_sensitivity() -> Dict:
    ticker = get_validation_ticker()
    start, end = get_validation_window()
    df = load_data(ticker, start=start.isoformat(), end=end.isoformat())
    if df is None or df.empty:
        return {"status": "no_data", "ticker": ticker}

    df = df[(df.index.date >= start) & (df.index.date <= end)]
    if df.empty:
        return {"status": "no_data", "ticker": ticker}

    trades = _trade_indices(df)
    closes = df["Close"].values
    opens = df["Open"].values if "Open" in df.columns else closes

    slippage = float(getattr(Backtester(df, ticker), "slippage_pct", 0.0))
    spread = float(getattr(Backtester(df, ticker), "spread_pct", 0.0))
    fee_pct = float(getattr(Backtester(df, ticker), "fee_pct", 0.0))

    variants = [
        ExecutionVariant.CLOSE_REFERENCE,
        ExecutionVariant.SAME_BAR_OPEN,
        ExecutionVariant.NEXT_OPEN_SHIFT,
        ExecutionVariant.HYBRID_TEST,
    ]

    policy_returns = {}
    fixed_returns = {}
    comp_returns = {}
    variant_trade_returns = {}

    for variant in variants:
        returns = _execution_returns(
            trades, closes, opens, slippage, spread, fee_pct, variant
        )
        variant_trade_returns[variant] = returns

        fixed_metrics = _equity_metrics(returns, 1.0, position_size_pct=0.1)
        comp_metrics = _equity_metrics(returns, 1.0, position_size_pct=None)
        fixed_returns[variant] = fixed_metrics
        comp_returns[variant] = comp_metrics
        policy_returns[variant] = {
            "fixed_size": fixed_metrics,
            "compounding": comp_metrics,
        }

    price_sampling_effect = (
        fixed_returns[ExecutionVariant.SAME_BAR_OPEN]["total_return"]
        - fixed_returns[ExecutionVariant.CLOSE_REFERENCE]["total_return"]
    )
    timing_effect = (
        fixed_returns[ExecutionVariant.NEXT_OPEN_SHIFT]["total_return"]
        - fixed_returns[ExecutionVariant.SAME_BAR_OPEN]["total_return"]
    )
    exit_drift = (
        fixed_returns[ExecutionVariant.HYBRID_TEST]["total_return"]
        - fixed_returns[ExecutionVariant.CLOSE_REFERENCE]["total_return"]
    )

    gap_stats = _gap_stats(trades, closes, opens)

    close_returns = variant_trade_returns[ExecutionVariant.CLOSE_REFERENCE]
    same_open_returns = variant_trade_returns[ExecutionVariant.SAME_BAR_OPEN]
    next_open_returns = variant_trade_returns[ExecutionVariant.NEXT_OPEN_SHIFT]

    stat_tests = {
        "close_vs_same_open": _paired_t_test(close_returns, same_open_returns),
        "same_open_vs_next_open": _paired_t_test(same_open_returns, next_open_returns),
    }

    fixed_gap = abs(
        fixed_returns[ExecutionVariant.NEXT_OPEN_SHIFT]["total_return"]
        - fixed_returns[ExecutionVariant.CLOSE_REFERENCE]["total_return"]
    ) / max(abs(fixed_returns[ExecutionVariant.CLOSE_REFERENCE]["total_return"]), 1e-9)
    comp_gap = abs(
        comp_returns[ExecutionVariant.NEXT_OPEN_SHIFT]["total_return"]
        - comp_returns[ExecutionVariant.CLOSE_REFERENCE]["total_return"]
    ) / max(abs(comp_returns[ExecutionVariant.CLOSE_REFERENCE]["total_return"]), 1e-9)
    amplification = comp_gap / fixed_gap if fixed_gap > 0 else None

    interpretation = {
        "timing_dominates": abs(timing_effect) > abs(price_sampling_effect),
        "exit_bias_dominates": abs(gap_stats.get("mean_exit_gap", 0.0))
        > abs(gap_stats.get("mean_entry_gap", 0.0)),
        "compounding_amplification": amplification is not None and amplification > 1.5,
    }

    return {
        "status": "ok",
        "ticker": ticker,
        "trade_count": len(trades),
        "policy_returns": policy_returns,
        "price_sampling_effect": price_sampling_effect,
        "timing_effect": timing_effect,
        "exit_drift": exit_drift,
        "gap_stats": gap_stats,
        "compounding_amplification": amplification,
        "stat_tests": stat_tests,
        "interpretation": interpretation,
    }
