"""Execution drift monitoring utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.analysis.analyzer import compute_signals, get_params
from app.backtesting.backtester import Backtester
from app.backtesting.equity_engine import EquityEngine
from app.backtesting.execution_utils import normalize_action
from app.config.config import Config
from app.validation.execution_stress import evaluate_execution_stress
from app.validation.bias_metrics import compare_execution_modes


@dataclass(frozen=True)
class _TradeReturn:
    trade_return: float


class ExecutionVariant:
    CLOSE_REFERENCE = "close_reference"
    SAME_BAR_OPEN = "same_bar_open"
    NEXT_OPEN_SHIFT = "next_open_shift"
    HYBRID_TEST = "hybrid_test"


DRIFT_STATE_PATH = Path(Config.LOG_DIR) / "execution_drift_state.json"


def _trade_indices(df, ticker: str, params: dict) -> List:
    signals, _ = compute_signals(df, ticker, params, return_series=True)
    if hasattr(signals, "tolist"):
        signals = signals.tolist()
    actions = [normalize_action(s) for s in signals]
    return Backtester(df, ticker)._generate_trade_indices(actions)


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
    trade_returns: List[Optional[float]], initial_capital: float
) -> Dict:
    usable = [r for r in trade_returns if r is not None]
    executions = [_TradeReturn(trade_return=r) for r in usable]
    equity_engine = EquityEngine(initial_capital)
    result = equity_engine.apply(executions, position_size_pct=0.1)
    equity_curve = result.equity_curve
    if not equity_curve:
        return {"total_return": 0.0}
    total_return = (equity_curve[-1] / initial_capital) - 1
    return {"total_return": total_return}


def compute_execution_drift(df, ticker: str, params: dict | None = None) -> Dict:
    if params is None:
        params = get_params(ticker)

    trades = _trade_indices(df, ticker, params)
    if not trades:
        return {"status": "no_trades", "ticker": ticker}

    closes = df["Close"].values
    opens = df["Open"].values if "Open" in df.columns else closes

    backtester = Backtester(df, ticker)
    slippage = float(getattr(backtester, "slippage_pct", 0.0))
    spread = float(getattr(backtester, "spread_pct", 0.0))
    fee_pct = float(getattr(backtester, "fee_pct", 0.0))

    close_returns = _execution_returns(
        trades,
        closes,
        opens,
        slippage,
        spread,
        fee_pct,
        ExecutionVariant.CLOSE_REFERENCE,
    )
    same_open_returns = _execution_returns(
        trades,
        closes,
        opens,
        slippage,
        spread,
        fee_pct,
        ExecutionVariant.SAME_BAR_OPEN,
    )
    next_open_returns = _execution_returns(
        trades,
        closes,
        opens,
        slippage,
        spread,
        fee_pct,
        ExecutionVariant.NEXT_OPEN_SHIFT,
    )
    hybrid_returns = _execution_returns(
        trades,
        closes,
        opens,
        slippage,
        spread,
        fee_pct,
        ExecutionVariant.HYBRID_TEST,
    )

    close_metrics = _equity_metrics(close_returns, 1.0)
    next_metrics = _equity_metrics(next_open_returns, 1.0)
    same_metrics = _equity_metrics(same_open_returns, 1.0)
    hybrid_metrics = _equity_metrics(hybrid_returns, 1.0)

    relative_gap = compare_execution_modes(
        close_metrics["total_return"],
        next_metrics["total_return"],
    ).get("relative_gap")

    timing_effect = next_metrics["total_return"] - same_metrics["total_return"]
    exit_drift = hybrid_metrics["total_return"] - close_metrics["total_return"]

    stress_metrics = None
    if Config.ENABLE_EXECUTION_STRESS:
        stress_metrics = evaluate_execution_stress(
            df,
            ticker,
            params,
            enable_stress=True,
        )

    return {
        "status": "ok",
        "ticker": ticker,
        "trade_count": len(trades),
        "relative_gap": (
            float(relative_gap) if isinstance(relative_gap, (int, float)) else None
        ),
        "timing_effect": float(timing_effect),
        "exit_drift": float(exit_drift),
        "live_sharpe": (
            stress_metrics.get("baseline_sharpe") if stress_metrics else None
        ),
        "live_sharpe_std": stress_metrics.get("sharpe_std") if stress_metrics else None,
        "live_worst_case_proxy": (
            stress_metrics.get("worst_case_sharpe") if stress_metrics else None
        ),
    }


def load_drift_state(path: Path | None = None) -> Dict:
    state_path = path or DRIFT_STATE_PATH
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def update_drift_state(metrics: Dict, path: Path | None = None) -> Dict:
    state_path = path or DRIFT_STATE_PATH
    previous = load_drift_state(state_path)
    previous_gap = previous.get("relative_gap")
    current_gap = metrics.get("relative_gap")

    drift_warning = False
    change_pct = None
    if isinstance(previous_gap, (int, float)) and isinstance(current_gap, (int, float)):
        if abs(previous_gap) > 0:
            change_pct = (current_gap - previous_gap) / abs(previous_gap)
            drift_warning = change_pct > 0.3

    live_sharpe = metrics.get("live_sharpe")
    sharpe_warning = isinstance(live_sharpe, (int, float)) and live_sharpe < 0.3

    updated = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "relative_gap": current_gap,
        "timing_effect": metrics.get("timing_effect"),
        "exit_drift": metrics.get("exit_drift"),
        "live_sharpe": live_sharpe,
        "live_sharpe_std": metrics.get("live_sharpe_std"),
        "live_worst_case_proxy": metrics.get("live_worst_case_proxy"),
        "drift_warning": drift_warning,
        "relative_gap_change_pct": change_pct,
        "sharpe_warning": sharpe_warning,
    }

    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(updated, indent=2), encoding="utf-8")
    except Exception:
        pass

    return updated
