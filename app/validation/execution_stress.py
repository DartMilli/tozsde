"""Execution stress evaluation for GA robustness."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.analysis.analyzer import compute_signals, get_params
from app.backtesting.backtester import Backtester
from app.backtesting.equity_engine import EquityEngine
from app.backtesting.execution_utils import normalize_action
from app.validation import get_settings
from app.validation.bias_metrics import compare_execution_modes


@dataclass(frozen=True)
class _TradeReturn:
    trade_return: float


class ExecutionScenario:
    BASELINE_NEXT_OPEN = "next_open"
    ENTRY_DELAY = "next_open_entry_delay"
    EXIT_DELAY = "next_open_exit_delay"
    SLIPPAGE_NOISE = "next_open_slippage_noise"
    CLOSE_REFERENCE = "close_reference"


def _trade_indices(df, ticker: str, params: dict) -> List:
    signals, _ = compute_signals(df, ticker, params, return_series=True)
    if hasattr(signals, "tolist"):
        signals = signals.tolist()
    actions = [normalize_action(s) for s in signals]
    return Backtester(df, ticker)._generate_trade_indices(actions)


def _scenario_prices(
    trade, closes, opens, scenario: str
) -> Optional[Tuple[float, float]]:
    entry_idx = trade.entry_idx
    exit_idx = trade.exit_idx

    if scenario == ExecutionScenario.CLOSE_REFERENCE:
        return float(closes[entry_idx]), float(closes[exit_idx])

    if scenario == ExecutionScenario.BASELINE_NEXT_OPEN:
        entry_at = entry_idx + 1
        exit_at = exit_idx + 1
    elif scenario == ExecutionScenario.ENTRY_DELAY:
        entry_at = entry_idx + 2
        exit_at = exit_idx + 1
    elif scenario == ExecutionScenario.EXIT_DELAY:
        entry_at = entry_idx + 1
        exit_at = exit_idx + 2
    else:
        entry_at = entry_idx + 1
        exit_at = exit_idx + 1

    if entry_at >= len(opens) or exit_at >= len(opens):
        return None
    if entry_at >= exit_at:
        return None

    return float(opens[entry_at]), float(opens[exit_at])


def _trade_returns(
    trades,
    closes,
    opens,
    slippage,
    spread,
    fee_pct,
    scenario: str,
    rng: Optional[np.random.Generator] = None,
) -> List[Optional[float]]:
    returns: List[Optional[float]] = []
    for trade in trades:
        prices = _scenario_prices(trade, closes, opens, scenario)
        if prices is None:
            returns.append(None)
            continue
        entry_price_raw, exit_price_raw = prices
        trade_slippage = slippage
        if scenario == ExecutionScenario.SLIPPAGE_NOISE:
            noise = rng.normal(0.0, 0.0005) if rng is not None else 0.0
            trade_slippage = max(0.0, slippage + noise)
        buy_price = entry_price_raw * (1 + trade_slippage + spread / 2 + fee_pct)
        sell_price = exit_price_raw * (1 - trade_slippage - spread / 2 - fee_pct)
        trade_return = sell_price / buy_price - 1
        returns.append(trade_return)
    return returns


def _equity_metrics(
    trade_returns: List[Optional[float]],
    initial_capital: float,
    position_size_pct: Optional[float] = 0.1,
) -> Dict:
    usable = [r for r in trade_returns if r is not None]
    executions = [_TradeReturn(trade_return=r) for r in usable]
    equity_engine = EquityEngine(initial_capital)
    result = equity_engine.apply(executions, position_size_pct=position_size_pct)
    equity_curve = result.equity_curve
    if not equity_curve:
        return {
            "total_return": 0.0,
            "trade_count": 0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "equity_curve": [],
        }

    total_return = (equity_curve[-1] / initial_capital) - 1
    returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
    sharpe = 0.0
    if returns.size > 1 and float(returns.std()) > 0:
        sharpe = float(returns.mean() / returns.std()) * math.sqrt(252)

    peak = equity_curve[0]
    max_dd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = (val - peak) / peak if peak else 0.0
        if dd < max_dd:
            max_dd = dd

    return {
        "total_return": float(total_return),
        "trade_count": len(usable),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "equity_curve": equity_curve,
    }


def _equity_hash(values: list[float]) -> str:
    rounded = [round(float(v), 10) for v in values]
    payload = ",".join(str(v) for v in rounded)
    return str(hash(payload))


def evaluate_execution_stress(
    df,
    ticker: str,
    params: Optional[dict] = None,
    position_size_pct: float | None = 0.1,
    enable_stress: Optional[bool] = None,
) -> Dict:
    if params is None:
        params = get_params(ticker)

    if enable_stress is None:
        enable_stress = getattr(get_settings(), "ENABLE_EXECUTION_STRESS", False)

    trades = _trade_indices(df, ticker, params)
    if not trades:
        return {"status": "no_trades", "ticker": ticker, "stress_tested": False}

    closes = df["Close"].values
    opens = df["Open"].values if "Open" in df.columns else closes

    backtester = Backtester(df, ticker)
    slippage = float(getattr(backtester, "slippage_pct", 0.0))
    spread = float(getattr(backtester, "spread_pct", 0.0))
    fee_pct = float(getattr(backtester, "fee_pct", 0.0))

    rng = np.random.default_rng(42)

    scenarios = [
        ExecutionScenario.BASELINE_NEXT_OPEN,
        ExecutionScenario.ENTRY_DELAY,
        ExecutionScenario.SLIPPAGE_NOISE,
        ExecutionScenario.EXIT_DELAY,
        ExecutionScenario.CLOSE_REFERENCE,
    ]

    scenario_metrics: Dict[str, Dict] = {}
    for scenario in scenarios:
        returns = _trade_returns(
            trades,
            closes,
            opens,
            slippage,
            spread,
            fee_pct,
            scenario,
            rng=rng,
        )
        metrics = _equity_metrics(returns, 1.0, position_size_pct=position_size_pct)
        scenario_metrics[scenario] = metrics

    close_return = scenario_metrics[ExecutionScenario.CLOSE_REFERENCE]["total_return"]
    for metrics in scenario_metrics.values():
        rel_gap = compare_execution_modes(close_return, metrics["total_return"]).get(
            "relative_gap"
        )
        metrics["relative_gap_vs_close"] = (
            float(rel_gap) if isinstance(rel_gap, (int, float)) else None
        )

    baseline = scenario_metrics[ExecutionScenario.BASELINE_NEXT_OPEN]
    baseline_sharpe = float(baseline.get("sharpe", 0.0))
    baseline_trade_count = int(baseline.get("trade_count", 0))

    baseline_report = backtester.run(
        params,
        execution_policy="next_open",
        fixed_position_pct=0.1,
    )
    baseline_meta = baseline_report.meta or {}
    baseline_indices = baseline_meta.get("trade_indices", [])
    if len(baseline_indices) != len(trades):
        from app.validation.errors import EngineLogicError

        raise EngineLogicError("trade_count_mismatch")
    for left, right in zip(baseline_indices, trades):
        if left != {"entry_idx": right.entry_idx, "exit_idx": right.exit_idx}:
            from app.validation.errors import EngineLogicError

            raise EngineLogicError("trade_index_mismatch")

    baseline_equity = baseline_report.diagnostics.get("equity_curve")
    if hasattr(baseline_equity, "tolist"):
        baseline_equity = baseline_equity.tolist()
    stress_equity = baseline.get("equity_curve")
    if baseline_equity and stress_equity:
        scale = float(baseline_equity[0]) if baseline_equity else 1.0
        if scale == 0:
            scale = 1.0
        normalized_baseline = [float(v) / scale for v in baseline_equity]
        stress_scale = float(stress_equity[0]) if stress_equity else 1.0
        if stress_scale == 0:
            stress_scale = 1.0
        normalized_stress = [float(v) / stress_scale for v in stress_equity]
        if _equity_hash(normalized_baseline) != _equity_hash(normalized_stress):
            from app.validation.errors import EngineLogicError

            raise EngineLogicError("equity_curve_mismatch")

    if not enable_stress:
        return {
            "status": "ok",
            "ticker": ticker,
            "baseline_sharpe": baseline_sharpe,
            "robustness_score": baseline_sharpe,
            "sharpe_std": 0.0,
            "worst_case_sharpe": baseline_sharpe,
            "relative_gap_baseline": baseline.get("relative_gap_vs_close"),
            "fitness": baseline_sharpe,
            "constraint_passed": True,
            "constraint_fail_reasons": [],
            "stress_tested": False,
            "trade_count": baseline_trade_count,
            "scenarios": scenario_metrics,
        }

    sharpe_values = [float(v.get("sharpe", 0.0)) for v in scenario_metrics.values()]
    return_values = [
        float(v.get("total_return", 0.0)) for v in scenario_metrics.values()
    ]
    robustness_score = float(np.mean(sharpe_values)) if sharpe_values else 0.0
    sharpe_std = float(np.std(sharpe_values)) if sharpe_values else 0.0
    return_std = float(np.std(return_values)) if return_values else 0.0
    stability_factor = math.exp(-sharpe_std) * math.exp(-return_std)
    worst_case_sharpe = min(sharpe_values) if sharpe_values else 0.0

    relative_gap_baseline = baseline.get("relative_gap_vs_close")
    if not isinstance(relative_gap_baseline, (int, float)):
        relative_gap_baseline = 0.0

    mean_oos_sharpe = robustness_score

    constraint_passed, fail_reasons = evaluate_constraints(
        mean_oos_sharpe,
        relative_gap_baseline,
        worst_case_sharpe,
        min_sharpe=get_settings().MIN_OOS_SHARPE,
        max_relative_gap=get_settings().MAX_RELATIVE_GAP,
    )

    if baseline_trade_count < 40:
        fail_reasons.append("min_trade_count")
        constraint_passed = False

    fitness = mean_oos_sharpe * stability_factor
    fitness *= max(0.0, 1 - relative_gap_baseline)

    compounding_amplification = None
    base_comp = _equity_metrics(
        _trade_returns(
            trades,
            closes,
            opens,
            slippage,
            spread,
            fee_pct,
            ExecutionScenario.BASELINE_NEXT_OPEN,
            rng=rng,
        ),
        1.0,
        position_size_pct=None,
    )
    close_comp = _equity_metrics(
        _trade_returns(
            trades,
            closes,
            opens,
            slippage,
            spread,
            fee_pct,
            ExecutionScenario.CLOSE_REFERENCE,
            rng=rng,
        ),
        1.0,
        position_size_pct=None,
    )
    fixed_gap = abs(
        baseline.get("total_return", 0.0)
        - scenario_metrics[ExecutionScenario.CLOSE_REFERENCE]["total_return"]
    ) / max(
        abs(scenario_metrics[ExecutionScenario.CLOSE_REFERENCE]["total_return"]), 1e-9
    )
    comp_gap = abs(
        base_comp.get("total_return", 0.0) - close_comp.get("total_return", 0.0)
    ) / max(abs(close_comp.get("total_return", 0.0)), 1e-9)
    if fixed_gap > 0:
        compounding_amplification = comp_gap / fixed_gap
        if compounding_amplification > 1.5:
            fitness *= 0.8

    if not constraint_passed:
        fitness = NEG_INF

    return {
        "status": "ok",
        "ticker": ticker,
        "baseline_sharpe": baseline_sharpe,
        "mean_oos_sharpe": mean_oos_sharpe,
        "robustness_score": robustness_score,
        "sharpe_std": sharpe_std,
        "return_std": return_std,
        "worst_case_sharpe": worst_case_sharpe,
        "relative_gap_baseline": relative_gap_baseline,
        "trade_count": baseline_trade_count,
        "fitness": fitness,
        "constraint_passed": constraint_passed,
        "constraint_fail_reasons": fail_reasons,
        "stability_factor": stability_factor,
        "compounding_amplification": compounding_amplification,
        "stress_tested": True,
        "scenarios": scenario_metrics,
    }


def evaluate_constraints(
    mean_oos_sharpe: float,
    relative_gap_baseline: float,
    worst_case_sharpe: float,
    min_sharpe: float,
    max_relative_gap: float,
) -> tuple[bool, list[str]]:
    fail_reasons = []
    if mean_oos_sharpe < min_sharpe:
        fail_reasons.append("mean_oos_sharpe")
    if relative_gap_baseline > max_relative_gap:
        fail_reasons.append("relative_gap")
    if worst_case_sharpe < 0:
        fail_reasons.append("worst_case_sharpe")

    return not fail_reasons, fail_reasons


# Avoid circular import; local import is intentional for NEG_INF.
from app.optimization.fitness import NEG_INF  # noqa: E402
