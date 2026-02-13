from dataclasses import dataclass

import numpy as np

from app.config.config import Config
from app.reporting.metrics import WalkForwardMetrics


NEG_INF = -1e12
_FITNESS_CACHE = {}
_FITNESS_CACHE_MAX = 1024


@dataclass(frozen=True)
class WalkForwardResult:
    avg_profit: float
    avg_drawdown: float
    profit_std: float
    dd_std: float
    negative_fold_ratio: float
    raw_fitness: float
    normalized_score: float


def _cache_get(key):
    return _FITNESS_CACHE.get(key)


def _cache_set(key, value):
    if len(_FITNESS_CACHE) >= _FITNESS_CACHE_MAX:
        _FITNESS_CACHE.clear()
    _FITNESS_CACHE[key] = value


def _get_metric(metrics, key: str, default: float = 0.0):
    if hasattr(metrics, key):
        return getattr(metrics, key)
    if isinstance(metrics, dict):
        return metrics.get(key, default)
    return default


def fitness_single(metrics) -> float:  # BacktestReport.metrics
    trade_count = _get_metric(metrics, "trade_count")
    net_profit = _get_metric(metrics, "net_profit")
    max_drawdown = _get_metric(metrics, "max_drawdown")
    winrate = _get_metric(metrics, "winrate")

    key = (
        trade_count,
        net_profit,
        max_drawdown,
        winrate,
    )
    cached = _cache_get(("single",) + key)
    if cached is not None:
        return cached

    if trade_count < 30:
        return NEG_INF

    score = net_profit - 2.0 * max_drawdown + 0.5 * winrate * net_profit
    _cache_set(("single",) + key, score)
    return score


def fitness_walk_forward(wf_metrics: WalkForwardMetrics) -> float:
    key = (
        wf_metrics.avg_profit,
        wf_metrics.avg_dd,
        wf_metrics.profit_std,
        wf_metrics.dd_std,
        wf_metrics.negative_fold_ratio,
    )
    cached = _cache_get(("wf",) + key)
    if cached is not None:
        return cached

    if wf_metrics.avg_profit <= 0:
        return NEG_INF

    if wf_metrics.negative_fold_ratio > 0.5:
        return NEG_INF

    wf_penalty = (
        wf_metrics.profit_std * 1.5
        + wf_metrics.dd_std * 1.0
        + wf_metrics.negative_fold_ratio * abs(wf_metrics.avg_profit) * 2.0
    )

    score = wf_metrics.avg_profit - wf_metrics.avg_dd - wf_penalty
    _cache_set(("wf",) + key, score)
    return score


def normalize_wf_score(
    raw_fitness: float, stability_constant: float | None = None
) -> float:
    if stability_constant is None:
        stability_constant = float(Config.WF_STABILITY_CONSTANT)

    if not isinstance(raw_fitness, (int, float)):
        return 0.0

    if raw_fitness <= 0:
        return 0.0

    score = raw_fitness / (raw_fitness + stability_constant)
    return max(0.0, min(1.0, score))


def compute_walk_forward_metrics(oos_profits, oos_drawdowns) -> WalkForwardResult:
    oos_profits = np.array(oos_profits, dtype=float)
    oos_drawdowns = np.array(oos_drawdowns, dtype=float)

    avg_profit = float(np.mean(oos_profits)) if oos_profits.size else 0.0
    avg_drawdown = float(np.mean(oos_drawdowns)) if oos_drawdowns.size else 0.0
    profit_std = float(np.std(oos_profits)) if oos_profits.size else 0.0
    dd_std = float(np.std(oos_drawdowns)) if oos_drawdowns.size else 0.0
    negative_fold_ratio = float(np.mean(oos_profits < 0)) if oos_profits.size else 0.0

    wf_metrics = WalkForwardMetrics(
        avg_profit=avg_profit,
        avg_dd=avg_drawdown,
        profit_std=profit_std,
        dd_std=dd_std,
        negative_fold_ratio=negative_fold_ratio,
    )

    raw_fitness = fitness_walk_forward(wf_metrics)
    normalized_score = normalize_wf_score(raw_fitness)

    return WalkForwardResult(
        avg_profit=avg_profit,
        avg_drawdown=avg_drawdown,
        profit_std=profit_std,
        dd_std=dd_std,
        negative_fold_ratio=negative_fold_ratio,
        raw_fitness=raw_fitness,
        normalized_score=normalized_score,
    )
