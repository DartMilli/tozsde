import math
from app.reporting.metrics import WalkForwardMetrics


NEG_INF = -1e12
_FITNESS_CACHE = {}
_FITNESS_CACHE_MAX = 1024


def _cache_get(key):
    return _FITNESS_CACHE.get(key)


def _cache_set(key, value):
    if len(_FITNESS_CACHE) >= _FITNESS_CACHE_MAX:
        _FITNESS_CACHE.clear()
    _FITNESS_CACHE[key] = value


def fitness_single(metrics) -> float:  # BacktestReport.metrics
    key = (
        metrics.trade_count,
        metrics.net_profit,
        metrics.max_drawdown,
        metrics.winrate,
    )
    cached = _cache_get(("single",) + key)
    if cached is not None:
        return cached

    if metrics.trade_count < 30:
        return NEG_INF

    score = (
        metrics.net_profit
        - 2.0 * metrics.max_drawdown
        + 0.5 * metrics.winrate * metrics.net_profit
    )
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
