import math
from app.reporting.metrics import WalkForwardMetrics


NEG_INF = -1e12


def fitness_single(metrics) -> float:  # BacktestReport.metrics
    if metrics.trade_count < 30:
        return NEG_INF

    return (
        metrics.net_profit
        - 2.0 * metrics.max_drawdown
        + 0.5 * metrics.winrate * metrics.net_profit
    )


def fitness_walk_forward(wf_metrics: WalkForwardMetrics) -> float:
    if wf_metrics.avg_profit <= 0:
        return NEG_INF

    if wf_metrics.negative_fold_ratio > 0.5:
        return NEG_INF

    wf_penalty = (
        wf_metrics.profit_std * 1.5
        + wf_metrics.dd_std * 1.0
        + wf_metrics.negative_fold_ratio * abs(wf_metrics.avg_profit) * 2.0
    )

    return wf_metrics.avg_profit - wf_metrics.avg_dd - wf_penalty
