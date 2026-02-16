"""Trade distribution quality metrics for stability diagnostics."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from app.analysis.analyzer import get_params
from app.backtesting.backtester import Backtester
from app.config.config import Config
from app.data_access.data_loader import ensure_data_cached, load_data
from app.validation.utils import get_validation_ticker


def _ratio_top_contributors(values: Iterable[float], top_pct: float) -> float:
    values = [v for v in values if v > 0]
    if not values:
        return 0.0
    values_sorted = sorted(values, reverse=True)
    top_n = max(1, int(round(len(values_sorted) * top_pct)))
    top_sum = float(sum(values_sorted[:top_n]))
    total = float(sum(values_sorted))
    return top_sum / total if total else 0.0


def _ratio_worst_losses(values: Iterable[float], bottom_pct: float) -> float:
    losses = [abs(v) for v in values if v < 0]
    if not losses:
        return 0.0
    losses_sorted = sorted(losses, reverse=True)
    top_n = max(1, int(round(len(losses_sorted) * bottom_pct)))
    worst_sum = float(sum(losses_sorted[:top_n]))
    total = float(sum(losses_sorted))
    return worst_sum / total if total else 0.0


def _skewness(values: list[float]) -> float:
    if len(values) < 3:
        return 0.0
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    if std == 0:
        return 0.0
    centered = arr - mean
    return float((centered**3).mean() / (std**3))


def _kurtosis(values: list[float]) -> float:
    if len(values) < 4:
        return 0.0
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    if std == 0:
        return 0.0
    centered = arr - mean
    return float((centered**4).mean() / (std**4) - 3.0)


def analyze_trade_quality(
    ticker: Optional[str] = None,
    params: Optional[dict] = None,
) -> dict:
    ticker = ticker or get_validation_ticker()
    if params is None:
        params = get_params(ticker)

    if not ensure_data_cached(ticker, start=Config.START_DATE, end=Config.END_DATE):
        return {"status": "no_data", "ticker": ticker}

    df = load_data(ticker, start=Config.START_DATE, end=Config.END_DATE)
    backtester = Backtester(df, ticker)
    report = backtester.run(params, execution_policy="next_open")

    executions = report.meta.get("trade_executions", []) if report.meta else []
    trade_returns = [
        float(t.get("trade_return"))
        for t in executions
        if t.get("trade_return") is not None
    ]
    trade_count = len(trade_returns)

    top20_ratio = _ratio_top_contributors(trade_returns, 0.2)
    worst20_ratio = _ratio_worst_losses(trade_returns, 0.2)

    skewness = _skewness(trade_returns)
    kurtosis = _kurtosis(trade_returns)

    return {
        "status": "ok",
        "ticker": ticker,
        "trade_count": trade_count,
        "top20_contribution_ratio": round(top20_ratio, 6),
        "worst20_loss_ratio": round(worst20_ratio, 6),
        "skewness": round(skewness, 6),
        "kurtosis": round(kurtosis, 6),
        "edge_concentration": top20_ratio > 0.6,
    }
