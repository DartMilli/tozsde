"""Monthly production readiness checks."""

from __future__ import annotations

from datetime import datetime, timezone

from app.analysis.analyzer import get_params
from app.backtesting.backtester import Backtester
from app.data_access.data_loader import load_data
from app.validation.bias_metrics import compare_execution_modes
from app.validation.utils import get_validation_ticker, get_validation_window
from app.validation.wf_analysis import run_walk_forward_analysis


def production_health_check() -> dict:
    ticker = get_validation_ticker()
    start, end = get_validation_window()

    df = load_data(ticker, start=start.isoformat(), end=end.isoformat())
    if df is None or df.empty:
        return {"status": "no_data", "ticker": ticker}

    df = df[(df.index.date >= start) & (df.index.date <= end)]
    if df.empty:
        return {"status": "no_data", "ticker": ticker}

    params = get_params(ticker)
    bt = Backtester(df, ticker)

    prod_report = bt.run(params, execution_policy="next_open")
    close_report = bt.run(params, execution_policy="close_to_close")

    relative_gap = compare_execution_modes(
        float(close_report.metrics.get("net_profit", 0.0)),
        float(prod_report.metrics.get("net_profit", 0.0)),
    ).get("relative_gap")

    wf = run_walk_forward_analysis()
    mean_gap = wf.get("mean_execution_gap")
    timing_dependency = isinstance(mean_gap, (int, float)) and mean_gap > 0.7

    status = {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "production_policy": "NEXT_OPEN",
        "relative_gap": relative_gap,
        "walk_forward": wf,
        "high_timing_dependency": timing_dependency,
    }

    return status
