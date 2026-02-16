"""Risk stress adapter using historical drawdown metrics and sizing stress."""

from __future__ import annotations

import pandas as pd

from app.config.config import Config
from app.data_access.data_loader import load_data
from app.decision.position_sizer import PositionSizer
from app.validation.metrics_core import compute_max_drawdown
from app.validation.utils import get_validation_ticker, get_validation_window


def run_risk_stress_tests() -> dict:
    ticker = get_validation_ticker()
    start, end = get_validation_window()

    df = load_data(ticker, start=start.isoformat(), end=end.isoformat())
    if df is None or df.empty:
        return {"status": "no_data", "ticker": ticker}

    df = df[(df.index.date >= start) & (df.index.date <= end)]
    if df.empty:
        return {"status": "no_data", "ticker": ticker}

    equity = pd.Series(df["Close"].astype(float).values, index=df.index)
    max_dd = compute_max_drawdown(equity)

    sizing_stress = _run_position_sizing_stress()
    bear_windows = _run_bear_market_drawdowns(ticker)

    return {
        "status": "ok",
        "ticker": ticker,
        "crash_drawdown": round(float(max_dd), 6),
        "position_sizing": sizing_stress,
        "bear_windows": bear_windows,
    }


def _run_position_sizing_stress() -> dict:
    sizer = PositionSizer()
    scenarios = [
        {"equity": Config.INITIAL_CAPITAL * 0.2, "confidence": 0.3, "wf": 0.4},
        {"equity": Config.INITIAL_CAPITAL, "confidence": 0.6, "wf": 0.7},
        {"equity": Config.INITIAL_CAPITAL * 10, "confidence": 0.9, "wf": 0.9},
    ]
    results = []
    violations = 0

    for scenario in scenarios:
        base_size = scenario["equity"] * 0.2
        res = sizer.compute(
            base_position_size=base_size,
            confidence=scenario["confidence"],
            wf_score=scenario["wf"],
            safety_discount=0.0,
            equity=scenario["equity"],
        )
        allocation_pct = (
            res.final_size / scenario["equity"] if scenario["equity"] else 0
        )
        if allocation_pct > Config.P6_POSITION_MAX_PCT + 1e-6:
            violations += 1
        results.append(
            {
                "equity": round(float(scenario["equity"]), 2),
                "confidence": scenario["confidence"],
                "wf_score": scenario["wf"],
                "final_size": res.final_size,
                "allocation_pct": round(float(allocation_pct), 4),
                "capped": res.capped,
            }
        )

    return {"scenarios": results, "violations": violations}


def _run_bear_market_drawdowns(ticker: str) -> dict:
    windows = [
        ("2008", "2008-01-01", "2009-12-31"),
        ("2020", "2020-01-01", "2020-12-31"),
        ("2022", "2022-01-01", "2022-12-31"),
    ]
    out = {}
    for label, start, end in windows:
        df = load_data(ticker, start=start, end=end)
        if df is None or df.empty:
            out[label] = {"status": "no_data"}
            continue
        series = pd.Series(df["Close"].astype(float).values, index=df.index)
        out[label] = {
            "status": "ok",
            "max_drawdown": round(float(compute_max_drawdown(series)), 6),
        }
    return out
