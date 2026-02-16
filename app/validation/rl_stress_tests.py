"""RL stability adapter based on model outcome dispersion."""

from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
import pandas as pd

from pathlib import Path

from app.analysis.analyzer import get_params
from app.backtesting.backtester import Backtester
from app.config.config import Config
from app.data_access.data_loader import load_data
from app.data_access.data_manager import DataManager
from app.validation.metrics_core import compute_sharpe
from app.validation.utils import get_validation_ticker, get_validation_window


def _run_period_reversal_test(ticker: str) -> dict:
    periods = [
        ("2018-01-01", "2021-12-31"),
        ("2022-01-01", "2023-12-31"),
    ]

    params = get_params(ticker)
    results = []
    for start, end in periods:
        df = load_data(ticker, start=start, end=end)
        if df is None or df.empty:
            results.append({"status": "no_data", "start": start, "end": end})
            continue
        report = Backtester(df, ticker).run(
            params,
            execution_policy=Config.EXECUTION_POLICY,
        )
        results.append(
            {
                "status": "ok",
                "start": start,
                "end": end,
                "net_profit": float(report.metrics.get("net_profit", 0.0)),
                "sharpe": float(report.diagnostics.get("sharpe", 0.0)),
                "trade_count": int(report.metrics.get("trade_count", 0)),
            }
        )

    return {"status": "ok", "periods": results, "proxy": True}


def _run_noise_injection_test(ticker: str) -> dict:
    start, end = get_validation_window()
    params = get_params(ticker)
    df = load_data(ticker, start=start.isoformat(), end=end.isoformat())
    if df is None or df.empty:
        return {"status": "no_data"}

    report = Backtester(df, ticker).run(
        params, execution_policy=Config.EXECUTION_POLICY
    )
    base_return = float(report.metrics.get("net_profit", 0.0))

    noise = np.random.normal(0.0, 0.05, size=len(df))
    noisy = df.copy()
    for col in ("Open", "High", "Low", "Close"):
        if col in noisy.columns:
            noisy[col] = noisy[col] * (1.0 + noise)

    noisy_report = Backtester(noisy, ticker).run(
        params,
        execution_policy=Config.EXECUTION_POLICY,
    )
    noisy_return = float(noisy_report.metrics.get("net_profit", 0.0))

    return {
        "status": "ok",
        "base_return": base_return,
        "noisy_return": noisy_return,
        "drop": base_return - noisy_return,
        "proxy": True,
    }


def run_rl_stress_tests() -> dict:
    ticker = get_validation_ticker()
    start, end = get_validation_window()
    dm = DataManager()

    model_dir = Path(Config.MODEL_DIR)
    model_glob = f"*{ticker}*.zip"
    if not model_dir.exists() or not any(model_dir.glob(model_glob)):
        return {"status": "NO_MODEL", "sharpe_std": None}

    query = """
        SELECT dh.model_votes_json, o.pnl_pct
        FROM decision_history dh
        JOIN outcomes o ON o.decision_id = dh.id
        WHERE dh.ticker = ? AND date(dh.timestamp) BETWEEN date(?) AND date(?)
    """
    with dm.connection() as conn:
        rows = conn.execute(
            query, (ticker, start.isoformat(), end.isoformat())
        ).fetchall()

    if not rows:
        return {"status": "no_data", "ticker": ticker}

    per_model: Dict[str, List[float]] = {}
    for model_votes_json, pnl_pct in rows:
        votes = json.loads(model_votes_json) if model_votes_json else []
        for mv in votes:
            model_path = mv.get("model_path")
            if not model_path:
                continue
            per_model.setdefault(model_path, []).append(float(pnl_pct or 0.0))

    if not per_model:
        return {"status": "no_data", "ticker": ticker}

    sharpe_vals = []
    for model_path, returns in per_model.items():
        series = pd.Series(returns, dtype=float)
        if series.empty:
            continue
        sharpe_vals.append(compute_sharpe(series))

    sharpe_std = float(np.std(sharpe_vals)) if sharpe_vals else None
    sharpe_mean = float(np.mean(sharpe_vals)) if sharpe_vals else None

    period_reversal = _run_period_reversal_test(ticker)
    noise_injection = _run_noise_injection_test(ticker)

    return {
        "status": "ok",
        "ticker": ticker,
        "models": len(per_model),
        "sharpe_std": round(sharpe_std, 6) if sharpe_std is not None else None,
        "sharpe_mean": round(sharpe_mean, 6) if sharpe_mean is not None else None,
        "period_reversal": period_reversal,
        "noise_injection": noise_injection,
    }
