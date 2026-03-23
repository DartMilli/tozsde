"""Pipeline audit utilities for diagnosing trade collapse."""

from __future__ import annotations

from app.analysis.analyzer import get_params
from app.backtesting.backtester import Backtester
from app.validation import get_settings
from app.data_access.data_loader import load_data
from app.infrastructure.logger import setup_logger
from app.validation.data_integrity_check import run_data_integrity_checks
from app.validation.edge_diagnostics import (
    compute_fold_edge_stats,
    compute_global_summary,
)
from app.validation.utils import get_validation_ticker, get_validation_window

logger = setup_logger(__name__)


# Settings are injected at runtime via `set_settings()`; avoid resolving at import time
settings = None


def _lookback_from_params(params: dict) -> int:
    return max(
        [
            params.get("sma_period", 0),
            params.get("ema_period", 0),
            params.get("rsi_period", 0),
            params.get("macd_slow", 0) + params.get("macd_signal", 0),
            params.get("bbands_period", 0),
            params.get("atr_period", 0),
            params.get("adx_period", 0) * 2 + 1,
            params.get("stoch_k", 0) + params.get("stoch_d", 0),
        ]
    )


def _audit_fold(df, ticker: str, params: dict, fold_id: int) -> dict:
    cfg = get_settings()
    lookback = _lookback_from_params(params)
    raw_rows = len(df)
    cleaned = df.dropna(
        subset=[c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    )
    nan_rows_removed = raw_rows - len(cleaned)
    usable_rows = max(0, len(cleaned) - lookback)

    audit = {
        "fold_id": fold_id,
        "raw_rows": raw_rows,
        "usable_rows": usable_rows,
        "nan_rows_removed": nan_rows_removed,
        "lookback": lookback,
        "raw_signal_count": 0,
        "post_dropout_signal_count": 0,
        "post_edge_filter_signal_count": 0,
        "position_attempts": 0,
        "orders_created": 0,
        "executed_trades": 0,
        "capital_start": float(cfg.INITIAL_CAPITAL),
        "capital_end": float(cfg.INITIAL_CAPITAL),
        "size_zero_count": 0,
        "rejected_orders": 0,
        "slice_start": df.index[0].isoformat() if raw_rows else None,
        "slice_end": df.index[-1].isoformat() if raw_rows else None,
    }

    if cfg.EDGE_DIAGNOSTICS_MODE:
        audit["edge_expected_edges"] = []
        audit["edge_thresholds"] = []

    if raw_rows == 0:
        return audit

    backtester = Backtester(cleaned, ticker)
    backtester.run(params, execution_policy="next_open", audit=audit)

    audit["position_attempts"] = int(audit.get("post_edge_filter_signal_count", 0))
    audit["rejected_orders"] = max(
        0, audit.get("orders_created", 0) - audit.get("executed_trades", 0)
    )

    if cfg.EDGE_DIAGNOSTICS_MODE:
        audit["edge_distribution"] = compute_fold_edge_stats(
            fold_id=fold_id,
            raw_signal_count=int(audit.get("raw_signal_count", 0)),
            expected_edges=audit.get("edge_expected_edges", []),
            thresholds=audit.get("edge_thresholds", []),
        )

    return audit


def run_pipeline_audit(ticker: str | None = None, params: dict | None = None) -> dict:
    ticker = ticker or get_validation_ticker()
    params = params or get_params(ticker)
    start, end = get_validation_window()

    df = load_data(ticker, start=start.isoformat(), end=end.isoformat())
    if df is None or df.empty:
        return {"status": "no_data", "ticker": ticker, "folds": []}

    lookback = _lookback_from_params(params)
    integrity = run_data_integrity_checks(df, lookback=lookback)

    days_per_month = 21
    cfg = get_settings()
    train_window = int(cfg.TRAIN_WINDOW_MONTHS * days_per_month)
    test_window = int(cfg.TEST_WINDOW_MONTHS * days_per_month)
    step_size = int(cfg.WINDOW_STEP_MONTHS * days_per_month)

    folds = []
    start_idx = train_window
    end_idx = len(df) - test_window
    fold_id = 0
    while start_idx <= end_idx:
        test_df = df.iloc[start_idx : start_idx + test_window]
        folds.append(_audit_fold(test_df, ticker, params, fold_id))
        fold_id += 1
        start_idx += step_size

    edge_diagnostics_summary = None
    if cfg.EDGE_DIAGNOSTICS_MODE:
        distributions = [
            f.get("edge_distribution")
            for f in folds
            if isinstance(f.get("edge_distribution"), dict)
        ]
        edge_diagnostics_summary = compute_global_summary(distributions)

    payload = {
        "status": "ok",
        "ticker": ticker,
        "folds": folds,
        "data_integrity": integrity,
        "edge_diagnostics_summary": edge_diagnostics_summary,
        "edge_distributions": distributions if cfg.EDGE_DIAGNOSTICS_MODE else None,
    }
    return payload
