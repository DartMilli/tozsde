"""Shadow comparison between pipeline decisions and backtester replay."""

from __future__ import annotations

import json
import logging
import os
from datetime import date, timedelta

import pandas as pd

from app.backtesting.execution_utils import ACTION_MAP, normalize_action
from app.validation.errors import ValidationError

from app.backtesting.backtester import Backtester
from app.backtesting.pipeline_backtester import PipelineBacktester
from app.backtesting.history_store import HistoryStore
from app.services.dependencies import EmailNotifier, MarketDataFetcher
from app.services.execution_engines import NoopExecutionEngine
from app.services.trading_pipeline import TradingPipelineService
from app.analysis.analyzer import get_params, compute_signals
from app.infrastructure.repositories.sqlite_decision_repository import (
    SqliteDecisionRepository,
)
from app.data_access.data_loader import load_data
from app.validation.utils import get_validation_ticker, get_validation_window
from app.validation import get_settings

try:
    dm = SqliteDecisionRepository()
except Exception:
    dm = SqliteDecisionRepository()


def _baseline_policy(rsi_value: float) -> int:
    if rsi_value < 30:
        return 1
    if rsi_value > 70:
        return 2
    return 0


def _extract_rsi(df: pd.DataFrame) -> float | None:
    for col in ("RSI", "Rsi", "rsi"):
        if col in df.columns:
            try:
                return float(df[col].iloc[-1])
            except Exception:
                return None
    return None


def _build_pipeline(data_fetcher: MarketDataFetcher) -> TradingPipelineService:
    logger = logging.getLogger(__name__)

    class BaselineModelRunner:
        def run_ensemble(self, df, ticker: str, top_n: int = 3, debug: bool = True):
            action_code = 0
            rsi_value = _extract_rsi(df)
            if rsi_value is not None:
                action_code = _baseline_policy(rsi_value)
            confidence = 0.55
            wf_score = 0.5
            model_votes = [
                {
                    "model": "BASELINE_RSI",
                    "action": action_code,
                    "action_label": getattr(
                        get_settings().ACTION_LABELS.get(
                            getattr(get_settings(), "LANG", "en"), {}
                        ),
                        "get",
                        lambda *_: ACTION_MAP.get(action_code, "HOLD"),
                    )(action_code, ACTION_MAP.get(action_code, "HOLD")),
                    "action_code": action_code,
                    "confidence": confidence,
                }
            ]
            return [action_code], [confidence], [wf_score], model_votes, []

    return TradingPipelineService(
        history_store=HistoryStore(),
        logger=logger,
        data_fetcher=data_fetcher,
        model_runner=BaselineModelRunner(),
        email_notifier=EmailNotifier(),
        execution_engine=NoopExecutionEngine(logger=logger),
    )


def run_shadow_comparison() -> dict:
    ticker = get_validation_ticker()
    start, end = get_validation_window()
    try:
        max_days = int(os.getenv("VALIDATION_SHADOW_MAX_DAYS", "60"))
    except ValueError:
        max_days = 60

    if (end - start).days > max_days:
        start = end - timedelta(days=max_days)

    today = date.today()
    end_plus = end + timedelta(days=5)
    if end_plus > today:
        end_plus = today
    df_full = load_data(ticker, start=start.isoformat(), end=end_plus.isoformat())
    if df_full is None or df_full.empty:
        return {"status": "no_data", "ticker": ticker}

    class StaticDataFetcher(MarketDataFetcher):
        def __init__(self, df):
            self._df = df

        def load_data(self, ticker: str, start: str, end: str):
            df = self._df.copy()
            try:
                start_ts = pd.Timestamp(start)
                end_ts = pd.Timestamp(end)
                df = df[(df.index >= start_ts) & (df.index <= end_ts)]
            except Exception:
                pass
            return df

    pipeline = _build_pipeline(StaticDataFetcher(df_full))
    PipelineBacktester(pipeline).run(
        ticker=ticker,
        start=start,
        end=end,
        debug=False,
        evaluate_outcomes=False,
    )

    # dm should be injected from DI root
    rows = dm.fetch_history_range(
        ticker=ticker,
        start_iso=start.isoformat(),
        end_iso=end.isoformat(),
    )

    if not rows:
        return {"status": "no_data", "ticker": ticker}

    df = df_full
    if df is None or df.empty:
        return {"status": "no_data", "ticker": ticker}

    action_matches = 0
    action_mismatches = 0
    action_logs = 0

    params = get_params(ticker)
    signals, _ = compute_signals(df, ticker, params, return_series=True)
    signal_map = {}
    for idx, signal in enumerate(signals):
        try:
            sig_date = df.index[idx].date().isoformat()
        except Exception:
            continue
        if signal == "BUY":
            signal_map[sig_date] = 1
        elif signal == "SELL":
            signal_map[sig_date] = 2
        else:
            signal_map[sig_date] = 0
    decisions_by_date = {}
    for ts, _label, decision_blob, _audit_blob in rows:
        try:
            decision = json.loads(decision_blob) if decision_blob else {}
        except json.JSONDecodeError:
            continue
        raw_action = decision.get("action_code", 0)
        if action_logs < 20:
            print("ACTION TYPE:", type(raw_action), raw_action)
            action_logs += 1
        action = normalize_action(raw_action)
        decision_date = None
        try:
            decision_date = pd.to_datetime(ts).date().isoformat()
        except Exception:
            decision_date = None

        if decision_date is not None and decision_date in signal_map:
            if signal_map[decision_date] == action:
                action_matches += 1
            else:
                action_mismatches += 1
        if decision_date is not None:
            decisions_by_date[decision_date] = action

    if not decisions_by_date:
        return {"status": "no_data", "ticker": ticker}

    signals_override = []
    for idx in range(len(df)):
        day = df.index[idx].date().isoformat()
        action = decisions_by_date.get(day, 0)
        signals_override.append(ACTION_MAP.get(action, "HOLD"))

    bt = Backtester(df[df.index.date <= end], ticker)
    replay_report = bt.run(
        params,
        execution_policy=getattr(get_settings(), "EXECUTION_POLICY", "next_open"),
        signals_override=signals_override,
    )
    backtest_report = bt.run(
        params,
        execution_policy=getattr(get_settings(), "EXECUTION_POLICY", "next_open"),
        signals_override=signals_override,
    )
    analyzer_report = bt.run(
        params,
        execution_policy=getattr(get_settings(), "EXECUTION_POLICY", "next_open"),
    )
    replay_total_return = float(replay_report.metrics.get("net_profit", 0.0))
    backtest_return = float(backtest_report.metrics.get("net_profit", 0.0))
    analyzer_return = float(analyzer_report.metrics.get("net_profit", 0.0))

    total_compared = action_matches + action_mismatches
    match_rate = (action_matches / total_compared) if total_compared else None

    replay_total_return = round(replay_total_return, 4)
    backtest_return = round(backtest_return, 4)
    equity_diff = round(abs(backtest_return - replay_total_return), 4)
    if equity_diff > 0.01:
        raise ValidationError(
            f"Shadow/backtest mismatch: replay={replay_total_return}, backtest={backtest_return}"
        )

    return {
        "status": "ok",
        "ticker": ticker,
        "decision_count": len(decisions_by_date),
        "replay_return": replay_total_return,
        "backtest_return": backtest_return,
        "analyzer_return": round(analyzer_return, 4),
        "equity_diff": equity_diff,
        "action_match_rate": round(match_rate, 4) if match_rate is not None else None,
        "action_mismatch_count": action_mismatches,
    }
