import numpy as np
import pandas as pd

from app.backtesting.execution_utils import normalize_action, resolve_execution_price


def replay_decision(
    decision_event: dict,
    df: pd.DataFrame,
    horizon_days: int = 5,
    execution_policy: str = "next_open",
) -> dict:
    """
    P8.0 - Deterministic decision replay

    Returns realized outcome of a past decision.
    """
    action = normalize_action(decision_event.get("action"))
    ts = pd.to_datetime(decision_event.get("timestamp")).date()

    idx_matches = df.index.date == ts
    if not idx_matches.any():
        return {"status": "INSUFFICIENT_DATA"}

    idx = int(np.flatnonzero(idx_matches)[0])
    policy = (execution_policy or "next_open").lower()
    base_idx = idx + (1 if policy == "next_open" else 0)
    if base_idx + horizon_days >= len(df):
        return {"status": "INSUFFICIENT_DATA"}

    entry_price = resolve_execution_price(df, idx, execution_policy)
    if entry_price is None:
        return {"status": "INSUFFICIENT_DATA"}

    exit_price = float(df["Close"].iloc[base_idx + horizon_days])

    if action == 1:  # BUY
        raw_return = (exit_price - entry_price) / entry_price
    elif action == 2:  # SELL
        raw_return = (entry_price - exit_price) / entry_price
    else:  # HOLD
        raw_return = 0.0

    return {
        "status": "OK",
        "entry_price": round(entry_price, 4),
        "exit_price": round(exit_price, 4),
        "raw_return": round(raw_return, 4),
        "horizon_days": horizon_days,
    }
