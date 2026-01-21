import pandas as pd


def replay_decision(
    decision_event: dict,
    df: pd.DataFrame,
    horizon_days: int = 5,
) -> dict:
    """
    P8.0 – Deterministic decision replay

    Returns realized outcome of a past decision.
    """
    action = decision_event["action"]
    ts = pd.to_datetime(decision_event["timestamp"])

    # --- find entry index ---
    df_after = df[df.index > ts]
    if len(df_after) < horizon_days + 1:
        return {"status": "INSUFFICIENT_DATA"}

    entry_price = df_after.iloc[0]["Open"]
    exit_price = df_after.iloc[horizon_days]["Close"]

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
