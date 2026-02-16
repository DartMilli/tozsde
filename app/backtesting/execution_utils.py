"""Shared execution helpers for backtest and shadow replay."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

ACTION_MAP = {
    0: "HOLD",
    1: "BUY",
    2: "SELL",
}

ACTION_CODE_MAP = {v: k for k, v in ACTION_MAP.items()}


def seed_deterministic(seed: int = 42) -> None:
    np.random.seed(seed)
    if torch is not None:
        try:
            torch.manual_seed(seed)
        except Exception:
            pass


def normalize_action(action: Any) -> int:
    if action is None:
        return 0

    if torch is not None and isinstance(action, torch.Tensor):
        try:
            action = action.item()
        except Exception:
            action = action.detach().cpu().item()

    if isinstance(action, (np.integer,)):
        action = int(action)
    elif isinstance(action, (int,)):
        action = int(action)
    elif isinstance(action, float) and action.is_integer():
        action = int(action)
    elif isinstance(action, str):
        action = action.strip().upper()
        return ACTION_CODE_MAP.get(action, 0)

    if isinstance(action, int):
        return action if action in ACTION_MAP else 0

    return 0


def resolve_execution_price(df, idx: int, execution_policy: str):
    execution_policy = (execution_policy or "next_open").lower()
    if execution_policy not in {"close_to_close", "next_open"}:
        execution_policy = "next_open"

    if execution_policy == "close_to_close":
        return float(df["Close"].iloc[idx])

    next_idx = idx + 1
    if next_idx >= len(df):
        return None
    if "Open" in df.columns:
        return float(df["Open"].iloc[next_idx])
    return float(df["Close"].iloc[next_idx])
