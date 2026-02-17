"""Data integrity checks for pipeline diagnostics."""

from __future__ import annotations

from typing import Optional

import numpy as np

from app.config.config import Config
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)


def _count_streaks(values: list[bool], min_len: int) -> int:
    streaks = 0
    current = 0
    for flag in values:
        if flag:
            current += 1
        else:
            if current >= min_len:
                streaks += 1
            current = 0
    if current >= min_len:
        streaks += 1
    return streaks


def run_data_integrity_checks(
    df,
    lookback: int,
    minimum_required_rows: Optional[int] = None,
    streak_min: int = 5,
) -> dict:
    total_rows = len(df)
    duplicate_count = int(df.index.duplicated().sum()) if total_rows else 0
    monotonic_increasing = (
        bool(df.index.is_monotonic_increasing) if total_rows else True
    )

    ohlc_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    nan_ohlc_rows = int(df[ohlc_cols].isna().any(axis=1).sum()) if ohlc_cols else 0

    zero_volume_streaks = 0
    if "Volume" in df.columns:
        zero_volume_streaks = _count_streaks(
            [float(v) == 0.0 for v in df["Volume"].fillna(0.0).tolist()],
            streak_min,
        )

    constant_price_streaks = 0
    if "Close" in df.columns:
        close_vals = df["Close"].astype(float).tolist()
        constant_flags = [False]
        for idx in range(1, len(close_vals)):
            constant_flags.append(close_vals[idx] == close_vals[idx - 1])
        constant_price_streaks = _count_streaks(constant_flags, streak_min)

    gap_ratio = 0.0
    if total_rows > 1:
        diffs = df.index.to_series().diff().dt.days.fillna(0)
        gap_count = int((diffs > 3).sum())
        gap_ratio = float(gap_count / max(1, total_rows - 1))

    cleaned_rows = max(0, total_rows - nan_ohlc_rows)
    usable_rows_after_lookback = max(0, cleaned_rows - max(0, lookback))

    if minimum_required_rows is None:
        minimum_required_rows = lookback + 5

    warnings = []
    if usable_rows_after_lookback < minimum_required_rows:
        warnings.append("usable_rows_below_minimum")
        logger.warning(
            "Data integrity warning: usable_rows_after_lookback=%s below minimum=%s",
            usable_rows_after_lookback,
            minimum_required_rows,
        )

    payload = {
        "total_rows": int(total_rows),
        "monotonic_increasing": monotonic_increasing,
        "duplicate_index_count": int(duplicate_count),
        "nan_ohlc_rows": int(nan_ohlc_rows),
        "zero_volume_streaks": int(zero_volume_streaks),
        "constant_price_streaks": int(constant_price_streaks),
        "gap_ratio": float(gap_ratio),
        "usable_rows_after_lookback": int(usable_rows_after_lookback),
        "warnings": warnings,
    }

    return payload
