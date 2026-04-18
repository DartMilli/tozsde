import numpy as np
import pandas as pd

from app.core.indicators import cached_result, fingerprint_multi


def bbands(data, period=20, std_dev=2):
    data = np.asarray(data, dtype=float)
    fingerprint = fingerprint_multi(data)
    key = ("bbands", fingerprint, period, std_dev)

    def _compute():
        mid = np.convolve(data, np.ones(period) / period, mode="valid")
        mid = np.concatenate([np.full(period - 1, np.nan), mid])
        series = pd.Series(data)
        std = series.rolling(window=period, min_periods=period).std().values
        upper = mid + std_dev * std
        lower = mid - std_dev * std
        return upper, mid, lower

    return cached_result(key, _compute)


def atr(high, low, close, period=14):
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    n = len(close)
    fingerprint = fingerprint_multi(high, low, close)
    key = ("atr", fingerprint, period)

    def _compute():
        prev_close = np.concatenate([[np.nan], close[:-1]])
        tr = np.maximum.reduce(
            [
                high - low,
                np.abs(high - prev_close),
                np.abs(low - prev_close),
            ]
        )
        tr[0] = high[0] - low[0]

        # Wilder smoothing: seed with SMA of first `period` bars
        atr_v = np.full(n, np.nan)
        if n < period:
            return atr_v

        atr_v[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr_v[i] = (atr_v[i - 1] * (period - 1) + tr[i]) / period

        return atr_v

    return cached_result(key, _compute)
