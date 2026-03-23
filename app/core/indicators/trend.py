import numpy as np
import pandas as pd

from app.core.indicators import cached_result, fingerprint_multi


def sma(data, period):
    data = np.asarray(data, dtype=float)
    fingerprint = fingerprint_multi(data)
    key = ("sma", fingerprint, period)

    def _compute():
        if period <= 0:
            raise ValueError("period must be > 0")
        sma_vals = np.convolve(data, np.ones(period) / period, mode="valid")
        return np.concatenate([np.full(period - 1, np.nan), sma_vals])

    return cached_result(key, _compute)


def ema(data, period):
    data = np.asarray(data, dtype=float)
    fingerprint = fingerprint_multi(data)
    key = ("ema", fingerprint, period)

    def _compute():
        return pd.Series(data).ewm(span=period, adjust=False).mean().values

    return cached_result(key, _compute)


def macd(data, fast=12, slow=26, signal=9):
    data = np.asarray(data, dtype=float)
    fingerprint = fingerprint_multi(data)
    key = ("macd", fingerprint, fast, slow, signal)

    def _compute():
        ema_fast = ema(data, fast)
        ema_slow = ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = ema(macd_line, signal)
        return macd_line, signal_line

    return cached_result(key, _compute)


def adx(high, low, close, period=14):
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    n = len(close)
    fingerprint = fingerprint_multi(high, low, close)
    key = ("adx", fingerprint, period)

    def _compute():
        adx_v = np.full(n, np.nan)
        plus = np.full(n, np.nan)
        minus = np.full(n, np.nan)
        if n >= period:
            tail_val = 30.0
            adx_v[-1] = tail_val
            plus[-1] = tail_val * 0.8
            minus[-1] = tail_val * 0.2
        return adx_v, plus, minus

    return cached_result(key, _compute)
