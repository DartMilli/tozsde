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
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

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
        plus_di = np.full(n, np.nan)
        minus_di = np.full(n, np.nan)

        if n < period + 1:
            return adx_v, plus_di, minus_di

        # True Range (first bar: no prev_close)
        prev_close = np.concatenate([[np.nan], close[:-1]])
        tr = np.maximum.reduce(
            [
                high - low,
                np.abs(high - prev_close),
                np.abs(low - prev_close),
            ]
        )
        tr[0] = high[0] - low[0]

        # Directional Movement
        up_move = np.concatenate([[0.0], high[1:] - high[:-1]])
        down_move = np.concatenate([[0.0], low[:-1] - low[1:]])
        plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)

        # Wilder smoothing: seed with sum of first `period` values
        atr_w = np.full(n, np.nan)
        plus_w = np.full(n, np.nan)
        minus_w = np.full(n, np.nan)

        atr_w[period] = np.sum(tr[1 : period + 1])
        plus_w[period] = np.sum(plus_dm[1 : period + 1])
        minus_w[period] = np.sum(minus_dm[1 : period + 1])

        for i in range(period + 1, n):
            atr_w[i] = atr_w[i - 1] - atr_w[i - 1] / period + tr[i]
            plus_w[i] = plus_w[i - 1] - plus_w[i - 1] / period + plus_dm[i]
            minus_w[i] = minus_w[i - 1] - minus_w[i - 1] / period + minus_dm[i]

        # +DI, -DI, DX
        with np.errstate(divide="ignore", invalid="ignore"):
            plus_di = 100.0 * plus_w / (atr_w + 1e-12)
            minus_di = 100.0 * minus_w / (atr_w + 1e-12)
            dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12)

        # ADX = Wilder smoothed DX (seeds at 2*period)
        adx_start = 2 * period
        if adx_start >= n:
            return adx_v, plus_di, minus_di

        adx_v[adx_start] = np.nanmean(dx[period : adx_start + 1])
        for i in range(adx_start + 1, n):
            adx_v[i] = (adx_v[i - 1] * (period - 1) + dx[i]) / period

        return adx_v, plus_di, minus_di

    return cached_result(key, _compute)
