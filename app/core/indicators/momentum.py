import numpy as np

from app.core.indicators import cached_result, fingerprint_multi


def rsi(data, period=14):
    data = np.asarray(data, dtype=float)
    fingerprint = fingerprint_multi(data)
    key = ("rsi", fingerprint, period)

    def _compute():
        n = len(data)
        if n < period + 1:
            return np.full(n, np.nan)

        deltas = np.diff(data)
        ups = np.where(deltas > 0, deltas, 0.0)
        downs = np.where(deltas < 0, -deltas, 0.0)

        # Wilder smoothing: seed with SMA of first `period` deltas
        avg_up = np.full(n, np.nan)
        avg_down = np.full(n, np.nan)

        avg_up[period] = np.mean(ups[:period])
        avg_down[period] = np.mean(downs[:period])

        for i in range(period, len(deltas)):
            avg_up[i + 1] = (avg_up[i] * (period - 1) + ups[i]) / period
            avg_down[i + 1] = (avg_down[i] * (period - 1) + downs[i]) / period

        with np.errstate(divide="ignore", invalid="ignore"):
            rs = avg_up / (avg_down + 1e-12)
            rsi_val = 100.0 - (100.0 / (1.0 + rs))

        return rsi_val

    return cached_result(key, _compute)


def stoch(high, low, close, k_period=14, d_period=3):
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    fingerprint = fingerprint_multi(high, low, close)
    key = ("stoch", fingerprint, k_period, d_period)

    def _compute():
        lowest_low = np.array(
            [
                np.min(low[i - k_period + 1 : i + 1]) if i >= k_period - 1 else np.nan
                for i in range(len(low))
            ]
        )
        highest_high = np.array(
            [
                np.max(high[i - k_period + 1 : i + 1]) if i >= k_period - 1 else np.nan
                for i in range(len(high))
            ]
        )
        k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        d = np.convolve(k[~np.isnan(k)], np.ones(d_period) / d_period, mode="valid")
        d_full = np.concatenate([np.full(len(k) - len(d), np.nan), d])
        return k, d_full

    return cached_result(key, _compute)
