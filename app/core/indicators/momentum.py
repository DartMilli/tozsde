import numpy as np

from app.core.indicators import cached_result, fingerprint_multi


def rsi(data, period=14):
    data = np.asarray(data, dtype=float)
    fingerprint = fingerprint_multi(data)
    key = ("rsi", fingerprint, period)

    def _compute():
        if len(data) < period:
            return np.full(len(data), np.nan)

        deltas = np.diff(data)
        ups = np.where(deltas > 0, deltas, 0.0)
        downs = np.where(deltas < 0, -deltas, 0.0)
        roll_up = np.convolve(ups, np.ones(period) / period, mode="full")[: len(deltas)]
        roll_down = np.convolve(downs, np.ones(period) / period, mode="full")[
            : len(deltas)
        ]
        rs = np.full(len(data), np.nan)
        avg_up = np.concatenate([np.full(period, np.nan), roll_up[period - 1 :]])
        avg_down = np.concatenate([np.full(period, np.nan), roll_down[period - 1 :]])
        with np.errstate(divide="ignore", invalid="ignore"):
            rs_val = avg_up / (avg_down + 1e-12)
            rsi_val = 100 - (100 / (1 + rs_val))
        rs[: len(rsi_val)] = rsi_val
        return rs

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
