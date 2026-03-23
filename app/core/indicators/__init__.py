import hashlib
from collections import OrderedDict

import numpy as np


_CACHE_MAX = 128
_indicator_cache = OrderedDict()

technical_indicators_summary = {
    "sma": "Simple moving average",
    "ema": "Exponential moving average",
    "rsi": "Relative Strength Index",
    "macd": "MACD line and signal",
    "bbands": "Bollinger Bands (middle, upper, lower)",
    "atr": "Average True Range",
    "stoch": "Stochastic oscillator (K and D)",
    "adx": "Average Directional Index",
}


def fingerprint_multi(*arrays):
    hasher = hashlib.blake2b(digest_size=8)
    total_len = 0
    for arr in arrays:
        arr = np.asarray(arr, dtype=float)
        total_len += arr.shape[0]
        hasher.update(arr.view(np.uint8))
    return total_len, hasher.hexdigest()


def cached_result(key, compute_fn):
    cached = _indicator_cache.get(key)
    if cached is not None:
        _indicator_cache.move_to_end(key)
        return cached

    value = compute_fn()
    _indicator_cache[key] = value
    _indicator_cache.move_to_end(key)
    if len(_indicator_cache) > _CACHE_MAX:
        _indicator_cache.popitem(last=False)
    return value


def get_indicator_description():
    out = {}
    for key, value in technical_indicators_summary.items():
        out[key] = value
        out[key.upper()] = value
    return out
