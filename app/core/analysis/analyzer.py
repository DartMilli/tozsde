import numpy as np
import pandas as pd
import json
import logging

from app.bootstrap.build_settings import build_settings

# Avoid resolving settings at import time; injection required from bootstrap
settings = None


def _conf(provided=None):
    """Return a config-like object: prefer provided `settings`, else call package `get_settings()` at runtime.

    This avoids resolving settings at import time.
    """
    if provided is not None:
        return provided
    try:
        return build_settings()
    except Exception:
        return None


from app.core.indicators import trend, momentum, volatility
from app.infrastructure.logger import setup_logger
import app.indicators.technical as ta

# Back-compat: prefer technical indicators module for analyzer's `trend`/`momentum`/`volatility`
# so tests can monkeypatch `analyzer.ta` and affect internal indicator calls.
trend = ta
momentum = ta
volatility = ta

logger = setup_logger(__name__)

# Module-level settings proxy for diagnostics and other checks; prefer DI
# (module-level settings placeholder)


# Parameterek amiket erdemes finomhangolni...
# az uj ertekek 20 generacios 30-as populacioval (az utolso generacioban meg nott a fitness)
"""
[20] Legjobb egyen: [39, 34, 29, 18, 34, 5, 27, 2, 15, 5, 15, 4], fitnesz: 37.9173
gen     nevals  duration_sec    avg     std     min     max    
0       30      22836.2         21.1599 5.8969  8.26534 29.7593
1       13      9104.68         26.2295 2.75627 20.8482 30.1309
2       16      11156.1         27.6511 3.26469 19.4027 35.601
3       15      10597.1         29.6193 3.54901 22.2204 35.601
4       16      11297.8         31.9184 3.65792 20.6781 35.601
5       21      18131.4         34.2199 2.00149 27.5749 35.7082
6       15      13876.9         34.6462 2.35714 26.0112 36.5729
7       16      73162           34.7132 2.81783 22.0651 35.601
8       18      37257.7         35.2048 1.52047 27.8682 36.3474
9       19      43821.6         35.1291 1.92443 27.6605 37.3009
10      18      16086.1         35.1124 2.04422 29.0916 37.3009
11      9       6526.89         35.5073 2.76862 24.0967 37.3009
12      18      22607.7         36.131  2.67684 26.1916 37.3009
13      24      75935.5         37.0057 1.21875 30.8622 37.3009
14      22      42749.5         35.7059 4.55169 18.1408 37.3009
15      21      16039.6         36.2244 2.90199 23.1201 37.3009
16      18      32192.5         35.5802 3.65811 20.1974 37.8799
17      10      9179.93         36.7955 1.52216 30.4955 37.8799
18      20      40047.8         35.6835 3.75739 21.768  37.8799
19      22      23359.3         36.1872 2.89372 23.8714 37.8799
20      16      26084.8         35.9693 3.60994 19.9141 37.9173
"""
default_params = {
    "sma_period": 20,
    "ema_period": 10,
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bbands_period": 20,
    "bbands_stddev": 2.0,
    "atr_period": 14,
    "adx_period": 14,
    "stoch_k": 14,
    "stoch_d": 3,
    "use_sma": True,
    "use_ema": True,
    "use_rsi": True,
    "use_macd": True,
    "use_bbands": True,
    "use_atr": True,
    "use_adx": True,
    "use_stoch": True,
}

_PARAMS_CACHE = {"mtime": None, "data": None}

# Parameterek hatarai
param_bounds = {
    "sma_period": (5, 50),
    "ema_period": (5, 50),
    "rsi_period": (5, 30),
    "macd_fast": (5, 20),
    "macd_slow": (21, 50),
    "macd_signal": (5, 15),
    "bbands_period": (10, 50),
    "bbands_stddev": (1, 3),
    "atr_period": (5, 30),
    "adx_period": (5, 30),
    "stoch_k": (5, 20),
    "stoch_d": (3, 10),
}


def get_params(ticker):
    try:
        cfg = _conf()
        PARAMS_FILE = getattr(cfg, "PARAMS_FILE_PATH", None)
        if PARAMS_FILE is None:
            logging.warning("PARAMS_FILE_PATH not set in config; using default values.")
            return get_default_params()

        try:
            mtime = PARAMS_FILE.stat().st_mtime
        except Exception:
            mtime = None

        if _PARAMS_CACHE["data"] is None or _PARAMS_CACHE["mtime"] != mtime:
            with open(PARAMS_FILE, "r") as f:
                _PARAMS_CACHE["data"] = json.load(f)
                _PARAMS_CACHE["mtime"] = mtime

        params_all = _PARAMS_CACHE["data"] or {}
        if ticker not in params_all:
            logging.warning(
                f"No optimized parameters found for {ticker}; using default values."
            )
            return get_default_params()
        params = params_all[ticker]
        # Az stddev erteke lehet, hogy int-kent mentodott, de float kell
        if "bbands_stddev" in params:
            params["bbands_stddev"] = float(params["bbands_stddev"])
        return params
    except (FileNotFoundError, json.JSONDecodeError):
        logging.warning(
            "Optimized parameter file not found or invalid; using default values."
        )
        return get_default_params()


def _load_params_file():
    cfg = _conf()
    params_path = getattr(cfg, "PARAMS_FILE_PATH", None)
    if params_path is None:
        return {}
    try:
        with open(params_path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_params_for_ticker(ticker, params):
    if not isinstance(params, dict):
        logging.warning("Optimized params must be a dict; skipping save")
        return False

    cfg = _conf()
    params_path = getattr(cfg, "PARAMS_FILE_PATH", None)
    if params_path is None:
        logging.warning("PARAMS_FILE_PATH not set in config; skipping save")
        return False
    params_path.parent.mkdir(parents=True, exist_ok=True)

    params_all = _load_params_file()
    params_all[ticker] = params

    tmp_path = params_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(params_all, f, indent=2, sort_keys=True)
    tmp_path.replace(params_path)

    try:
        _PARAMS_CACHE["data"] = params_all
        _PARAMS_CACHE["mtime"] = params_path.stat().st_mtime
    except Exception:
        _PARAMS_CACHE["data"] = params_all
        _PARAMS_CACHE["mtime"] = None

    return True


def get_default_params():
    return default_params


def compute_signals(df, ticker, params, return_series=False, audit=None):
    """
    return_series = False  -> eredeti viselkedes (esemenylista)
    return_series = True   -> per-bar lista: signals[i] = [...signals...]
    """

    if params is None:
        params = get_params(ticker)
    else:
        # Ensure missing keys are filled from the saved/default params so
        # callers can pass partial param dicts (tests sometimes pass {'p':1}).
        try:
            base = get_params(ticker)
        except Exception:
            base = get_default_params()
        merged = dict(base)
        merged.update(params)
        params = merged

    # Biztonsagi ellenorzes ures DataFrame-re
    if df.empty:
        # Ha nincs adat, nincs mit elemezni
        empty_indicators = {
            "SMA": None,
            "EMA": None,
            "RSI": None,
            "MACD": None,
            "MACD_SIGNAL": None,
            "BB_upper": None,
            "BB_middle": None,
            "BB_lower": None,
            "ATR": None,
            "ADX": None,
            "PLUS_DI": None,
            "MINUS_DI": None,
            "STOCH_K": None,
            "STOCH_D": None,
        }
        return [], empty_indicators

    signals = []

    # A jelzes datumanak kinyerese a DataFrame utolso napja alapjan
    signal_date_str = df.index[-1].strftime("%Y-%m-%d")

    def is_valid(arr, n=2):
        return arr is not None and len(arr) >= n and not np.isnan(arr[-n:]).any()

    use_sma = params.get("use_sma", True)
    use_ema = params.get("use_ema", True)
    use_rsi = params.get("use_rsi", True)
    use_macd = params.get("use_macd", True)
    use_bbands = params.get("use_bbands", True)
    use_atr = params.get("use_atr", True)
    use_adx = params.get("use_adx", True)
    use_stoch = params.get("use_stoch", True)

    try:
        sma = trend.sma(df["Close"], period=params["sma_period"]) if use_sma else None
    except Exception:
        sma = None
    try:
        ema = trend.ema(df["Close"], period=params["ema_period"]) if use_ema else None
    except Exception:
        ema = None

    # EMA/SMA crossover
    if use_sma and use_ema and is_valid(sma) and is_valid(ema):
        if ema[-2] < sma[-2] and ema[-1] > sma[-1]:
            signals.append(f"BUY: EMA crossed above SMA on {signal_date_str}")
        elif ema[-2] > sma[-2] and ema[-1] < sma[-1]:
            signals.append(f"SELL: EMA crossed below SMA on {signal_date_str}")

    # RSI
    rsi = momentum.rsi(df["Close"], period=params["rsi_period"]) if use_rsi else None
    if use_rsi and is_valid(rsi):
        if rsi[-2] < 30 and rsi[-1] > 30:
            signals.append(f"BUY: RSI broke above 30 on {signal_date_str}")
        elif rsi[-2] > 70 and rsi[-1] < 70:
            signals.append(f"SELL: RSI broke below 70 on {signal_date_str}")

    # MACD
    macd, macdsignal = (None, None)
    if use_macd:
        try:
            macd, macdsignal = trend.macd(
                df["Close"],
                params["macd_fast"],
                params["macd_slow"],
                params["macd_signal"],
            )
        except Exception:
            macd, macdsignal = (None, None)
        if is_valid(macd) and is_valid(macdsignal):
            if macd[-2] < macdsignal[-2] and macd[-1] > macdsignal[-1]:
                signals.append(f"BUY: MACD crossover on {signal_date_str}")
            elif macd[-2] > macdsignal[-2] and macd[-1] < macdsignal[-1]:
                signals.append(f"SELL: MACD crossunder on {signal_date_str}")

    # Bollinger Bands
    upper, middle, lower = (None, None, None)
    if use_bbands:
        try:
            middle, upper, lower = volatility.bbands(
                df["Close"],
                period=params["bbands_period"],
                num_std=params["bbands_stddev"],
            )
        except Exception:
            middle, upper, lower = (None, None, None)
        if is_valid(upper, 1) and is_valid(lower, 1) and len(df["Close"]) > 0:
            close_val = df["Close"].iloc[-1]
            if not np.isnan(close_val):
                if close_val < lower[-1]:
                    signals.append(
                        f"BUY: Price below Bollinger Lower Band on {signal_date_str}"
                    )
                elif close_val > upper[-1]:
                    signals.append(
                        f"SELL: Price above Bollinger Upper Band on {signal_date_str}"
                    )

    # ATR
    try:
        atr = (
            volatility.atr(
                df["High"], df["Low"], df["Close"], period=params["atr_period"]
            )
            if use_atr
            else None
        )
    except Exception:
        atr = None
    if use_atr and is_valid(atr):
        if atr[-1] > atr[-2] * 1.5:
            signals.append(f"ALERT: ATR volatility spike on {signal_date_str}")

    # ADX
    adx, plus_di_vals, minus_di_vals = (None, None, None)
    if use_adx:
        try:
            adx, plus_di_vals, minus_di_vals = trend.adx(
                df["High"], df["Low"], df["Close"], period=params["adx_period"]
            )
        except Exception:
            adx, plus_di_vals, minus_di_vals = (None, None, None)
        if is_valid(adx, 1):
            if adx[-1] > 25:
                signals.append(
                    f"INFO: Strong trend detected (ADX > 25) on {signal_date_str}"
                )

    # STOCH
    slowk, slowd = (None, None)
    if use_stoch:
        try:
            slowk, slowd = momentum.stoch(
                df["High"],
                df["Low"],
                df["Close"],
                k_period=params["stoch_k"],
                d_period=params["stoch_d"],
            )
        except Exception:
            slowk, slowd = (None, None)
        if is_valid(slowk) and is_valid(slowd):
            if slowk[-2] < slowd[-2] and slowk[-1] > slowd[-1]:
                signals.append(f"BUY: Stochastic crossover on {signal_date_str}")
            elif slowk[-2] > slowd[-2] and slowk[-1] < slowd[-1]:
                signals.append(f"SELL: Stochastic crossunder on {signal_date_str}")

    indicators = {
        "SMA": sma,
        "EMA": ema,
        "RSI": rsi,
        "MACD": macd,
        "MACD_SIGNAL": macdsignal,
        "BB_upper": upper,
        "BB_middle": middle,
        "BB_lower": lower,
        "ATR": atr,
        "ADX": adx,
        "PLUS_DI": plus_di_vals,
        "MINUS_DI": minus_di_vals,
        "STOCH_K": slowk,
        "STOCH_D": slowd,
    }

    if return_series:
        # 1:1 signals-per-bar tomb epitese - egyszerusitett verzio
        # Indikatorok alapjan per-bar jeleket general
        signals_per_bar = []
        rolling_std = df["Close"].pct_change().rolling(window=20).std()

        if isinstance(audit, dict):
            audit.setdefault("raw_signal_count", 0)
            audit.setdefault("post_edge_filter_signal_count", 0)

        for i in range(len(df)):
            signal = "HOLD"  # Default

            # EMA/SMA crossover check
            if (
                use_sma
                and use_ema
                and i >= 1
                and sma is not None
                and len(sma) > i
                and not np.isnan(sma[i - 1 : i + 1]).any()
                and ema is not None
                and len(ema) > i
                and not np.isnan(ema[i - 1 : i + 1]).any()
            ):
                if ema[i - 1] < sma[i - 1] and ema[i] > sma[i]:
                    signal = "BUY"
                elif ema[i - 1] > sma[i - 1] and ema[i] < sma[i]:
                    signal = "SELL"

            raw_signal = signal

            # Signal quality filter based on expected edge
            if use_sma and use_ema and sma is not None and ema is not None:
                if (
                    i < len(sma)
                    and i < len(ema)
                    and not np.isnan(sma[i])
                    and sma[i] != 0
                ):
                    expected_edge = (ema[i] - sma[i]) / sma[i]
                else:
                    expected_edge = 0.0
                threshold = (
                    rolling_std.iloc[i] * 0.5 if i < len(rolling_std) else np.nan
                )
                if not np.isnan(threshold) and abs(expected_edge) < threshold:
                    signal = "HOLD"

                if (
                    isinstance(audit, dict)
                    and getattr(_conf(settings), "edge_diagnostics_mode", False)
                    and raw_signal in {"BUY", "SELL"}
                ):
                    if not np.isnan(threshold) and threshold > 0:
                        edge_values = audit.setdefault("edge_expected_edges", [])
                        edge_thresholds = audit.setdefault("edge_thresholds", [])
                        edge_values.append(float(abs(expected_edge)))
                        edge_thresholds.append(float(threshold))

            if isinstance(audit, dict):
                if raw_signal in {"BUY", "SELL"}:
                    audit["raw_signal_count"] += 1
                if signal in {"BUY", "SELL"}:
                    audit["post_edge_filter_signal_count"] += 1

            signals_per_bar.append(signal)

        return signals_per_bar, indicators

    return signals, indicators


if __name__ == "__main__":
    from app.data_access.data_loader import load_data, get_supported_ticker_list

    tickers = get_supported_ticker_list()

    for ticker in tickers:
        logger.info(f"Loading data for {ticker}...")
        df = load_data(ticker, start="2020-01-01", end="2025-06-30")
        signals, indicators = compute_signals(df, ticker, params=None)
        if signals:
            logger.info("Trading signals:\n%s", "\n".join(signals))
        else:
            logger.info("No new signal.")
