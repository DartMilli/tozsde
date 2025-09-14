import app.utils.technical_analizer as ta
import numpy as np
import pandas as pd
import json

import app.utils.router as rtr
from app.utils.data_cleaner import sanitize_dataframe


# Paraméterek amiket érdemes finomhangolni...
# az új értékek 20 generációs 30-as populációval (az utolsó generációban még nőtt a fitness)
"""
[20] Legjobb egyén: [39, 34, 29, 18, 34, 5, 27, 2, 15, 5, 15, 4], fitnesz: 37.9173
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
    "bbands_stddev": 2,
    "atr_period": 14,
    "adx_period": 14,
    "stoch_k": 14,
    "stoch_d": 3,
}

# Paraméterek határai
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


def get_params():
    try:
        PARAMS_FILE = rtr.PARAMS_FILE_PATH

        with open(PARAMS_FILE, "r") as f:
            params = json.load(f)
        # Az stddev értéke lehet, hogy int-ként mentődött, de float kell
        if "bbands_stddev" in params:
            params["bbands_stddev"] = float(params["bbands_stddev"])
        return params
    except (FileNotFoundError, json.JSONDecodeError):
        print(
            "[WARN] Optimalizált paraméterfájl nem található, alapértelmezett értékek használata."
        )
        return get_default_params()


def get_default_params():
    return default_params


# A meglévő e-mail küldő metódusodat használjuk:
def send_email(subject: str, body: str):
    print(f"[EMAIL] {subject}\n{body}\n")  # Debug célra terminálra írjuk most


def compute_signals(df, params=None):
    if params is None:
        params = get_params()

    # Biztonsági ellenőrzés üres DataFrame-re
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

    # A jelzés dátumának kinyerése a DataFrame utolsó napja alapján
    signal_date_str = df.index[-1].strftime("%Y-%m-%d")

    def is_valid(arr, n=2):
        return arr is not None and len(arr) >= n and not np.isnan(arr[-n:]).any()

    # EMA/SMA crossover
    sma = ta.sma(df["Close"], period=params["sma_period"])
    ema = ta.ema(df["Close"], period=params["ema_period"])
    if is_valid(sma) and is_valid(ema):
        if ema[-2] < sma[-2] and ema[-1] > sma[-1]:
            # <<< VÁLTOZTATÁS: Dátum hozzáadása a jelzéshez
            signals.append(f"BUY: EMA crossed above SMA on {signal_date_str}")
        elif ema[-2] > sma[-2] and ema[-1] < sma[-1]:
            # <<< VÁLTOZTATÁS: Dátum hozzáadása a jelzéshez
            signals.append(f"SELL: EMA crossed below SMA on {signal_date_str}")

    # RSI
    rsi = ta.rsi(df["Close"], period=params["rsi_period"])
    if is_valid(rsi):
        if rsi[-2] < 30 and rsi[-1] > 30:
            signals.append(f"BUY: RSI broke above 30 on {signal_date_str}")
        elif rsi[-2] > 70 and rsi[-1] < 70:
            signals.append(f"SELL: RSI broke below 70 on {signal_date_str}")

    # MACD
    macd, macdsignal = ta.macd(
        df["Close"], params["macd_fast"], params["macd_slow"], params["macd_signal"]
    )
    if is_valid(macd) and is_valid(macdsignal):
        if macd[-2] < macdsignal[-2] and macd[-1] > macdsignal[-1]:
            signals.append(f"BUY: MACD crossover on {signal_date_str}")
        elif macd[-2] > macdsignal[-2] and macd[-1] < macdsignal[-1]:
            signals.append(f"SELL: MACD crossunder on {signal_date_str}")

    # Bollinger Bands
    upper, middle, lower = ta.bbands(
        df["Close"], period=params["bbands_period"], std_dev=params["bbands_stddev"]
    )
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
    atr = ta.atr(df["High"], df["Low"], df["Close"], period=params["atr_period"])
    if is_valid(atr):
        if atr[-1] > atr[-2] * 1.5:
            signals.append(f"ALERT: ATR volatility spike on {signal_date_str}")

    # ADX
    adx, plus_di_vals, minus_di_vals = ta.adx(
        df["High"], df["Low"], df["Close"], period=params["adx_period"]
    )
    if is_valid(adx, 1):
        if adx[-1] > 25:
            signals.append(
                f"INFO: Strong trend detected (ADX > 25) on {signal_date_str}"
            )

    # STOCH
    slowk, slowd = ta.stoch(
        df["High"],
        df["Low"],
        df["Close"],
        k_period=params["stoch_k"],
        d_period=params["stoch_d"],
    )
    if is_valid(slowk) and is_valid(slowd):
        if slowk[-2] < slowd[-2] and slowk[-1] > slowd[-1]:
            signals.append(f"BUY: Stochastic crossover on {signal_date_str}")
        elif slowk[-2] > slowd[-2] and slowk[-1] < slowd[-1]:
            signals.append(f"SELL: Stochastic crossunder on {signal_date_str}")

    return signals, {
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


def run_signal_backtest(ticker, start_date, end_date, initial_capital=10000):
    df = load_data(ticker, start=start_date, end=end_date)
    df = sanitize_dataframe(df)
    params = get_params()

    capital = initial_capital
    cash = initial_capital
    shares = 0
    portfolio_values = []

    for i in range(1, len(df)):
        sub_df = df.iloc[:i]
        signals, _ = compute_signals(sub_df, params)
        current_price = df.iloc[i]["Close"]

        if any("BUY" in s for s in signals) and cash > 0:
            shares = cash / current_price
            cash = 0
        elif any("SELL" in s for s in signals) and shares > 0:
            cash = shares * current_price
            shares = 0

        portfolio_values.append(cash + shares * current_price)

    # Eredmények összesítése
    report = {}
    report["total_return_pct"] = ((portfolio_values[-1] / initial_capital) - 1) * 100

    returns = pd.Series(portfolio_values).pct_change().dropna()
    if not returns.empty and returns.std() > 0:
        report["sharpe_ratio"] = returns.mean() / returns.std() * np.sqrt(252)
    else:
        report["sharpe_ratio"] = 0

    # Max Drawdown
    roll_max = pd.Series(portfolio_values).cummax()
    drawdown = (pd.Series(portfolio_values) - roll_max) / roll_max
    report["max_drawdown_pct"] = drawdown.min() * 100

    # Equity görbe DataFrame
    equity_curve = pd.DataFrame(
        {"date": df.index[1:], "portfolio_value": portfolio_values}
    )

    return equity_curve, report


if __name__ == "__main__":
    from app.core.data_loader import load_data, get_supported_ticker_list

    tickers = get_supported_ticker_list()

    for ticker in tickers:
        print(f"Adatok betöltése: {ticker}...")
        df = load_data(ticker, start="2020-01-01", end="2025-06-30")
        signals, indicators = compute_signals(df)
        if signals:
            send_email("Kereskedési szignálok", "\n".join(signals))
        else:
            print("Nincs új szignál.")
