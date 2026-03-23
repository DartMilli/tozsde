import numpy as np


def compute_normalized_volatility(
    df,
    atr_period: int = 14,
    lookback: int = 20,
) -> float:
    high = df["High"] if "High" in df.columns else df["high"]
    low = df["Low"] if "Low" in df.columns else df["low"]
    close = df["Close"] if "Close" in df.columns else df["close"]

    tr = np.maximum(
        high - low,
        np.maximum(
            abs(high - close.shift(1)),
            abs(low - close.shift(1)),
        ),
    )

    atr = tr.rolling(atr_period).mean()

    atr_recent = atr.iloc[-lookback:].mean()
    close_recent = close.iloc[-lookback:].mean()

    if close_recent == 0 or np.isnan(atr_recent):
        return 0.0

    return float(atr_recent / close_recent)


def scale_confidence_by_volatility(
    confidence: float,
    volatility: float,
    soft_cap: float = 0.03,
) -> float:
    if confidence is None:
        return confidence

    if volatility <= soft_cap:
        return confidence

    scale = soft_cap / volatility
    return max(0.0, min(confidence * scale, 1.0))
