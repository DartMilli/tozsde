"""Compatibility facade for technical indicators.

Core implementations live in `app.core.indicators.*`.
This module keeps legacy imports and aliases stable.
"""

from app.core.indicators import get_indicator_description, technical_indicators_summary
from app.core.indicators.momentum import rsi, stoch
from app.core.indicators.trend import adx, ema, macd, sma
from app.core.indicators.volatility import atr, bbands


def rsi_old(data, period=14):
    return rsi(data, period=period)


def ema_old(data, period=14):
    return ema(data, period=period)
