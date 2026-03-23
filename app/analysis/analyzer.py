import logging

from app.analysis import get_settings
from app.core.analysis import analyzer as _core_analyzer
from app.infrastructure.logger import setup_logger
import app.indicators.technical as ta

settings = None

# Back-compat patch surfaces used in tests
trend = ta
momentum = ta
volatility = ta

logger = setup_logger(__name__)

# Shared data/constants for compatibility
default_params = _core_analyzer.default_params
param_bounds = _core_analyzer.param_bounds
_PARAMS_CACHE = _core_analyzer._PARAMS_CACHE
_CORE_GET_PARAMS = _core_analyzer.get_params
_CORE_GET_DEFAULT_PARAMS = _core_analyzer.get_default_params
_CORE_SAVE_PARAMS_FOR_TICKER = _core_analyzer.save_params_for_ticker


def _conf(provided=None):
    if provided is not None:
        return provided
    try:
        return get_settings()
    except Exception:
        return None


def _sync_core_patch_points():
    _core_analyzer.settings = settings
    _core_analyzer._conf = _conf
    _core_analyzer.ta = ta
    _core_analyzer.trend = trend
    _core_analyzer.momentum = momentum
    _core_analyzer.volatility = volatility
    _core_analyzer.get_params = get_params
    _core_analyzer.get_default_params = get_default_params


def get_default_params():
    return _CORE_GET_DEFAULT_PARAMS()


def get_params(ticker):
    _sync_core_patch_points()
    return _CORE_GET_PARAMS(ticker)


def save_params_for_ticker(ticker, params):
    _sync_core_patch_points()
    return _CORE_SAVE_PARAMS_FOR_TICKER(ticker, params)


def compute_signals(df, ticker, params, return_series=False, audit=None):
    _sync_core_patch_points()
    return _core_analyzer.compute_signals(
        df,
        ticker,
        params,
        return_series=return_series,
        audit=audit,
    )


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
