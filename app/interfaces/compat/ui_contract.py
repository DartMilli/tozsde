from datetime import datetime


def get_supported_tickers():
    from app.data_access.data_loader import (
        get_supported_tickers as _get_supported_tickers,
    )

    return _get_supported_tickers()


def load_data(*args, **kwargs):
    from app.data_access.data_loader import load_data as _load_data

    return _load_data(*args, **kwargs)


def sanitize_dataframe(*args, **kwargs):
    from app.data_access.data_cleaner import sanitize_dataframe as _sanitize_dataframe

    return _sanitize_dataframe(*args, **kwargs)


def compute_signals(*args, **kwargs):
    from app.analysis.analyzer import compute_signals as _compute_signals

    return _compute_signals(*args, **kwargs)


def get_params(*args, **kwargs):
    from app.analysis.analyzer import get_params as _get_params

    return _get_params(*args, **kwargs)


def get_indicator_description(*args, **kwargs):
    from app.indicators.technical import (
        get_indicator_description as _get_indicator_description,
    )

    return _get_indicator_description(*args, **kwargs)


def get_candle_img_buffer(*args, **kwargs):
    from app.reporting.plotter import get_candle_img_buffer as _get_candle_img_buffer

    return _get_candle_img_buffer(*args, **kwargs)


def get_equity_curve_buffer(*args, **kwargs):
    from app.reporting.plotter import (
        get_equity_curve_buffer as _get_equity_curve_buffer,
    )

    return _get_equity_curve_buffer(*args, **kwargs)


def get_drawdown_curve_buffer(*args, **kwargs):
    from app.reporting.plotter import (
        get_drawdown_curve_buffer as _get_drawdown_curve_buffer,
    )

    return _get_drawdown_curve_buffer(*args, **kwargs)


def Backtester(*args, **kwargs):
    from app.backtesting.backtester import Backtester as _Backtester

    return _Backtester(*args, **kwargs)


def parse_date(value: str):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def validate_ticker(ticker: str, ticker_provider):
    if not ticker:
        return False
    return ticker.upper() in {t.upper() for t in ticker_provider()}
