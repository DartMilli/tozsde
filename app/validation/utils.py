"""Shared helpers for validation modules."""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta

from app.validation import get_settings
from app.data_access.data_loader import get_supported_ticker_list


def _parse_date(value: str | None, fallback: date) -> date:
    if not value:
        return fallback
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return fallback


def get_validation_ticker() -> str:
    ticker = os.getenv("VALIDATION_TICKER")
    if ticker:
        return ticker
    try:
        settings = get_settings()
        if settings.TICKERS:
            return settings.TICKERS[0]
        excluded = set(getattr(settings, "EXCLUDED_TICKERS", []))
        for supported in get_supported_ticker_list():
            if supported not in excluded:
                return supported
        return "VOO"
    except Exception:
        return "VOO"


def get_validation_window() -> tuple[date, date]:
    default_start = _parse_date(
        get_settings().START_DATE, date.today() - timedelta(days=365)
    )
    default_end = _parse_date(get_settings().END_DATE, date.today())

    start = _parse_date(os.getenv("VALIDATION_START_DATE"), default_start)
    end = _parse_date(os.getenv("VALIDATION_END_DATE"), default_end)
    if start > end:
        start, end = end, start
    return start, end


def get_horizon_days() -> int:
    try:
        return int(os.getenv("VALIDATION_HORIZON_DAYS", "5"))
    except ValueError:
        return 5


def get_scenario() -> str:
    return os.getenv("VALIDATION_SCENARIO", "elevated_volatility")


def get_validation_mode() -> str:
    return os.getenv("VALIDATION_MODE", "quick").lower()


def is_light_mode() -> bool:
    return get_validation_mode() in {"quick", "light"}
