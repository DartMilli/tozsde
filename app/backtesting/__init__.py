"""Backtesting package DI helpers."""

from app.config.build_settings import build_settings

_settings = None


def set_settings(s):
    global _settings
    _settings = s


def get_settings():
    global _settings
    if _settings is None:
        _settings = build_settings()
    return _settings
