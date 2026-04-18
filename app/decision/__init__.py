"""Decision package DI helpers."""

import os
import warnings

from app.config.build_settings import build_settings


def emit_compat_warning(module_name: str) -> None:
    if os.getenv("ENABLE_COMPAT_DEPRECATION_WARNINGS", "false").lower() == "true":
        warnings.warn(
            f"{module_name} is deprecated; use app.core.decision directly.",
            DeprecationWarning,
            stacklevel=2,
        )


_settings = None


def set_settings(s):
    global _settings
    _settings = s


def get_settings():
    global _settings
    if _settings is None:
        _settings = build_settings()
    return _settings
