import logging
import os
from datetime import datetime, timezone
from app.bootstrap.build_settings import build_settings


def _get_setting(settings, name, default=None):
    # Prefer explicit settings object, then build_settings fallback, then default
    if settings is None:
        try:
            settings = build_settings()
        except Exception:
            settings = None

    if settings is not None and hasattr(settings, name):
        return getattr(settings, name)

    return default


def is_logger_debug(logger):
    return logger.level == logging.DEBUG


def setup_logger(name: str, settings=None) -> logging.Logger:
    log_dir = _get_setting(settings, "LOG_DIR", "logs")
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        try:
            os.makedirs(str(log_dir), exist_ok=True)
        except Exception:
            pass

    logger = logging.getLogger(name)

    lvl = _get_setting(settings, "LOGGING_LEVEL", "INFO")
    if lvl == "DEBUG":
        logger.setLevel(logging.DEBUG)
    elif lvl == "WARNING":
        logger.setLevel(logging.WARNING)
    elif lvl == "ERROR":
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    log_path = os.path.join(
        log_dir,
        f"app_{datetime.now(timezone.utc).strftime('%Y_%m')}.log",
    )

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
