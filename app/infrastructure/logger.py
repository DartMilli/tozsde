import logging
import os
from datetime import datetime, timezone
from app.config.config import Config


def is_logger_debug(logger):
    return logger.level == logging.DEBUG


def setup_logger(name: str) -> logging.Logger:
    os.makedirs(Config.LOG_DIR, exist_ok=True)

    logger = logging.getLogger(name)

    if Config.LOGGING_LEVEL == "DEBUG":
        logger.setLevel(logging.DEBUG)
    elif Config.LOGGING_LEVEL == "WARNING":
        logger.setLevel(logging.WARNING)
    elif Config.LOGGING_LEVEL == "ERROR":
        logger.setLevel(logging.WARNING)
    else:  # Config.LOGGING_LEVEL == "INFO"
        logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger  # ne duplikáljunk

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    log_path = os.path.join(
        Config.LOG_DIR,
        f"app_{datetime.now(timezone.utc).strftime('%Y_%m')}.log",
    )

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
