"""
P4.2 - DB Schema Apply Script

Responsibility:
    - Create / migrate database schema
    - Idempotent (safe to run multiple times)
    - NO business logic
    - NO data mutation beyond schema

Usage:
    python -m app.scripts.apply_schema
"""

from app.infrastructure.repositories import DataManagerRepository as DataManager
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)


def apply_schema(dm: DataManager | None = None, settings=None):
    logger.info("Applying DB schema...")
    try:
        if dm is None:
            try:
                dm = DataManager(settings=settings)
            except TypeError:
                dm = DataManager()
        dm.initialize_tables()
        logger.info("DB schema applied successfully")

    except Exception:
        logger.exception("DB schema apply FAILED")
        raise


if __name__ == "__main__":
    apply_schema()
