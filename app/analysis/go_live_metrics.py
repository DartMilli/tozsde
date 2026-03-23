from typing import Dict

from app.analysis import get_settings
from app.core.analysis.go_live_metrics import (
    compute_drawdown_summary as core_compute_drawdown_summary,
    compute_loss_streak as core_compute_loss_streak,
)
from app.infrastructure.repositories import DataManagerRepository

DataManager = DataManagerRepository


def _create_data_repository():
    try:
        return DataManager(settings=get_settings())
    except TypeError:
        return DataManager()


def compute_drawdown_summary(ticker: str) -> Dict:
    return core_compute_drawdown_summary(
        ticker=ticker,
        data_manager=_create_data_repository(),
        settings=get_settings(),
    )


def compute_loss_streak(ticker: str) -> Dict:
    return core_compute_loss_streak(
        ticker=ticker,
        data_manager=_create_data_repository(),
        settings=get_settings(),
    )
