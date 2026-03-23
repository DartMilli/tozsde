from app.bootstrap.build_settings import build_settings
from app.core.decision.allocation import (
    allocate_capital as core_allocate_capital,
    enforce_correlation_limits as core_enforce_correlation_limits,
)
from app.infrastructure.repositories import DataManagerRepository


def allocate_capital(decisions: list, settings=None) -> list:
    return core_allocate_capital(
        decisions,
        settings=settings,
        get_correlation_matrix_fn=_get_correlation_matrix,
    )


def _get_correlation_matrix(tickers, ref_date=None, settings=None):
    try:
        dm = DataManagerRepository(settings=settings)
    except TypeError:
        dm = DataManagerRepository()
    return dm.get_correlation_matrix(tickers, ref_date=ref_date)


def enforce_correlation_limits(
    decisions: list, max_correlation: float = 0.7, settings=None
) -> list:
    return core_enforce_correlation_limits(
        decisions,
        max_correlation=max_correlation,
        settings=settings,
        get_correlation_matrix_fn=_get_correlation_matrix,
    )
