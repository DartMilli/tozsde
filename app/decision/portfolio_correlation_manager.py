from app.core.decision.portfolio_correlation_manager import (  # noqa: F401
    RiskDecomposition,
    PortfolioCorrelationManager as CorePortfolioCorrelationManager,
)
from app.infrastructure.repositories import DataManagerRepository

DataManager = DataManagerRepository


class PortfolioCorrelationManager(CorePortfolioCorrelationManager):
    def _create_data_repository(self):
        cfg = self.settings
        try:
            return DataManager(settings=cfg)
        except TypeError:
            return DataManager()


def check_portfolio_diversification(portfolio):
    manager = PortfolioCorrelationManager()
    return manager.calculate_diversification_score(portfolio)


def find_uncorrelated_assets(base_ticker, candidates, max_correlation: float = 0.5):
    manager = PortfolioCorrelationManager()

    all_tickers = [base_ticker] + candidates
    corr_matrix = manager.compute_correlation_matrix(all_tickers)

    if corr_matrix.empty:
        return []

    uncorrelated = []
    for candidate in candidates:
        try:
            corr = corr_matrix.loc[base_ticker, candidate]
            if abs(corr) <= max_correlation:
                uncorrelated.append(candidate)
        except KeyError:
            continue

    return uncorrelated
