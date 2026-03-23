from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.bootstrap.build_settings import build_settings
from app.infrastructure.logger import setup_logger
from app.infrastructure.repositories import DataManagerRepository

logger = setup_logger(__name__)


@dataclass
class RiskDecomposition:
    total_risk: float
    idiosyncratic_risk: float
    systematic_risk: float
    correlation_contribution: float
    top_risk_contributors: List[Tuple[str, float]]


class PortfolioCorrelationManager:
    def __init__(self, lookback_days: int = 90, settings=None):
        self.lookback_days = lookback_days
        self.settings = settings
        try:
            if getattr(self, "dm", None) is None:
                self.dm = self._create_data_repository()
        except Exception:
            self.dm = self._create_data_repository()

        self._correlation_cache = {}
        self._cache_timestamp = None

    def compute_correlation_matrix(
        self, tickers: List[str], use_cache: bool = True
    ) -> pd.DataFrame:
        cache_key = "_".join(sorted(tickers))
        if use_cache and cache_key in self._correlation_cache:
            if self._is_cache_valid():
                logger.debug(
                    f"Using cached correlation matrix for {len(tickers)} tickers"
                )
                return self._correlation_cache[cache_key]

        returns_dict = {}

        for ticker in tickers:
            df = self.dm.load_ohlcv(ticker)

            if df is None or len(df) < 30:
                logger.warning(
                    f"Insufficient data for {ticker}, using default correlation"
                )
                continue

            df_recent = df.tail(self.lookback_days)
            returns = df_recent["Close"].pct_change().dropna()
            returns_dict[ticker] = returns

        if not returns_dict:
            logger.error("No valid ticker data for correlation calculation")
            return pd.DataFrame()

        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.fillna(0)

        corr_matrix = returns_df.corr()

        self._correlation_cache[cache_key] = corr_matrix
        self._cache_timestamp = datetime.now()

        logger.info(f"Computed correlation matrix for {len(tickers)} tickers")

        return corr_matrix

    def calculate_diversification_score(self, portfolio: Dict[str, float]) -> float:
        tickers = list(portfolio.keys())
        weights = np.array([portfolio[t] for t in tickers])

        if len(tickers) == 1:
            return 0.0

        corr_matrix = self.compute_correlation_matrix(tickers)

        if corr_matrix.empty:
            return 0.5

        volatilities = []
        for ticker in tickers:
            df = self.dm.load_ohlcv(ticker)
            if df is None or len(df) < 30:
                volatilities.append(0.20)
                continue

            returns = df["Close"].pct_change().dropna().tail(self.lookback_days)
            vol = returns.std() * np.sqrt(252)
            volatilities.append(vol)

        vols = np.array(volatilities)

        cov_matrix = corr_matrix.values * np.outer(vols, vols)
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))

        individual_variances = vols**2
        weighted_avg_variance = np.dot(weights**2, individual_variances)

        if weighted_avg_variance < 1e-6:
            return 0.0

        score = 1.0 - (portfolio_variance / weighted_avg_variance)

        return max(0.0, min(1.0, score))

    def decompose_portfolio_risk(
        self, portfolio: Dict[str, float]
    ) -> RiskDecomposition:
        tickers = list(portfolio.keys())
        weights = np.array([portfolio[t] for t in tickers])

        corr_matrix = self.compute_correlation_matrix(tickers)

        volatilities = []
        for ticker in tickers:
            df = self.dm.load_ohlcv(ticker)
            if df is None or len(df) < 30:
                volatilities.append(0.20)
                continue
            returns = df["Close"].pct_change().dropna().tail(self.lookback_days)
            vol = returns.std() * np.sqrt(252)
            volatilities.append(vol)

        vols = np.array(volatilities)

        if not corr_matrix.empty:
            cov_matrix = corr_matrix.values * np.outer(vols, vols)
        else:
            cov_matrix = np.diag(vols**2)

        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        total_risk = np.sqrt(portfolio_variance)

        idiosyncratic_variance = np.dot(weights**2, vols**2)
        idiosyncratic_risk = np.sqrt(idiosyncratic_variance)

        systematic_variance = max(0, portfolio_variance - idiosyncratic_variance)
        systematic_risk = np.sqrt(systematic_variance)

        correlation_contribution = total_risk - idiosyncratic_risk

        marginal_contributions = np.dot(cov_matrix, weights)
        risk_contributions = weights * marginal_contributions / portfolio_variance

        risk_contrib_dict = dict(zip(tickers, risk_contributions))
        top_contributors = sorted(
            risk_contrib_dict.items(), key=lambda x: abs(x[1]), reverse=True
        )[:5]

        return RiskDecomposition(
            total_risk=total_risk,
            idiosyncratic_risk=idiosyncratic_risk,
            systematic_risk=systematic_risk,
            correlation_contribution=correlation_contribution,
            top_risk_contributors=top_contributors,
        )

    def optimize_for_low_correlation(
        self, candidates: List[str], target_size: int = 10, max_correlation: float = 0.7
    ) -> List[str]:
        if not candidates:
            return []

        if len(candidates) <= target_size:
            return candidates

        corr_matrix = self.compute_correlation_matrix(candidates)

        if corr_matrix.empty:
            logger.warning("No correlation data, returning first tickers")
            return candidates[:target_size]

        selected = []
        remaining = candidates.copy()

        selected.append(remaining.pop(0))

        best_avg_corr = float("inf")
        while len(selected) < target_size and remaining:
            best_ticker = None

            for candidate in remaining:
                correlations = []
                for selected_ticker in selected:
                    try:
                        corr = corr_matrix.loc[candidate, selected_ticker]
                        correlations.append(abs(corr))
                    except KeyError:
                        correlations.append(0.5)

                avg_corr = np.mean(correlations)
                max_corr = max(correlations) if correlations else 0.0

                if max_corr <= max_correlation and avg_corr < best_avg_corr:
                    best_avg_corr = avg_corr
                    best_ticker = candidate

            if best_ticker is None:
                logger.info(
                    f"No more candidates meeting max_correlation={max_correlation}"
                )
                break

            selected.append(best_ticker)
            remaining.remove(best_ticker)

        logger.info(
            f"Selected {len(selected)} tickers with avg correlation {best_avg_corr:.2f}"
        )

        return selected

    def get_highly_correlated_pairs(
        self, tickers: List[str], threshold: float = 0.8
    ) -> List[Tuple[str, str, float]]:
        corr_matrix = self.compute_correlation_matrix(tickers)

        if corr_matrix.empty:
            return []

        high_corr_pairs = []

        for i, ticker1 in enumerate(tickers):
            for j, ticker2 in enumerate(tickers):
                if i >= j:
                    continue

                try:
                    corr = corr_matrix.loc[ticker1, ticker2]
                    if abs(corr) >= threshold:
                        high_corr_pairs.append((ticker1, ticker2, corr))
                except KeyError:
                    continue

        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        return high_corr_pairs

    def _is_cache_valid(self) -> bool:
        if self._cache_timestamp is None:
            return False

        age = datetime.now() - self._cache_timestamp
        return age < timedelta(hours=24)

    def clear_cache(self):
        self._correlation_cache = {}
        self._cache_timestamp = None
        logger.info("Correlation cache cleared")

    def _create_data_repository(self):
        cfg = self.settings or build_settings()
        try:
            return DataManagerRepository(settings=cfg)
        except TypeError:
            return DataManagerRepository()


def check_portfolio_diversification(portfolio: Dict[str, float]) -> float:
    manager = PortfolioCorrelationManager()
    return manager.calculate_diversification_score(portfolio)


def find_uncorrelated_assets(
    base_ticker: str, candidates: List[str], max_correlation: float = 0.5
) -> List[str]:
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
