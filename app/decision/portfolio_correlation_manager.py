"""
Portfolio Correlation Manager (P7 — Portfolio Optimization)

Responsibility:
    - Real-time correlation matrix computation
    - Cross-asset correlation monitoring
    - Diversification scoring
    - Portfolio risk decomposition
    - Low-correlation portfolio optimization

Features:
    - Rolling correlation matrix calculation
    - Diversification score (portfolio variance decomposition)
    - Risk attribution by asset
    - Correlation-based portfolio selection

Usage:
    manager = PortfolioCorrelationManager()
    
    # Compute correlation matrix
    corr_matrix = manager.compute_correlation_matrix(["AAPL", "MSFT", "VOO"])
    
    # Calculate diversification score
    score = manager.calculate_diversification_score({"AAPL": 0.3, "MSFT": 0.3, "VOO": 0.4})
    
    # Optimize for low correlation
    optimal_portfolio = manager.optimize_for_low_correlation(
        candidates=["AAPL", "MSFT", "GOOGL", "JNJ", "XOM"],
        target_size=3
    )
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.data_access.data_manager import DataManager
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class RiskDecomposition:
    """Portfolio risk decomposition result."""
    total_risk: float  # Portfolio standard deviation
    idiosyncratic_risk: float  # Diversifiable risk
    systematic_risk: float  # Market risk
    correlation_contribution: float  # Risk from correlations
    top_risk_contributors: List[Tuple[str, float]]  # (ticker, % contribution)


class PortfolioCorrelationManager:
    """
    Manages portfolio correlation analysis and optimization.
    
    Provides correlation matrix computation, diversification scoring,
    risk decomposition, and correlation-aware portfolio selection.
    """
    
    def __init__(self, lookback_days: int = 90):
        """
        Initialize correlation manager.
        
        Args:
            lookback_days: Days of historical data for correlation calculation
        """
        self.lookback_days = lookback_days
        self.dm = DataManager()
        self._correlation_cache = {}  # Cache for performance
        self._cache_timestamp = None
    
    def compute_correlation_matrix(
        self, tickers: List[str], use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Compute pairwise correlation matrix for tickers.
        
        Args:
            tickers: List of ticker symbols
            use_cache: Whether to use cached result (updated daily)
            
        Returns:
            DataFrame with correlation matrix (N×N)
        """
        # Check cache
        cache_key = "_".join(sorted(tickers))
        if use_cache and cache_key in self._correlation_cache:
            if self._is_cache_valid():
                logger.debug(f"Using cached correlation matrix for {len(tickers)} tickers")
                return self._correlation_cache[cache_key]
        
        # Load price data for all tickers
        returns_dict = {}
        
        for ticker in tickers:
            df = self.dm.load_ohlcv(ticker)
            
            if df is None or len(df) < 30:
                logger.warning(f"Insufficient data for {ticker}, using default correlation")
                continue
            
            # Get recent data
            df_recent = df.tail(self.lookback_days)
            
            # Calculate returns
            returns = df_recent["Close"].pct_change().dropna()
            returns_dict[ticker] = returns
        
        # Create returns DataFrame
        if not returns_dict:
            logger.error("No valid ticker data for correlation calculation")
            return pd.DataFrame()
        
        returns_df = pd.DataFrame(returns_dict)
        
        # Fill missing values (align dates)
        returns_df = returns_df.fillna(0)
        
        # Compute correlation matrix
        corr_matrix = returns_df.corr()
        
        # Cache result
        self._correlation_cache[cache_key] = corr_matrix
        self._cache_timestamp = datetime.now()
        
        logger.info(f"Computed correlation matrix for {len(tickers)} tickers")
        
        return corr_matrix
    
    def calculate_diversification_score(
        self, portfolio: Dict[str, float]
    ) -> float:
        """
        Calculate diversification score using portfolio variance decomposition.
        
        Score = 1 - (portfolio_variance / weighted_avg_individual_variance)
        
        - Score = 1.0: Perfect diversification (zero correlation)
        - Score = 0.0: No diversification benefit (perfect correlation or single asset)
        
        Args:
            portfolio: {ticker: weight} where weights sum to 1.0
            
        Returns:
            Diversification score between 0.0 and 1.0
        """
        tickers = list(portfolio.keys())
        weights = np.array([portfolio[t] for t in tickers])
        
        if len(tickers) == 1:
            return 0.0  # Single asset = no diversification
        
        # Get correlation matrix
        corr_matrix = self.compute_correlation_matrix(tickers)
        
        if corr_matrix.empty:
            return 0.5  # Default when no data
        
        # Load volatilities (standard deviation of returns)
        volatilities = []
        for ticker in tickers:
            df = self.dm.load_ohlcv(ticker)
            if df is None or len(df) < 30:
                volatilities.append(0.20)  # Default 20% annual vol
                continue
            
            returns = df["Close"].pct_change().dropna().tail(self.lookback_days)
            vol = returns.std() * np.sqrt(252)  # Annualized
            volatilities.append(vol)
        
        vols = np.array(volatilities)
        
        # Portfolio variance
        # σ_p^2 = w' Σ w, where Σ = diag(σ) * Corr * diag(σ)
        cov_matrix = corr_matrix.values * np.outer(vols, vols)
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        
        # Weighted average individual variance (no diversification case)
        individual_variances = vols ** 2
        weighted_avg_variance = np.dot(weights ** 2, individual_variances)
        
        # Diversification score
        if weighted_avg_variance < 1e-6:
            return 0.0
        
        score = 1.0 - (portfolio_variance / weighted_avg_variance)
        
        return max(0.0, min(1.0, score))
    
    def decompose_portfolio_risk(
        self, portfolio: Dict[str, float]
    ) -> RiskDecomposition:
        """
        Decompose portfolio risk into components.
        
        Args:
            portfolio: {ticker: weight}
            
        Returns:
            RiskDecomposition with variance breakdown
        """
        tickers = list(portfolio.keys())
        weights = np.array([portfolio[t] for t in tickers])
        
        # Get correlation matrix and volatilities
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
        
        # Covariance matrix
        if not corr_matrix.empty:
            cov_matrix = corr_matrix.values * np.outer(vols, vols)
        else:
            cov_matrix = np.diag(vols ** 2)  # Uncorrelated fallback
        
        # Total portfolio risk
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        total_risk = np.sqrt(portfolio_variance)
        
        # Idiosyncratic risk (diversifiable)
        # Approximation: variance if all correlations = 0
        idiosyncratic_variance = np.dot(weights ** 2, vols ** 2)
        idiosyncratic_risk = np.sqrt(idiosyncratic_variance)
        
        # Systematic risk (undiversifiable)
        # Residual after removing idiosyncratic
        systematic_variance = max(0, portfolio_variance - idiosyncratic_variance)
        systematic_risk = np.sqrt(systematic_variance)
        
        # Correlation contribution
        # Difference between total and idiosyncratic
        correlation_contribution = total_risk - idiosyncratic_risk
        
        # Risk contributors (marginal contribution to variance)
        marginal_contributions = np.dot(cov_matrix, weights)
        risk_contributions = weights * marginal_contributions / portfolio_variance
        
        # Top contributors
        risk_contrib_dict = dict(zip(tickers, risk_contributions))
        top_contributors = sorted(
            risk_contrib_dict.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
        
        return RiskDecomposition(
            total_risk=total_risk,
            idiosyncratic_risk=idiosyncratic_risk,
            systematic_risk=systematic_risk,
            correlation_contribution=correlation_contribution,
            top_risk_contributors=top_contributors
        )
    
    def optimize_for_low_correlation(
        self,
        candidates: List[str],
        target_size: int = 10,
        max_correlation: float = 0.7
    ) -> List[str]:
        """
        Select portfolio with lowest pairwise correlations.
        
        Uses greedy algorithm:
        1. Start with random ticker
        2. Iteratively add ticker with lowest avg correlation to current portfolio
        3. Stop when target size reached or no candidates meet max_correlation constraint
        
        Args:
            candidates: List of candidate tickers
            target_size: Desired portfolio size
            max_correlation: Maximum pairwise correlation allowed
            
        Returns:
            List of selected tickers
        """
        if not candidates:
            return []
        
        if len(candidates) <= target_size:
            return candidates
        
        # Compute correlation matrix for all candidates
        corr_matrix = self.compute_correlation_matrix(candidates)
        
        if corr_matrix.empty:
            # Fallback: return first target_size tickers
            logger.warning("No correlation data, returning first tickers")
            return candidates[:target_size]
        
        selected = []
        remaining = candidates.copy()
        
        # Start with first ticker (or could be random)
        selected.append(remaining.pop(0))
        
        # Greedy selection
        while len(selected) < target_size and remaining:
            best_ticker = None
            best_avg_corr = float('inf')
            
            for candidate in remaining:
                # Calculate average correlation with already selected tickers
                correlations = []
                for selected_ticker in selected:
                    try:
                        corr = corr_matrix.loc[candidate, selected_ticker]
                        correlations.append(abs(corr))
                    except KeyError:
                        correlations.append(0.5)  # Default
                
                avg_corr = np.mean(correlations)
                
                # Check if meets constraint
                max_corr = max(correlations) if correlations else 0.0
                
                if max_corr <= max_correlation and avg_corr < best_avg_corr:
                    best_avg_corr = avg_corr
                    best_ticker = candidate
            
            if best_ticker is None:
                # No candidates meet constraint
                logger.info(f"No more candidates meeting max_correlation={max_correlation}")
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
        """
        Find pairs of tickers with high correlation.
        
        Args:
            tickers: List of ticker symbols
            threshold: Correlation threshold (default 0.8)
            
        Returns:
            List of (ticker1, ticker2, correlation) tuples
        """
        corr_matrix = self.compute_correlation_matrix(tickers)
        
        if corr_matrix.empty:
            return []
        
        high_corr_pairs = []
        
        for i, ticker1 in enumerate(tickers):
            for j, ticker2 in enumerate(tickers):
                if i >= j:  # Skip diagonal and duplicates
                    continue
                
                try:
                    corr = corr_matrix.loc[ticker1, ticker2]
                    if abs(corr) >= threshold:
                        high_corr_pairs.append((ticker1, ticker2, corr))
                except KeyError:
                    continue
        
        # Sort by correlation (highest first)
        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return high_corr_pairs
    
    # Helper methods
    
    def _is_cache_valid(self) -> bool:
        """Check if correlation cache is still valid (< 24 hours old)."""
        if self._cache_timestamp is None:
            return False
        
        age = datetime.now() - self._cache_timestamp
        return age < timedelta(hours=24)
    
    def clear_cache(self):
        """Clear correlation cache (force recomputation)."""
        self._correlation_cache = {}
        self._cache_timestamp = None
        logger.info("Correlation cache cleared")


# Utility functions

def check_portfolio_diversification(portfolio: Dict[str, float]) -> float:
    """
    Quick diversification check for portfolio.
    
    Args:
        portfolio: {ticker: weight}
        
    Returns:
        Diversification score (0-1)
    """
    manager = PortfolioCorrelationManager()
    return manager.calculate_diversification_score(portfolio)


def find_uncorrelated_assets(
    base_ticker: str, candidates: List[str], max_correlation: float = 0.5
) -> List[str]:
    """
    Find assets with low correlation to base ticker.
    
    Args:
        base_ticker: Reference ticker
        candidates: List of candidates to check
        max_correlation: Maximum allowed correlation
        
    Returns:
        List of low-correlation tickers
    """
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
