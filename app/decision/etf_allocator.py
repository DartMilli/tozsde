"""
ETF Allocator Module (P7 — Portfolio Optimization)

Responsibility:
    - ETF vs. stock classification
    - Sector exposure analysis
    - ETF cost comparison
    - Optimal portfolio mix calculation (ETF + stock)

Features:
    - Asset type detection (ETF vs. individual stock)
    - Sector exposure balancing
    - Cost-aware ETF selection
    - Mix optimization for diversification + cost

Usage:
    allocator = ETFAllocator()
    asset_type = allocator.classify_asset_type("VOO")  # AssetType.ETF
    
    mix = allocator.calculate_optimal_mix(
        target_sectors={"Tech": 0.40, "Healthcare": 0.30, "Finance": 0.30"},
        available_etfs=["VOO", "QQQ", "XLV"],
        available_stocks=["AAPL", "MSFT", "JNJ"]
    )
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import pandas as pd

from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)


class AssetType(Enum):
    """Asset classification."""
    ETF = "ETF"
    STOCK = "STOCK"
    UNKNOWN = "UNKNOWN"


@dataclass
class CostComparison:
    """Cost comparison between two ETFs."""
    ticker1: str
    ticker2: str
    expense_ratio1: float  # Annual TER (e.g., 0.0003 = 0.03%)
    expense_ratio2: float
    cheaper_option: str
    cost_difference: float  # Basis points
    annual_savings_per_10k: float  # Dollars saved per $10k invested


@dataclass
class PortfolioMix:
    """Optimal portfolio mix result."""
    etf_weights: Dict[str, float]  # {ticker: weight}
    stock_weights: Dict[str, float]
    total_estimated_cost: float  # Annual cost in basis points
    sector_exposure: Dict[str, float]  # {sector: percentage}
    diversification_score: float  # 0.0 to 1.0
    optimization_notes: List[str]  # Explanation of choices


class ETFAllocator:
    """
    Manages ETF allocation and portfolio mix optimization.
    
    Provides classification, sector analysis, cost comparison,
    and optimal ETF + stock mix calculation.
    """
    
    # Known ETF tickers (expand as needed)
    KNOWN_ETFS = {
        # Broad market
        "SPY", "VOO", "IVV",  # S&P 500
        "QQQ", "QQQM",  # Nasdaq 100
        "VTI", "ITOT",  # Total US market
        "IWM", "VB",  # Small cap
        
        # Sector ETFs
        "XLK",  # Technology
        "XLV",  # Healthcare
        "XLF",  # Financials
        "XLE",  # Energy
        "XLI",  # Industrials
        "XLP",  # Consumer Staples
        "XLY",  # Consumer Discretionary
        "XLU",  # Utilities
        "XLB",  # Materials
        "XLRE",  # Real Estate
        
        # International
        "VEA", "IEFA",  # Developed markets
        "VWO", "IEMG",  # Emerging markets
        "EFA",  # EAFE
        
        # Bond ETFs (for future)
        "AGG", "BND",  # Aggregate bonds
        "TLT", "IEF",  # Treasury bonds
    }
    
    # ETF expense ratios (basis points, approximate)
    ETF_EXPENSE_RATIOS = {
        "VOO": 3,  # 0.03%
        "SPY": 9,  # 0.09%
        "IVV": 3,  # 0.03%
        "QQQ": 20,  # 0.20%
        "QQQM": 15,  # 0.15%
        "VTI": 3,  # 0.03%
        "IWM": 19,  # 0.19%
        "XLK": 10,  # 0.10%
        "XLV": 10,
        "XLF": 10,
        "XLE": 10,
        "VEA": 5,
        "VWO": 8,
    }
    
    # Sector mappings (simplified, for demo)
    SECTOR_MAP = {
        # Tech stocks
        "AAPL": "Technology",
        "MSFT": "Technology",
        "GOOGL": "Technology",
        "AMZN": "Technology",
        "NVDA": "Technology",
        "META": "Technology",
        
        # Healthcare
        "JNJ": "Healthcare",
        "UNH": "Healthcare",
        "PFE": "Healthcare",
        "ABBV": "Healthcare",
        
        # Finance
        "JPM": "Financials",
        "BAC": "Financials",
        "WFC": "Financials",
        "GS": "Financials",
        
        # Consumer
        "WMT": "Consumer Staples",
        "PG": "Consumer Staples",
        "KO": "Consumer Staples",
        "HD": "Consumer Discretionary",
        "NKE": "Consumer Discretionary",
        
        # Energy
        "XOM": "Energy",
        "CVX": "Energy",
        
        # Industrials
        "BA": "Industrials",
        "CAT": "Industrials",
        "GE": "Industrials",
    }
    
    def __init__(self):
        """Initialize ETF allocator."""
        pass
    
    def classify_asset_type(self, ticker: str) -> AssetType:
        """
        Classify asset as ETF or individual stock.
        
        Args:
            ticker: Asset ticker symbol
            
        Returns:
            AssetType.ETF or AssetType.STOCK
        """
        ticker_upper = ticker.upper()
        
        if ticker_upper in self.KNOWN_ETFS:
            return AssetType.ETF
        elif ticker_upper in self.SECTOR_MAP:
            return AssetType.STOCK
        else:
            # Default: assume stock if not in known ETFs
            logger.warning(f"Unknown ticker {ticker}, assuming STOCK")
            return AssetType.STOCK
    
    def analyze_sector_exposure(
        self, portfolio: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate sector exposure for current portfolio.
        
        Args:
            portfolio: {ticker: weight} where weights sum to ~1.0
            
        Returns:
            {sector: exposure_percentage} dict
        """
        sector_exposure = {}
        
        for ticker, weight in portfolio.items():
            # Check if it's an ETF (broad exposure)
            asset_type = self.classify_asset_type(ticker)
            
            if asset_type == AssetType.ETF:
                # For ETFs, distribute across multiple sectors
                # Simplified: assume broad market ETFs are diversified
                if ticker.upper() in ["VOO", "SPY", "IVV", "VTI"]:
                    # S&P 500 / Total market - rough sector breakdown
                    broad_sectors = {
                        "Technology": 0.28,
                        "Healthcare": 0.13,
                        "Financials": 0.13,
                        "Consumer Discretionary": 0.11,
                        "Consumer Staples": 0.07,
                        "Industrials": 0.08,
                        "Energy": 0.04,
                        "Other": 0.16
                    }
                    for sector, sector_weight in broad_sectors.items():
                        sector_exposure[sector] = sector_exposure.get(sector, 0) + weight * sector_weight
                
                elif ticker.upper() == "QQQ":
                    # Tech-heavy Nasdaq
                    sector_exposure["Technology"] = sector_exposure.get("Technology", 0) + weight * 0.50
                    sector_exposure["Consumer Discretionary"] = sector_exposure.get("Consumer Discretionary", 0) + weight * 0.20
                    sector_exposure["Healthcare"] = sector_exposure.get("Healthcare", 0) + weight * 0.10
                    sector_exposure["Other"] = sector_exposure.get("Other", 0) + weight * 0.20
                
                elif ticker.upper().startswith("XL"):
                    # Sector-specific ETF
                    sector_name = self._get_sector_from_etf(ticker.upper())
                    sector_exposure[sector_name] = sector_exposure.get(sector_name, 0) + weight
                
                else:
                    # Unknown ETF, assume diversified
                    sector_exposure["Other"] = sector_exposure.get("Other", 0) + weight
            
            else:
                # Individual stock
                sector = self.SECTOR_MAP.get(ticker.upper(), "Other")
                sector_exposure[sector] = sector_exposure.get(sector, 0) + weight
        
        return sector_exposure
    
    def compare_etf_costs(self, ticker1: str, ticker2: str) -> CostComparison:
        """
        Compare costs between two ETFs.
        
        Args:
            ticker1: First ETF ticker
            ticker2: Second ETF ticker
            
        Returns:
            CostComparison object with expense ratios
        """
        expense1 = self.ETF_EXPENSE_RATIOS.get(ticker1.upper(), 20)  # Default 0.20%
        expense2 = self.ETF_EXPENSE_RATIOS.get(ticker2.upper(), 20)
        
        cheaper = ticker1 if expense1 < expense2 else ticker2
        cost_diff = abs(expense1 - expense2)
        
        # Calculate annual savings per $10k invested
        # cost_diff is in basis points (1 bp = 0.01%)
        annual_savings = (cost_diff / 10000) * 10000  # For $10k investment
        
        return CostComparison(
            ticker1=ticker1,
            ticker2=ticker2,
            expense_ratio1=expense1 / 10000,  # Convert bp to decimal
            expense_ratio2=expense2 / 10000,
            cheaper_option=cheaper,
            cost_difference=cost_diff,
            annual_savings_per_10k=annual_savings
        )
    
    def calculate_optimal_mix(
        self,
        target_sectors: Dict[str, float],
        available_etfs: List[str],
        available_stocks: List[str],
        max_etf_weight: float = 0.7
    ) -> PortfolioMix:
        """
        Calculate optimal portfolio mix of ETFs and stocks.
        
        Strategy:
        1. Use low-cost ETFs for broad exposure
        2. Use individual stocks for targeted sector bets
        3. Minimize total costs
        4. Maximize diversification
        
        Args:
            target_sectors: Desired sector allocation {sector: weight}
            available_etfs: List of ETF tickers to consider
            available_stocks: List of stock tickers to consider
            max_etf_weight: Maximum total ETF weight (default 70%)
            
        Returns:
            PortfolioMix with optimal allocation
        """
        etf_weights = {}
        stock_weights = {}
        notes = []
        
        # Step 1: Select low-cost broad market ETF for base exposure
        broad_etfs = [e for e in available_etfs if e.upper() in ["VOO", "VTI", "IVV", "SPY"]]
        
        if broad_etfs:
            # Pick cheapest broad market ETF
            cheapest_broad = min(
                broad_etfs,
                key=lambda e: self.ETF_EXPENSE_RATIOS.get(e.upper(), 100)
            )
            
            # Allocate 40-50% to broad market ETF
            base_weight = min(0.50, max_etf_weight)
            etf_weights[cheapest_broad] = base_weight
            notes.append(f"Base allocation: {base_weight:.1%} to {cheapest_broad} (low-cost broad)")
        
        # Step 2: Add sector-specific exposure via stocks or sector ETFs
        remaining_weight = 1.0 - sum(etf_weights.values())
        
        # Distribute remaining weight to stocks based on target sectors
        stock_sectors = {}
        for stock in available_stocks:
            sector = self.SECTOR_MAP.get(stock.upper(), "Other")
            if sector not in stock_sectors:
                stock_sectors[sector] = []
            stock_sectors[sector].append(stock)
        
        # Allocate to sectors
        for sector, target_pct in target_sectors.items():
            if sector in stock_sectors and stock_sectors[sector]:
                # Distribute sector weight evenly among available stocks
                stocks_in_sector = stock_sectors[sector]
                weight_per_stock = (target_pct * remaining_weight) / len(stocks_in_sector)
                
                for stock in stocks_in_sector[:3]:  # Max 3 stocks per sector
                    stock_weights[stock] = weight_per_stock
                    notes.append(f"{stock}: {weight_per_stock:.1%} for {sector} exposure")
        
        # Normalize weights
        total_weight = sum(etf_weights.values()) + sum(stock_weights.values())
        if total_weight > 0:
            etf_weights = {k: v / total_weight for k, v in etf_weights.items()}
            stock_weights = {k: v / total_weight for k, v in stock_weights.items()}
        
        # Calculate metrics
        total_cost = self._calculate_portfolio_cost(etf_weights, stock_weights)
        sector_exposure = self.analyze_sector_exposure({**etf_weights, **stock_weights})
        diversification_score = self._calculate_diversification_score(etf_weights, stock_weights)
        
        return PortfolioMix(
            etf_weights=etf_weights,
            stock_weights=stock_weights,
            total_estimated_cost=total_cost,
            sector_exposure=sector_exposure,
            diversification_score=diversification_score,
            optimization_notes=notes
        )
    
    # Helper methods
    
    def _get_sector_from_etf(self, etf_ticker: str) -> str:
        """Map sector ETF ticker to sector name."""
        sector_etf_map = {
            "XLK": "Technology",
            "XLV": "Healthcare",
            "XLF": "Financials",
            "XLE": "Energy",
            "XLI": "Industrials",
            "XLP": "Consumer Staples",
            "XLY": "Consumer Discretionary",
            "XLU": "Utilities",
            "XLB": "Materials",
            "XLRE": "Real Estate"
        }
        return sector_etf_map.get(etf_ticker, "Other")
    
    def _calculate_portfolio_cost(
        self, etf_weights: Dict[str, float], stock_weights: Dict[str, float]
    ) -> float:
        """
        Calculate total portfolio cost in basis points.
        
        ETFs have expense ratios, stocks have transaction costs (ignored here).
        """
        total_cost = 0.0
        
        for etf, weight in etf_weights.items():
            expense_ratio = self.ETF_EXPENSE_RATIOS.get(etf.upper(), 20)  # bp
            total_cost += weight * expense_ratio
        
        # Stocks: assume no ongoing expense (just transaction costs at rebalance)
        # Could add estimated annual turnover cost here
        
        return total_cost
    
    def _calculate_diversification_score(
        self, etf_weights: Dict[str, float], stock_weights: Dict[str, float]
    ) -> float:
        """
        Calculate diversification score (0-1).
        
        Higher score = more diversified (more holdings, more even distribution).
        """
        all_weights = {**etf_weights, **stock_weights}
        
        if not all_weights:
            return 0.0
        
        # Number of holdings
        num_holdings = len(all_weights)
        
        # Herfindahl index (concentration)
        herfindahl = sum(w ** 2 for w in all_weights.values())
        
        # Diversification score: inverse of Herfindahl, normalized
        # Perfect diversification (equal weights): H = 1/N, score = 1.0
        # Single holding: H = 1.0, score = 0.0
        if num_holdings == 1:
            return 0.0
        
        max_diversification = 1.0 / num_holdings  # Equal weights
        score = 1.0 - (herfindahl - max_diversification) / (1.0 - max_diversification)
        
        return max(0.0, min(1.0, score))


# Utility functions

def get_low_cost_etf(sector: Optional[str] = None) -> str:
    """
    Get low-cost ETF recommendation for sector.
    
    Args:
        sector: Sector name, or None for broad market
        
    Returns:
        ETF ticker symbol
    """
    allocator = ETFAllocator()
    
    if sector is None or sector == "Broad":
        # Broad market: VOO is cheapest S&P 500
        return "VOO"
    
    # Sector-specific
    sector_etfs = {
        "Technology": "XLK",
        "Healthcare": "XLV",
        "Financials": "XLF",
        "Energy": "XLE",
        "Industrials": "XLI",
        "Consumer Staples": "XLP",
        "Consumer Discretionary": "XLY"
    }
    
    return sector_etfs.get(sector, "VOO")


def estimate_portfolio_cost(portfolio: Dict[str, float]) -> float:
    """
    Estimate annual portfolio cost in basis points.
    
    Args:
        portfolio: {ticker: weight}
        
    Returns:
        Total cost in basis points
    """
    allocator = ETFAllocator()
    
    etfs = {}
    stocks = {}
    
    for ticker, weight in portfolio.items():
        if allocator.classify_asset_type(ticker) == AssetType.ETF:
            etfs[ticker] = weight
        else:
            stocks[ticker] = weight
    
    return allocator._calculate_portfolio_cost(etfs, stocks)


def classify_asset_type(ticker: str) -> AssetType:
    """
    Classify asset type (ETF vs. stock).
    
    Args:
        ticker: Ticker symbol
        
    Returns:
        AssetType enum (ETF, STOCK, or UNKNOWN)
    """
    allocator = ETFAllocator()
    return allocator.classify_asset_type(ticker)
