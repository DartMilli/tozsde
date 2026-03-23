from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)


class AssetType(Enum):
    ETF = "ETF"
    STOCK = "STOCK"
    UNKNOWN = "UNKNOWN"


@dataclass
class CostComparison:
    ticker1: str
    ticker2: str
    expense_ratio1: float
    expense_ratio2: float
    cheaper_option: str
    cost_difference: float
    annual_savings_per_10k: float


@dataclass
class PortfolioMix:
    etf_weights: Dict[str, float]
    stock_weights: Dict[str, float]
    total_estimated_cost: float
    sector_exposure: Dict[str, float]
    diversification_score: float
    optimization_notes: List[str]


class ETFAllocator:
    KNOWN_ETFS = {
        "SPY",
        "VOO",
        "IVV",
        "QQQ",
        "QQQM",
        "VTI",
        "ITOT",
        "IWM",
        "VB",
        "XLK",
        "XLV",
        "XLF",
        "XLE",
        "XLI",
        "XLP",
        "XLY",
        "XLU",
        "XLB",
        "XLRE",
        "VEA",
        "IEFA",
        "VWO",
        "IEMG",
        "EFA",
        "AGG",
        "BND",
        "TLT",
        "IEF",
    }

    ETF_EXPENSE_RATIOS = {
        "VOO": 3,
        "SPY": 9,
        "IVV": 3,
        "QQQ": 20,
        "QQQM": 15,
        "VTI": 3,
        "IWM": 19,
        "XLK": 10,
        "XLV": 10,
        "XLF": 10,
        "XLE": 10,
        "VEA": 5,
        "VWO": 8,
    }

    SECTOR_MAP = {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "GOOGL": "Technology",
        "AMZN": "Technology",
        "NVDA": "Technology",
        "META": "Technology",
        "JNJ": "Healthcare",
        "UNH": "Healthcare",
        "PFE": "Healthcare",
        "ABBV": "Healthcare",
        "JPM": "Financials",
        "BAC": "Financials",
        "WFC": "Financials",
        "GS": "Financials",
        "WMT": "Consumer Staples",
        "PG": "Consumer Staples",
        "KO": "Consumer Staples",
        "HD": "Consumer Discretionary",
        "NKE": "Consumer Discretionary",
        "XOM": "Energy",
        "CVX": "Energy",
        "BA": "Industrials",
        "CAT": "Industrials",
        "GE": "Industrials",
    }

    def classify_asset_type(self, ticker: str) -> AssetType:
        ticker_upper = ticker.upper()

        if ticker_upper in self.KNOWN_ETFS:
            return AssetType.ETF
        if ticker_upper in self.SECTOR_MAP:
            return AssetType.STOCK

        logger.warning(f"Unknown ticker {ticker}, assuming STOCK")
        return AssetType.STOCK

    def analyze_sector_exposure(self, portfolio: Dict[str, float]) -> Dict[str, float]:
        sector_exposure: Dict[str, float] = {}

        for ticker, weight in portfolio.items():
            asset_type = self.classify_asset_type(ticker)

            if asset_type == AssetType.ETF:
                if ticker.upper() in ["VOO", "SPY", "IVV", "VTI"]:
                    broad_sectors = {
                        "Technology": 0.28,
                        "Healthcare": 0.13,
                        "Financials": 0.13,
                        "Consumer Discretionary": 0.11,
                        "Consumer Staples": 0.07,
                        "Industrials": 0.08,
                        "Energy": 0.04,
                        "Other": 0.16,
                    }
                    for sector, sector_weight in broad_sectors.items():
                        sector_exposure[sector] = (
                            sector_exposure.get(sector, 0) + weight * sector_weight
                        )
                elif ticker.upper() == "QQQ":
                    sector_exposure["Technology"] = (
                        sector_exposure.get("Technology", 0) + weight * 0.50
                    )
                    sector_exposure["Consumer Discretionary"] = (
                        sector_exposure.get("Consumer Discretionary", 0) + weight * 0.20
                    )
                    sector_exposure["Healthcare"] = (
                        sector_exposure.get("Healthcare", 0) + weight * 0.10
                    )
                    sector_exposure["Other"] = (
                        sector_exposure.get("Other", 0) + weight * 0.20
                    )
                elif ticker.upper().startswith("XL"):
                    sector_name = self._get_sector_from_etf(ticker.upper())
                    sector_exposure[sector_name] = (
                        sector_exposure.get(sector_name, 0) + weight
                    )
                else:
                    sector_exposure["Other"] = sector_exposure.get("Other", 0) + weight
            else:
                sector = self.SECTOR_MAP.get(ticker.upper(), "Other")
                sector_exposure[sector] = sector_exposure.get(sector, 0) + weight

        return sector_exposure

    def compare_etf_costs(self, ticker1: str, ticker2: str) -> CostComparison:
        expense1 = self.ETF_EXPENSE_RATIOS.get(ticker1.upper(), 20)
        expense2 = self.ETF_EXPENSE_RATIOS.get(ticker2.upper(), 20)

        cheaper = ticker1 if expense1 < expense2 else ticker2
        cost_diff = abs(expense1 - expense2)
        annual_savings = (cost_diff / 10000) * 10000

        return CostComparison(
            ticker1=ticker1,
            ticker2=ticker2,
            expense_ratio1=expense1 / 10000,
            expense_ratio2=expense2 / 10000,
            cheaper_option=cheaper,
            cost_difference=cost_diff,
            annual_savings_per_10k=annual_savings,
        )

    def calculate_optimal_mix(
        self,
        target_sectors: Dict[str, float],
        available_etfs: List[str],
        available_stocks: List[str],
        max_etf_weight: float = 0.7,
    ) -> PortfolioMix:
        etf_weights: Dict[str, float] = {}
        stock_weights: Dict[str, float] = {}
        notes: List[str] = []

        broad_etfs = [
            e for e in available_etfs if e.upper() in ["VOO", "VTI", "IVV", "SPY"]
        ]

        if broad_etfs:
            cheapest_broad = min(
                broad_etfs, key=lambda e: self.ETF_EXPENSE_RATIOS.get(e.upper(), 100)
            )
            base_weight = min(0.50, max_etf_weight)
            etf_weights[cheapest_broad] = base_weight
            notes.append(
                f"Base allocation: {base_weight:.1%} to {cheapest_broad} (low-cost broad)"
            )

        remaining_weight = 1.0 - sum(etf_weights.values())

        stock_sectors: Dict[str, List[str]] = {}
        for stock in available_stocks:
            sector = self.SECTOR_MAP.get(stock.upper(), "Other")
            if sector not in stock_sectors:
                stock_sectors[sector] = []
            stock_sectors[sector].append(stock)

        for sector, target_pct in target_sectors.items():
            if sector in stock_sectors and stock_sectors[sector]:
                stocks_in_sector = stock_sectors[sector]
                weight_per_stock = (target_pct * remaining_weight) / len(
                    stocks_in_sector
                )

                for stock in stocks_in_sector[:3]:
                    stock_weights[stock] = weight_per_stock
                    notes.append(
                        f"{stock}: {weight_per_stock:.1%} for {sector} exposure"
                    )

        total_weight = sum(etf_weights.values()) + sum(stock_weights.values())
        if total_weight > 0:
            etf_weights = {k: v / total_weight for k, v in etf_weights.items()}
            stock_weights = {k: v / total_weight for k, v in stock_weights.items()}

        total_cost = self._calculate_portfolio_cost(etf_weights, stock_weights)
        sector_exposure = self.analyze_sector_exposure({**etf_weights, **stock_weights})
        diversification_score = self._calculate_diversification_score(
            etf_weights, stock_weights
        )

        return PortfolioMix(
            etf_weights=etf_weights,
            stock_weights=stock_weights,
            total_estimated_cost=total_cost,
            sector_exposure=sector_exposure,
            diversification_score=diversification_score,
            optimization_notes=notes,
        )

    def _get_sector_from_etf(self, etf_ticker: str) -> str:
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
            "XLRE": "Real Estate",
        }
        return sector_etf_map.get(etf_ticker, "Other")

    def _calculate_portfolio_cost(
        self, etf_weights: Dict[str, float], stock_weights: Dict[str, float]
    ) -> float:
        total_cost = 0.0

        for etf, weight in etf_weights.items():
            expense_ratio = self.ETF_EXPENSE_RATIOS.get(etf.upper(), 20)
            total_cost += weight * expense_ratio

        return total_cost

    def _calculate_diversification_score(
        self, etf_weights: Dict[str, float], stock_weights: Dict[str, float]
    ) -> float:
        all_weights = {**etf_weights, **stock_weights}

        if not all_weights:
            return 0.0

        num_holdings = len(all_weights)
        herfindahl = sum(w**2 for w in all_weights.values())

        if num_holdings == 1:
            return 0.0

        max_diversification = 1.0 / num_holdings
        score = 1.0 - (herfindahl - max_diversification) / (1.0 - max_diversification)

        return max(0.0, min(1.0, score))


def get_low_cost_etf(sector: Optional[str] = None) -> str:
    if sector is None or sector == "Broad":
        return "VOO"

    sector_etfs = {
        "Technology": "XLK",
        "Healthcare": "XLV",
        "Financials": "XLF",
        "Energy": "XLE",
        "Industrials": "XLI",
        "Consumer Staples": "XLP",
        "Consumer Discretionary": "XLY",
    }

    return sector_etfs.get(sector, "VOO")


def estimate_portfolio_cost(portfolio: Dict[str, float]) -> float:
    allocator = ETFAllocator()

    etfs: Dict[str, float] = {}
    stocks: Dict[str, float] = {}

    for ticker, weight in portfolio.items():
        if allocator.classify_asset_type(ticker) == AssetType.ETF:
            etfs[ticker] = weight
        else:
            stocks[ticker] = weight

    return allocator._calculate_portfolio_cost(etfs, stocks)


def classify_asset_type(ticker: str) -> AssetType:
    allocator = ETFAllocator()
    return allocator.classify_asset_type(ticker)
