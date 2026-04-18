from typing import Dict, List

import numpy as np

from app.config.build_settings import build_settings
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)


class RiskParityAllocator:
    """Allocates capital across positions using risk-parity (inverse volatility) weighting.

    Args:
        lookback_days: Rolling window for volatility calculation.
        settings: Strongly recommended – pass an explicit Settings instance from the
            DI container.  If omitted, ``build_settings()`` is used as a fallback,
            which performs I/O and hinders isolated unit testing.  This fallback is
            **deprecated** (A1 – Sprint 14 tech-debt) and will be removed later.
    """

    def __init__(self, lookback_days: int = 60, settings=None):
        self.lookback_days = lookback_days
        self.settings = settings
        if settings is None:
            logger.debug(
                "RiskParityAllocator instantiated without explicit settings; "
                "build_settings() fallback will be used. Pass settings= for "
                "clean-architecture compliance."
            )

    def allocate(
        self,
        decisions: List[Dict],
        price_history: Dict[str, np.ndarray],
        current_equity: float | None = None,
    ) -> List[Dict]:
        logger.info(f"Risk parity allocation for {len(decisions)} positions")

        tradeable_indices = [
            i
            for i, d in enumerate(decisions)
            if d.get("decision", {}).get("action_code", d.get("action_code", 0)) != 0
        ]

        if not tradeable_indices:
            logger.warning("No tradeable positions for risk parity allocation")
            return decisions

        tradeable_tickers = [decisions[i]["ticker"] for i in tradeable_indices]

        volatilities = self._compute_volatilities(price_history)
        weights = self._compute_inverse_volatility_weights(
            volatilities, tradeable_tickers
        )
        decisions = self._apply_allocation(
            decisions,
            weights,
            tradeable_indices,
            current_equity=current_equity,
        )

        return decisions

    def _compute_volatilities(
        self, price_history: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        volatilities = {}

        for ticker, prices in price_history.items():
            if len(prices) < 2:
                volatilities[ticker] = 0.1
                continue

            returns = np.diff(prices) / prices[:-1]

            daily_std = np.std(returns)
            annualized_vol = daily_std * np.sqrt(252)

            volatilities[ticker] = max(annualized_vol, 0.01)
            logger.debug(f"{ticker}: volatility = {annualized_vol:.2%}")

        return volatilities

    def _compute_inverse_volatility_weights(
        self, volatilities: Dict[str, float], tradeable_tickers: List[str]
    ) -> np.ndarray:
        inv_vols = np.array(
            [1.0 / max(volatilities.get(t, 0.1), 0.01) for t in tradeable_tickers]
        )
        weights = inv_vols / inv_vols.sum()

        for ticker, weight in zip(tradeable_tickers, weights):
            vol = volatilities.get(ticker, 0.1)
            logger.debug(
                f"Risk parity weight: {ticker} = {weight:.2%} "
                f"(volatility = {vol:.2%})"
            )

        return weights

    def _apply_allocation(
        self,
        decisions: List[Dict],
        weights: np.ndarray,
        tradeable_indices: List[int],
        current_equity: float | None = None,
    ) -> List[Dict]:
        cfg = self._get_settings()
        initial_capital = current_equity or getattr(cfg, "INITIAL_CAPITAL")
        reserve_pct = getattr(cfg, "RISK_PARITY_RESERVE_PCT", 0.05)
        capital = initial_capital * (1.0 - reserve_pct)

        total_allocated = 0.0

        for idx, weight in zip(tradeable_indices, weights):
            allocation_amt = capital * weight
            decisions[idx]["allocation_amount"] = round(allocation_amt, 2)
            decisions[idx]["allocation_pct"] = round(weight, 4)
            total_allocated += allocation_amt

            logger.info(
                f"{decisions[idx]['ticker']}: {weight:.1%} allocation "
                f"(${allocation_amt:,.2f})"
            )

        logger.info(f"Total allocated: ${total_allocated:,.2f} / ${capital:,.2f}")

        return decisions

    def _get_settings(self):
        return self.settings or build_settings()


def apply_risk_parity(
    decisions: List[Dict],
    price_history: Dict[str, np.ndarray],
    current_equity: float | None = None,
) -> List[Dict]:
    allocator = RiskParityAllocator(lookback_days=60)
    return allocator.allocate(
        decisions,
        price_history,
        current_equity=current_equity,
    )
