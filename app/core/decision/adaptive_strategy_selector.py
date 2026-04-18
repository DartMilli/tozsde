"""
Adaptive Strategy Selector (P8 - Learning System)
"""

from typing import Dict, List, Optional
import random
from datetime import datetime

from app.core.decision.adaptive_strategy import (
    StrategyBandit,
    StrategySelection,
    beta_sample,
    calculate_selection_confidence,
    sample_strategy_weights,
    select_best_strategy,
    select_strategy_by_context,
)
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)


class AdaptiveStrategySelector:
    def __init__(self, epsilon: float = 0.1, settings=None, data_manager=None):
        self.epsilon = epsilon
        self.settings = settings
        self.db_path = getattr(settings, "DB_PATH", None)

        self._dm = data_manager
        if self._dm is None:
            try:
                from app.infrastructure.repositories.data_manager_repository import (
                    DataManagerRepository as _DM,
                )

                self._dm = _DM(settings=settings)
            except Exception:
                self._dm = None

        self.strategies = ["MA_CROSS", "RSI_MEAN", "MACD", "BB_MEAN", "MOMENTUM"]
        self.bandits = self._load_bandits()

    def select_strategy_weights(
        self, market_regime: Optional[str] = None
    ) -> Dict[str, float]:
        return sample_strategy_weights(self.strategies, self.bandits)

    def select_best_strategy(self, market_regime: Optional[str] = None) -> str:
        return select_best_strategy(self.bandits)

    def explore_or_exploit(
        self, market_regime: Optional[str] = None
    ) -> StrategySelection:
        if random.random() < self.epsilon:
            selected = random.choice(self.strategies)
            mode = "EXPLORE"
            weights = {s: 1.0 if s == selected else 0.0 for s in self.strategies}
            confidence = 0.5
        else:
            weights = self.select_strategy_weights(market_regime)
            mode = "EXPLOIT"
            confidence = calculate_selection_confidence(weights)

        return StrategySelection(
            selected_strategies=weights,
            selection_mode=mode,
            market_context=market_regime or "UNKNOWN",
            confidence_in_selection=confidence,
        )

    def select_strategy_by_context(self, market_regime: str) -> str:
        return select_strategy_by_context(
            market_regime=market_regime,
            bandits=self.bandits,
            strategies=self.strategies,
        )

    def update_strategy(self, strategy_name: str, success: bool):
        if strategy_name not in self.bandits:
            self.bandits[strategy_name] = StrategyBandit(
                name=strategy_name, alpha=1.0, beta=1.0, total_trials=0
            )

        bandit = self.bandits[strategy_name]

        if success:
            bandit.alpha += 1.0
            logger.info(f"Strategy {strategy_name} SUCCESS: alpha={bandit.alpha:.1f}")
        else:
            bandit.beta += 1.0
            logger.info(f"Strategy {strategy_name} FAILURE: beta={bandit.beta:.1f}")

        bandit.total_trials += 1
        self._save_bandits()

    def get_strategy_stats(self) -> List[Dict]:
        stats = []

        for name, bandit in self.bandits.items():
            stats.append(
                {
                    "strategy": name,
                    "expected_win_rate": bandit.expected_value,
                    "uncertainty": bandit.uncertainty,
                    "total_trials": bandit.total_trials,
                    "successes": int(bandit.alpha - 1),
                    "failures": int(bandit.beta - 1),
                }
            )

        stats.sort(key=lambda x: x["expected_win_rate"], reverse=True)
        return stats

    def reset_strategy(self, strategy_name: str):
        if strategy_name in self.bandits:
            self.bandits[strategy_name] = StrategyBandit(
                name=strategy_name, alpha=1.0, beta=1.0, total_trials=0
            )
            self._save_bandits()
            logger.info(f"Reset strategy {strategy_name} to uniform prior")

    def _beta_sample(self, alpha: float, beta: float) -> float:
        return beta_sample(alpha, beta)

    def _calculate_confidence(self, weights: Dict[str, float]) -> float:
        return calculate_selection_confidence(weights)

    def _load_bandits(self) -> Dict[str, StrategyBandit]:
        bandits = {}

        try:
            if self._dm is None:
                raise Exception("No DataManager available")

            with self._dm.connection() as conn:
                cur = conn.cursor()

                cur.execute(
                    """
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='strategy_bandits'
                """
                )

                if not cur.fetchone():
                    cur.execute(
                        """
                        CREATE TABLE strategy_bandits (
                            strategy_name TEXT PRIMARY KEY,
                            alpha REAL,
                            beta REAL,
                            total_trials INTEGER,
                            last_updated TEXT
                        )
                    """
                    )
                    conn.commit()
                    logger.info("Created strategy_bandits table")
                else:
                    cur.execute(
                        "SELECT strategy_name, alpha, beta, total_trials FROM strategy_bandits"
                    )
                    rows = cur.fetchall()

                    for row in rows:
                        name, alpha, beta, trials = row
                        bandits[name] = StrategyBandit(
                            name=name, alpha=alpha, beta=beta, total_trials=trials
                        )

                    logger.info(f"Loaded {len(bandits)} strategy bandits from database")

        except Exception as e:
            logger.error(f"Failed to load bandits: {e}")

        for strategy in self.strategies:
            if strategy not in bandits:
                bandits[strategy] = StrategyBandit(
                    name=strategy, alpha=1.0, beta=1.0, total_trials=0
                )

        return bandits

    def _save_bandits(self):
        try:
            if self._dm is None:
                return

            with self._dm.connection() as conn:
                cur = conn.cursor()

                for name, bandit in self.bandits.items():
                    cur.execute(
                        """
                        INSERT OR REPLACE INTO strategy_bandits
                        (strategy_name, alpha, beta, total_trials, last_updated)
                        VALUES (?, ?, ?, ?, ?)
                    """,
                        (
                            name,
                            bandit.alpha,
                            bandit.beta,
                            bandit.total_trials,
                            datetime.now().isoformat(),
                        ),
                    )

                conn.commit()

        except Exception as e:
            logger.error(f"Failed to save bandits: {e}")

    def _resolve_db_path(self):
        if self.db_path:
            return self.db_path
        return getattr(self.settings, "DB_PATH", None)


def get_top_strategies(n: int = 3, settings=None) -> List[str]:
    selector = AdaptiveStrategySelector(settings=settings)
    stats = selector.get_strategy_stats()
    top_n = stats[:n]
    return [s["strategy"] for s in top_n]


def recommend_strategy_for_regime(market_regime: str, settings=None) -> str:
    selector = AdaptiveStrategySelector(settings=settings)
    return selector.select_strategy_by_context(market_regime)
