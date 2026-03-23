from app.core.decision.adaptive_strategy import (  # noqa: F401
    StrategyBandit,
    StrategySelection,
)
from app.core.decision.adaptive_strategy_selector import (
    AdaptiveStrategySelector as CoreAdaptiveStrategySelector,
)
from app.decision import get_settings


class AdaptiveStrategySelector(CoreAdaptiveStrategySelector):
    def __init__(self, epsilon: float = 0.1, settings=None):
        super().__init__(epsilon=epsilon, settings=settings or get_settings())


def get_top_strategies(n: int = 3):
    selector = AdaptiveStrategySelector()
    stats = selector.get_strategy_stats()
    return [s["strategy"] for s in stats[:n]]


def recommend_strategy_for_regime(market_regime: str) -> str:
    selector = AdaptiveStrategySelector()
    return selector.select_strategy_by_context(market_regime)
