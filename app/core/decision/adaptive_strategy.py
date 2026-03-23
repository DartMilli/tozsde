from dataclasses import dataclass
from typing import Dict, List, Optional
import random


@dataclass
class StrategySelection:
    selected_strategies: Dict[str, float]
    selection_mode: str
    market_context: str
    confidence_in_selection: float


@dataclass
class StrategyBandit:
    name: str
    alpha: float
    beta: float
    total_trials: int

    @property
    def expected_value(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def uncertainty(self) -> float:
        variance = (self.alpha * self.beta) / (
            (self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1)
        )
        return variance**0.5


REGIME_PREFERENCES = {
    "BULL": ["MOMENTUM", "MA_CROSS"],
    "BEAR": ["RSI_MEAN", "BB_MEAN"],
    "RANGING": ["RSI_MEAN", "BB_MEAN", "MACD"],
    "VOLATILE": ["MA_CROSS", "MACD"],
}


def beta_sample(alpha: float, beta: float) -> float:
    return random.betavariate(alpha, beta)


def sample_strategy_weights(
    strategies: List[str],
    bandits: Dict[str, StrategyBandit],
) -> Dict[str, float]:
    samples = {}

    for strategy in strategies:
        bandit = bandits.get(strategy)

        if bandit is None:
            bandit = StrategyBandit(strategy, alpha=1.0, beta=1.0, total_trials=0)
            bandits[strategy] = bandit

        sample = beta_sample(bandit.alpha, bandit.beta)
        samples[strategy] = sample

    total = sum(samples.values())
    if total > 0:
        return {strategy: value / total for strategy, value in samples.items()}

    return {strategy: 1.0 / len(strategies) for strategy in strategies}


def select_best_strategy(bandits: Dict[str, StrategyBandit]) -> str:
    return max(bandits.items(), key=lambda item: item[1].expected_value)[0]


def select_strategy_by_context(
    market_regime: str,
    bandits: Dict[str, StrategyBandit],
    strategies: List[str],
) -> str:
    preferred = REGIME_PREFERENCES.get(market_regime, strategies)

    best = None
    best_value = -1.0

    for strategy in preferred:
        if strategy in bandits:
            bandit = bandits[strategy]
            if bandit.expected_value > best_value:
                best_value = bandit.expected_value
                best = strategy

    if best is None:
        return select_best_strategy(bandits)

    return best


def calculate_selection_confidence(weights: Dict[str, float]) -> float:
    entropy = 0.0
    for weight in weights.values():
        if weight > 0:
            entropy -= weight * (weight**0.5)

    max_entropy = len(weights) ** 0.5
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    confidence = 1.0 - abs(normalized_entropy)
    return max(0.0, min(1.0, confidence))
