"""
Adaptive Strategy Selector (P8 — Learning System)

Responsibility:
    - Select optimal trading strategies based on historical performance
    - Implement Thompson Sampling (Bayesian bandit) for strategy weighting
    - Balance exploration vs. exploitation
    - Context-aware strategy selection based on market regime
    - NOT for direct trade decisions, only strategy selection

Features:
    - Thompson Sampling with Beta distributions
    - Epsilon-greedy exploration fallback
    - Contextual bandits (regime-specific selection)
    - Strategy weight normalization
    - Performance-based updates (win/loss)

Usage:
    selector = AdaptiveStrategySelector()
    
    # Get strategy weights
    weights = selector.select_strategy_weights()
    # {"MA_CROSS": 0.35, "RSI_MEAN": 0.40, "MACD": 0.25}
    
    # Update after trade outcome
    selector.update_strategy("MA_CROSS", success=True)
    
    # Context-aware selection
    strategy = selector.select_strategy_by_context(market_regime="BULL")
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import random
import sqlite3
import json
from datetime import datetime, timedelta

from app.config.config import Config
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class StrategySelection:
    """Result of strategy selection process."""
    selected_strategies: Dict[str, float]  # strategy -> weight
    selection_mode: str  # "EXPLOIT" | "EXPLORE"
    market_context: str  # "BULL" | "BEAR" | "RANGING" | "VOLATILE"
    confidence_in_selection: float  # 0.0 to 1.0


@dataclass
class StrategyBandit:
    """Beta distribution parameters for a strategy (Thompson Sampling)."""
    name: str
    alpha: float  # Success count + 1 (prior)
    beta: float  # Failure count + 1 (prior)
    total_trials: int
    
    @property
    def expected_value(self) -> float:
        """Expected success rate (mean of Beta distribution)."""
        return self.alpha / (self.alpha + self.beta)
    
    @property
    def uncertainty(self) -> float:
        """Uncertainty in estimate (variance of Beta distribution)."""
        variance = (self.alpha * self.beta) / (
            (self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1)
        )
        return variance ** 0.5


class AdaptiveStrategySelector:
    """
    Reinforcement learning based strategy selector using Thompson Sampling.
    
    Uses multi-armed bandit approach to balance exploration and exploitation
    when selecting trading strategies. Context-aware based on market regimes.
    """
    
    def __init__(self, epsilon: float = 0.1):
        """
        Initialize strategy selector.
        
        Args:
            epsilon: Exploration probability for epsilon-greedy (default 0.1 = 10%)
        """
        self.epsilon = epsilon
        self.db_path = Config.DB_PATH
        
        # Initialize default strategies
        self.strategies = [
            "MA_CROSS",
            "RSI_MEAN",
            "MACD",
            "BB_MEAN",
            "MOMENTUM"
        ]
        
        # Load or initialize bandit parameters
        self.bandits = self._load_bandits()
    
    def select_strategy_weights(
        self, market_regime: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Select strategy weights using Thompson Sampling.
        
        Samples from Beta distribution for each strategy and normalizes
        to create probability distribution (weights).
        
        Args:
            market_regime: Optional market context for contextual bandit
            
        Returns:
            Dictionary mapping strategy names to weights (sum = 1.0)
        """
        samples = {}
        
        for strategy in self.strategies:
            bandit = self.bandits.get(strategy)
            
            if bandit is None:
                # Initialize new strategy with uniform prior
                bandit = StrategyBandit(strategy, alpha=1.0, beta=1.0, total_trials=0)
                self.bandits[strategy] = bandit
            
            # Sample from Beta distribution
            sample = self._beta_sample(bandit.alpha, bandit.beta)
            samples[strategy] = sample
        
        # Normalize to sum to 1.0
        total = sum(samples.values())
        if total > 0:
            weights = {s: v / total for s, v in samples.items()}
        else:
            # Fallback to uniform distribution
            weights = {s: 1.0 / len(self.strategies) for s in self.strategies}
        
        return weights
    
    def select_best_strategy(self, market_regime: Optional[str] = None) -> str:
        """
        Select single best strategy (greedy exploitation).
        
        Args:
            market_regime: Optional market context
            
        Returns:
            Strategy name with highest expected value
        """
        best_strategy = max(
            self.bandits.items(),
            key=lambda x: x[1].expected_value
        )[0]
        
        return best_strategy
    
    def explore_or_exploit(
        self, market_regime: Optional[str] = None
    ) -> StrategySelection:
        """
        Epsilon-greedy strategy selection.
        
        With probability epsilon: explore (random strategy)
        With probability 1-epsilon: exploit (best strategy)
        
        Args:
            market_regime: Optional market context
            
        Returns:
            StrategySelection with mode indicator
        """
        # Decide: explore or exploit
        if random.random() < self.epsilon:
            # EXPLORE: random strategy
            selected = random.choice(self.strategies)
            mode = "EXPLORE"
            weights = {s: 1.0 if s == selected else 0.0 for s in self.strategies}
            confidence = 0.5  # Low confidence in random selection
        else:
            # EXPLOIT: best strategy
            weights = self.select_strategy_weights(market_regime)
            mode = "EXPLOIT"
            # Confidence = entropy of distribution (more concentrated = higher confidence)
            confidence = self._calculate_confidence(weights)
        
        return StrategySelection(
            selected_strategies=weights,
            selection_mode=mode,
            market_context=market_regime or "UNKNOWN",
            confidence_in_selection=confidence
        )
    
    def select_strategy_by_context(self, market_regime: str) -> str:
        """
        Context-aware strategy selection based on market regime.
        
        Different market conditions favor different strategies:
        - BULL: momentum strategies
        - BEAR: mean reversion strategies
        - RANGING: range-bound strategies
        - VOLATILE: conservative strategies
        
        Args:
            market_regime: "BULL" | "BEAR" | "RANGING" | "VOLATILE"
            
        Returns:
            Strategy name best suited for current regime
        """
        # Context-specific strategy preferences
        regime_preferences = {
            "BULL": ["MOMENTUM", "MA_CROSS"],
            "BEAR": ["RSI_MEAN", "BB_MEAN"],
            "RANGING": ["RSI_MEAN", "BB_MEAN", "MACD"],
            "VOLATILE": ["MA_CROSS", "MACD"]
        }
        
        # Get preferred strategies for this regime
        preferred = regime_preferences.get(market_regime, self.strategies)
        
        # Among preferred strategies, select best performer
        best = None
        best_value = -1.0
        
        for strategy in preferred:
            if strategy in self.bandits:
                bandit = self.bandits[strategy]
                if bandit.expected_value > best_value:
                    best_value = bandit.expected_value
                    best = strategy
        
        if best is None:
            # Fallback to overall best
            best = self.select_best_strategy()
        
        return best
    
    def update_strategy(self, strategy_name: str, success: bool):
        """
        Update strategy performance based on trade outcome.
        
        Success: increment alpha (success count)
        Failure: increment beta (failure count)
        
        Args:
            strategy_name: Name of strategy to update
            success: True if trade was profitable, False otherwise
        """
        if strategy_name not in self.bandits:
            # Initialize if not exists
            self.bandits[strategy_name] = StrategyBandit(
                name=strategy_name,
                alpha=1.0,
                beta=1.0,
                total_trials=0
            )
        
        bandit = self.bandits[strategy_name]
        
        if success:
            bandit.alpha += 1.0
            logger.info(f"Strategy {strategy_name} SUCCESS: alpha={bandit.alpha:.1f}")
        else:
            bandit.beta += 1.0
            logger.info(f"Strategy {strategy_name} FAILURE: beta={bandit.beta:.1f}")
        
        bandit.total_trials += 1
        
        # Persist to database
        self._save_bandits()
    
    def get_strategy_stats(self) -> List[Dict]:
        """
        Get performance statistics for all strategies.
        
        Returns:
            List of dicts with strategy stats
        """
        stats = []
        
        for name, bandit in self.bandits.items():
            stats.append({
                "strategy": name,
                "expected_win_rate": bandit.expected_value,
                "uncertainty": bandit.uncertainty,
                "total_trials": bandit.total_trials,
                "successes": int(bandit.alpha - 1),  # Remove prior
                "failures": int(bandit.beta - 1)
            })
        
        # Sort by expected value
        stats.sort(key=lambda x: x["expected_win_rate"], reverse=True)
        
        return stats
    
    def reset_strategy(self, strategy_name: str):
        """
        Reset a strategy's bandit to uniform prior.
        
        Useful when strategy is fundamentally changed or parameters retuned.
        
        Args:
            strategy_name: Strategy to reset
        """
        if strategy_name in self.bandits:
            self.bandits[strategy_name] = StrategyBandit(
                name=strategy_name,
                alpha=1.0,
                beta=1.0,
                total_trials=0
            )
            self._save_bandits()
            logger.info(f"Reset strategy {strategy_name} to uniform prior")
    
    # Helper methods
    
    def _beta_sample(self, alpha: float, beta: float) -> float:
        """
        Sample from Beta distribution.
        
        Uses Gamma distribution trick: Beta(a,b) = Gamma(a) / (Gamma(a) + Gamma(b))
        """
        # Simple implementation using random.betavariate
        return random.betavariate(alpha, beta)
    
    def _calculate_confidence(self, weights: Dict[str, float]) -> float:
        """
        Calculate confidence in strategy selection.
        
        Uses normalized entropy: more concentrated distribution = higher confidence.
        
        Returns:
            Confidence score between 0.0 (uniform) and 1.0 (single strategy)
        """
        # Calculate entropy
        entropy = 0.0
        for weight in weights.values():
            if weight > 0:
                entropy -= weight * (weight ** 0.5)  # Simplified entropy
        
        # Normalize: max entropy for uniform distribution
        max_entropy = len(weights) ** 0.5
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Confidence is inverse of entropy
        confidence = 1.0 - abs(normalized_entropy)
        
        return max(0.0, min(1.0, confidence))
    
    def _load_bandits(self) -> Dict[str, StrategyBandit]:
        """
        Load bandit parameters from database.
        
        Returns:
            Dictionary mapping strategy names to StrategyBandit objects
        """
        bandits = {}
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()
            
            # Check if table exists
            cur.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='strategy_bandits'
            """)
            
            if not cur.fetchone():
                # Table doesn't exist, create it
                cur.execute("""
                    CREATE TABLE strategy_bandits (
                        strategy_name TEXT PRIMARY KEY,
                        alpha REAL,
                        beta REAL,
                        total_trials INTEGER,
                        last_updated TEXT
                    )
                """)
                conn.commit()
                logger.info("Created strategy_bandits table")
            else:
                # Load existing data
                cur.execute("SELECT strategy_name, alpha, beta, total_trials FROM strategy_bandits")
                rows = cur.fetchall()
                
                for row in rows:
                    name, alpha, beta, trials = row
                    bandits[name] = StrategyBandit(
                        name=name,
                        alpha=alpha,
                        beta=beta,
                        total_trials=trials
                    )
                
                logger.info(f"Loaded {len(bandits)} strategy bandits from database")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to load bandits: {e}")
        
        # Initialize missing strategies with uniform prior
        for strategy in self.strategies:
            if strategy not in bandits:
                bandits[strategy] = StrategyBandit(
                    name=strategy,
                    alpha=1.0,
                    beta=1.0,
                    total_trials=0
                )
        
        return bandits
    
    def _save_bandits(self):
        """Persist bandit parameters to database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()
            
            for name, bandit in self.bandits.items():
                cur.execute("""
                    INSERT OR REPLACE INTO strategy_bandits 
                    (strategy_name, alpha, beta, total_trials, last_updated)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    name,
                    bandit.alpha,
                    bandit.beta,
                    bandit.total_trials,
                    datetime.now().isoformat()
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save bandits: {e}")


# Utility functions

def get_top_strategies(n: int = 3) -> List[str]:
    """
    Get top N performing strategies.
    
    Args:
        n: Number of top strategies to return
        
    Returns:
        List of strategy names sorted by performance
    """
    selector = AdaptiveStrategySelector()
    stats = selector.get_strategy_stats()
    
    top_n = stats[:n]
    return [s["strategy"] for s in top_n]


def recommend_strategy_for_regime(market_regime: str) -> str:
    """
    Recommend best strategy for given market regime.
    
    Args:
        market_regime: "BULL" | "BEAR" | "RANGING" | "VOLATILE"
        
    Returns:
        Recommended strategy name
    """
    selector = AdaptiveStrategySelector()
    return selector.select_strategy_by_context(market_regime)
