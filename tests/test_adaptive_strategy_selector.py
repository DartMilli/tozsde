"""
Tests for Adaptive Strategy Selector (P8 — Learning System)

Tests Thompson Sampling, epsilon-greedy exploration,
contextual bandit, and strategy performance updates.
"""

import pytest
import tempfile
import sqlite3
from pathlib import Path

from app.decision.adaptive_strategy_selector import (
    AdaptiveStrategySelector,
    StrategySelection,
    StrategyBandit,
    get_top_strategies,
    recommend_strategy_for_regime
)
from app.config.config import Config


@pytest.fixture
def test_db():
    """Create temporary test database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        Config.DB_PATH = db_path
        
        # Create empty database
        conn = sqlite3.connect(str(db_path))
        conn.close()
        
        yield db_path


def test_thompson_sampling_weights(test_db):
    """Test Thompson Sampling weight calculation."""
    selector = AdaptiveStrategySelector()
    
    weights = selector.select_strategy_weights()
    
    # Weights should sum to 1.0
    assert abs(sum(weights.values()) - 1.0) < 0.01
    
    # All weights should be non-negative
    assert all(w >= 0 for w in weights.values())
    
    # Should have all strategies
    assert len(weights) == len(selector.strategies)


def test_strategy_update_on_success(test_db):
    """Test bandit update after successful trade."""
    selector = AdaptiveStrategySelector()
    
    # Get initial alpha
    initial_alpha = selector.bandits["MA_CROSS"].alpha
    
    # Update with success
    selector.update_strategy("MA_CROSS", success=True)
    
    # Alpha should increase
    assert selector.bandits["MA_CROSS"].alpha == initial_alpha + 1.0
    assert selector.bandits["MA_CROSS"].total_trials == 1


def test_strategy_update_on_failure(test_db):
    """Test bandit update after failed trade."""
    selector = AdaptiveStrategySelector()
    
    # Get initial beta
    initial_beta = selector.bandits["RSI_MEAN"].beta
    
    # Update with failure
    selector.update_strategy("RSI_MEAN", success=False)
    
    # Beta should increase
    assert selector.bandits["RSI_MEAN"].beta == initial_beta + 1.0
    assert selector.bandits["RSI_MEAN"].total_trials == 1


def test_epsilon_greedy_exploration(test_db, monkeypatch):
    """Test epsilon-greedy exploration mode."""
    selector = AdaptiveStrategySelector(epsilon=1.0)  # Always explore
    
    # Force exploration by setting random to return < epsilon
    monkeypatch.setattr("random.random", lambda: 0.05)
    
    selection = selector.explore_or_exploit()
    
    assert selection.selection_mode == "EXPLORE"
    # Should select single strategy with weight 1.0
    selected_count = sum(1 for w in selection.selected_strategies.values() if w > 0.9)
    assert selected_count == 1


def test_epsilon_greedy_exploitation(test_db, monkeypatch):
    """Test epsilon-greedy exploitation mode."""
    selector = AdaptiveStrategySelector(epsilon=0.0)  # Never explore
    
    # Force exploitation
    monkeypatch.setattr("random.random", lambda: 0.95)
    
    selection = selector.explore_or_exploit()
    
    assert selection.selection_mode == "EXPLOIT"
    # Should have weights from Thompson Sampling
    assert len(selection.selected_strategies) > 1


def test_contextual_bandit_bull_regime(test_db):
    """Test strategy selection for bull market."""
    selector = AdaptiveStrategySelector()
    
    # Add some performance history
    selector.update_strategy("MOMENTUM", success=True)
    selector.update_strategy("MOMENTUM", success=True)
    selector.update_strategy("RSI_MEAN", success=False)
    
    strategy = selector.select_strategy_by_context("BULL")
    
    # Should prefer momentum strategies in bull market
    assert strategy in ["MOMENTUM", "MA_CROSS"]


def test_contextual_bandit_bear_regime(test_db):
    """Test strategy selection for bear market."""
    selector = AdaptiveStrategySelector()
    
    # Add some performance history
    selector.update_strategy("RSI_MEAN", success=True)
    selector.update_strategy("BB_MEAN", success=True)
    selector.update_strategy("MOMENTUM", success=False)
    
    strategy = selector.select_strategy_by_context("BEAR")
    
    # Should prefer mean reversion in bear market
    assert strategy in ["RSI_MEAN", "BB_MEAN"]


def test_weight_normalization(test_db):
    """Test that weights are properly normalized."""
    selector = AdaptiveStrategySelector()
    
    # Add varying performance
    for _ in range(20):
        selector.update_strategy("MA_CROSS", success=True)
    
    for _ in range(3):
        selector.update_strategy("RSI_MEAN", success=True)
    
    weights = selector.select_strategy_weights()
    
    # Weights must sum to 1.0
    assert abs(sum(weights.values()) - 1.0) < 0.001
    
    # MA_CROSS should have higher expected value (more successes)
    assert selector.bandits["MA_CROSS"].expected_value > selector.bandits["RSI_MEAN"].expected_value


def test_strategy_stats_calculation(test_db):
    """Test strategy statistics calculation."""
    selector = AdaptiveStrategySelector()
    
    # Add performance data
    selector.update_strategy("MA_CROSS", success=True)
    selector.update_strategy("MA_CROSS", success=True)
    selector.update_strategy("MA_CROSS", success=False)
    
    stats = selector.get_strategy_stats()
    
    # Should return stats for all strategies
    assert len(stats) > 0
    
    # Stats should have correct format
    ma_cross_stat = next(s for s in stats if s["strategy"] == "MA_CROSS")
    assert "expected_win_rate" in ma_cross_stat
    assert "total_trials" in ma_cross_stat
    assert ma_cross_stat["total_trials"] == 3
    assert ma_cross_stat["successes"] == 2
    assert ma_cross_stat["failures"] == 1


def test_strategy_reset(test_db):
    """Test strategy bandit reset."""
    selector = AdaptiveStrategySelector()
    
    # Add performance
    selector.update_strategy("MACD", success=True)
    selector.update_strategy("MACD", success=True)
    
    assert selector.bandits["MACD"].total_trials == 2
    
    # Reset
    selector.reset_strategy("MACD")
    
    # Should be back to prior
    assert selector.bandits["MACD"].alpha == 1.0
    assert selector.bandits["MACD"].beta == 1.0
    assert selector.bandits["MACD"].total_trials == 0


def test_get_top_strategies(test_db):
    """Test utility function to get top performing strategies."""
    selector = AdaptiveStrategySelector()
    
    # Create performance gradient
    for _ in range(15):
        selector.update_strategy("MA_CROSS", success=True)
    for _ in range(10):
        selector.update_strategy("MOMENTUM", success=True)
    for _ in range(5):
        selector.update_strategy("RSI_MEAN", success=True)
    
    top = get_top_strategies(n=2)
    
    assert len(top) <= 2
    assert "MA_CROSS" in top  # Should be top


def test_recommend_strategy_for_regime(test_db):
    """Test regime-based strategy recommendation."""
    strategy = recommend_strategy_for_regime("BULL")
    assert strategy in ["MOMENTUM", "MA_CROSS"]
    
    strategy = recommend_strategy_for_regime("BEAR")
    assert strategy in ["RSI_MEAN", "BB_MEAN"]
    
    strategy = recommend_strategy_for_regime("RANGING")
    assert strategy in ["RSI_MEAN", "BB_MEAN", "MACD"]


def test_confidence_calculation(test_db):
    """Test confidence score in strategy selection."""
    selector = AdaptiveStrategySelector(epsilon=0.0)  # Pure exploitation
    
    # Create strong preference for one strategy
    for _ in range(20):
        selector.update_strategy("MA_CROSS", success=True)
    
    selection = selector.explore_or_exploit()
    
    # High concentration should give high confidence
    assert selection.confidence_in_selection > 0.5


def test_bandit_persistence(test_db):
    """Test that bandit parameters persist to database."""
    selector1 = AdaptiveStrategySelector()
    selector1.update_strategy("MA_CROSS", success=True)
    selector1.update_strategy("MA_CROSS", success=True)
    
    alpha_before = selector1.bandits["MA_CROSS"].alpha
    
    # Create new selector (should load from DB)
    selector2 = AdaptiveStrategySelector()
    
    # Should have same alpha
    assert selector2.bandits["MA_CROSS"].alpha == alpha_before
