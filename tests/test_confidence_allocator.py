"""
Tests for ConfidenceBucketAllocator module.
"""

import pytest
from datetime import datetime
from app.decision.confidence_allocator import (
    ConfidenceBucketAllocator,
    ConfidenceBucket,
    ConfidenceAllocation
)


@pytest.fixture
def allocator():
    """Create a test allocator."""
    return ConfidenceBucketAllocator(base_capital=1000.0)


class TestConfidenceBucketClassification:
    """Test confidence bucket classification."""
    
    def test_classify_strong_confidence(self, allocator):
        """Test STRONG bucket classification (confidence >= 0.75)."""
        bucket = allocator.classify_confidence_bucket(0.75)
        assert bucket == ConfidenceBucket.STRONG
        
        bucket = allocator.classify_confidence_bucket(0.95)
        assert bucket == ConfidenceBucket.STRONG
    
    def test_classify_normal_confidence(self, allocator):
        """Test NORMAL bucket classification (0.5 <= confidence < 0.75)."""
        bucket = allocator.classify_confidence_bucket(0.50)
        assert bucket == ConfidenceBucket.NORMAL
        
        bucket = allocator.classify_confidence_bucket(0.65)
        assert bucket == ConfidenceBucket.NORMAL
        
        bucket = allocator.classify_confidence_bucket(0.74)
        assert bucket == ConfidenceBucket.NORMAL
    
    def test_classify_weak_confidence(self, allocator):
        """Test WEAK bucket classification (confidence < 0.5)."""
        bucket = allocator.classify_confidence_bucket(0.25)
        assert bucket == ConfidenceBucket.WEAK
        
        bucket = allocator.classify_confidence_bucket(0.01)
        assert bucket == ConfidenceBucket.WEAK
    
    def test_invalid_confidence_too_high(self, allocator):
        """Test error on invalid confidence > 1.0."""
        with pytest.raises(ValueError):
            allocator.classify_confidence_bucket(1.5)
    
    def test_invalid_confidence_negative(self, allocator):
        """Test error on negative confidence."""
        with pytest.raises(ValueError):
            allocator.classify_confidence_bucket(-0.1)


class TestMultiplierRetrieval:
    """Test multiplier retrieval for buckets."""
    
    def test_strong_multiplier(self, allocator):
        """Test STRONG bucket multiplier is 1.5x."""
        multiplier = allocator.get_multiplier(ConfidenceBucket.STRONG)
        assert multiplier == 1.5
    
    def test_normal_multiplier(self, allocator):
        """Test NORMAL bucket multiplier is 1.0x."""
        multiplier = allocator.get_multiplier(ConfidenceBucket.NORMAL)
        assert multiplier == 1.0
    
    def test_weak_multiplier(self, allocator):
        """Test WEAK bucket multiplier is 0.5x."""
        multiplier = allocator.get_multiplier(ConfidenceBucket.WEAK)
        assert multiplier == 0.5


class TestCapitalAllocation:
    """Test capital allocation."""
    
    def test_allocate_strong_strategy(self, allocator):
        """Test allocation to STRONG confidence strategy."""
        allocation = allocator.allocate_capital("test_strategy", 0.85)
        
        assert allocation.strategy == "test_strategy"
        assert allocation.confidence_score == 0.85
        assert allocation.bucket == ConfidenceBucket.STRONG
        assert allocation.multiplier == 1.5
        assert allocation.allocated_capital == 1500.0  # 1000 * 1.5
        assert allocation.base_capital == 1000.0
    
    def test_allocate_normal_strategy(self, allocator):
        """Test allocation to NORMAL confidence strategy."""
        allocation = allocator.allocate_capital("test_strategy", 0.60)
        
        assert allocation.bucket == ConfidenceBucket.NORMAL
        assert allocation.multiplier == 1.0
        assert allocation.allocated_capital == 1000.0  # 1000 * 1.0
    
    def test_allocate_weak_strategy(self, allocator):
        """Test allocation to WEAK confidence strategy."""
        allocation = allocator.allocate_capital("test_strategy", 0.30)
        
        assert allocation.bucket == ConfidenceBucket.WEAK
        assert allocation.multiplier == 0.5
        assert allocation.allocated_capital == 500.0  # 1000 * 0.5


class TestMultipleAllocations:
    """Test allocating to multiple strategies."""
    
    def test_allocate_multiple_strategies(self, allocator):
        """Test allocating to multiple strategies."""
        strategies = {
            "strategy_a": 0.85,  # STRONG
            "strategy_b": 0.65,  # NORMAL
            "strategy_c": 0.30   # WEAK
        }
        
        allocations = allocator.allocate_capital_by_bucket(strategies)
        
        assert len(allocations) == 3
        assert allocations["strategy_a"].bucket == ConfidenceBucket.STRONG
        assert allocations["strategy_b"].bucket == ConfidenceBucket.NORMAL
        assert allocations["strategy_c"].bucket == ConfidenceBucket.WEAK
    
    def test_allocate_with_rebalancing(self, allocator):
        """Test allocation with rebalancing rate."""
        strategies = {
            "strategy_a": 0.80,
            "strategy_b": 0.60
        }
        
        allocations = allocator.allocate_capital_by_bucket(strategies, rebalance_rate=0.1)
        
        # With 10% rebalance rate, allocation should be increased
        assert allocations["strategy_a"].allocated_capital == 1500.0 * 1.1
        assert allocations["strategy_b"].allocated_capital == 1000.0 * 1.1
    
    def test_invalid_confidence_in_multiple(self, allocator):
        """Test error on invalid confidence in multi-allocation."""
        strategies = {
            "strategy_a": 0.80,
            "strategy_b": "invalid"  # Not numeric
        }
        
        with pytest.raises(ValueError):
            allocator.allocate_capital_by_bucket(strategies)


class TestBucketStatistics:
    """Test bucket statistics retrieval."""
    
    def test_bucket_statistics_empty(self, allocator):
        """Test bucket statistics when no data."""
        stats = allocator.get_bucket_statistics()
        assert len(stats) == 0  # No persistence without db_path


class TestAllocationHistory:
    """Test allocation history retrieval."""
    
    def test_allocation_history_empty(self, allocator):
        """Test allocation history when no data."""
        history = allocator.get_allocation_history()
        assert len(history) == 0  # No persistence without db_path


class TestRebalancingSuggestions:
    """Test rebalancing suggestions."""
    
    def test_suggest_no_rebalancing(self, allocator):
        """Test no rebalancing needed when allocations are balanced."""
        current = {
            "strategy_a": 1000.0,
            "strategy_b": 1000.0,
            "strategy_c": 1000.0
        }
        
        suggestions = allocator.suggest_rebalancing(current)
        assert len(suggestions) == 0
    
    def test_suggest_rebalancing_needed(self, allocator):
        """Test rebalancing suggested when allocations deviate."""
        current = {
            "strategy_a": 500.0,   # 50% deviation
            "strategy_b": 1000.0,
            "strategy_c": 1000.0
        }
        
        suggestions = allocator.suggest_rebalancing(current)
        assert "strategy_a" in suggestions
        assert suggestions["strategy_a"] == pytest.approx(833.33, rel=0.01)
    
    def test_suggest_rebalancing_empty(self, allocator):
        """Test rebalancing with empty allocations."""
        suggestions = allocator.suggest_rebalancing({})
        assert len(suggestions) == 0


class TestEdgeCases:
    """Test edge cases."""
    
    def test_zero_confidence(self, allocator):
        """Test allocation with zero confidence."""
        allocation = allocator.allocate_capital("test", 0.0)
        assert allocation.bucket == ConfidenceBucket.WEAK
        assert allocation.multiplier == 0.5
    
    def test_max_confidence(self, allocator):
        """Test allocation with maximum confidence."""
        allocation = allocator.allocate_capital("test", 1.0)
        assert allocation.bucket == ConfidenceBucket.STRONG
        assert allocation.multiplier == 1.5
    
    def test_boundary_confidence_75(self, allocator):
        """Test boundary case at 0.75 confidence."""
        allocation = allocator.allocate_capital("test", 0.75)
        assert allocation.bucket == ConfidenceBucket.STRONG
    
    def test_boundary_confidence_50(self, allocator):
        """Test boundary case at 0.50 confidence."""
        allocation = allocator.allocate_capital("test", 0.50)
        assert allocation.bucket == ConfidenceBucket.NORMAL
    
    def test_custom_base_capital(self):
        """Test with custom base capital."""
        allocator = ConfidenceBucketAllocator(base_capital=5000.0)
        allocation = allocator.allocate_capital("test", 0.85)
        assert allocation.allocated_capital == 7500.0  # 5000 * 1.5
