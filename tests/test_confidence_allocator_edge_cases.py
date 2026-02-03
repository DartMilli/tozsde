"""Edge case tests for ConfidenceBucketAllocator."""

import sqlite3
from pathlib import Path

import pytest

from app.decision.confidence_allocator import ConfidenceBucketAllocator, ConfidenceBucket


def test_classify_confidence_bucket_boundaries():
    allocator = ConfidenceBucketAllocator()

    assert allocator.classify_confidence_bucket(0.75) == ConfidenceBucket.STRONG
    assert allocator.classify_confidence_bucket(0.5) == ConfidenceBucket.NORMAL
    assert allocator.classify_confidence_bucket(0.49) == ConfidenceBucket.WEAK

    with pytest.raises(ValueError):
        allocator.classify_confidence_bucket(1.5)


def test_allocate_and_history(tmp_path):
    db_path = tmp_path / "conf.db"
    allocator = ConfidenceBucketAllocator(db_path=str(db_path), base_capital=100.0)

    allocation = allocator.allocate_capital("strategy1", 0.8)
    assert allocation.allocated_capital == 150.0

    history = allocator.get_allocation_history()
    assert len(history) == 1


def test_bucket_statistics(tmp_path):
    db_path = tmp_path / "conf.db"
    allocator = ConfidenceBucketAllocator(db_path=str(db_path), base_capital=100.0)

    allocator.allocate_capital("s1", 0.8)
    allocator.allocate_capital("s2", 0.6)
    allocator.allocate_capital("s3", 0.3)

    stats = allocator.get_bucket_statistics()
    assert ConfidenceBucket.STRONG in stats
    assert ConfidenceBucket.NORMAL in stats
    assert ConfidenceBucket.WEAK in stats


def test_allocate_capital_by_bucket_with_rebalance():
    allocator = ConfidenceBucketAllocator(base_capital=100.0)
    strategies = {"s1": 0.8, "s2": 0.4}

    allocations = allocator.allocate_capital_by_bucket(strategies, rebalance_rate=0.1)

    assert allocations["s1"].allocated_capital > 150.0
    assert allocations["s2"].allocated_capital > 50.0


def test_suggest_rebalancing():
    allocator = ConfidenceBucketAllocator()
    current = {"s1": 200.0, "s2": 100.0, "s3": 100.0}

    suggestions = allocator.suggest_rebalancing(current)
    assert "s1" in suggestions
