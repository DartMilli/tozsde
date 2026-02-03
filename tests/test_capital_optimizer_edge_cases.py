"""Edge case tests for CapitalUtilizationOptimizer."""

import sqlite3
from pathlib import Path

from app.decision.capital_optimizer import CapitalUtilizationOptimizer


def test_kelly_fraction_bounds():
    optimizer = CapitalUtilizationOptimizer()

    assert optimizer.calculate_kelly_fraction(0.5, avg_win=0, avg_loss=1) == 0.0
    assert optimizer.calculate_kelly_fraction(0.5, avg_win=1, avg_loss=0) == 0.0

    kelly = optimizer.calculate_kelly_fraction(1.0, avg_win=2.0, avg_loss=1.0)
    assert 0.01 <= kelly <= 0.50


def test_optimal_position_size_limits(tmp_path):
    db_path = tmp_path / "cap.db"
    optimizer = CapitalUtilizationOptimizer(
        total_capital=1000.0,
        max_position_pct=0.1,
        min_position_size=50.0,
        db_path=str(db_path)
    )

    position = optimizer.calculate_optimal_position_size(
        ticker="AAPL",
        kelly_fraction=0.5,
        volatility=0.0
    )

    assert position.risk_adjusted_size == 100.0
    assert position.portfolio_weight == 0.1


def test_optimize_allocation_and_history(tmp_path):
    db_path = tmp_path / "cap.db"
    optimizer = CapitalUtilizationOptimizer(
        total_capital=1000.0,
        max_position_pct=0.2,
        min_position_size=50.0,
        db_path=str(db_path)
    )

    positions = {
        "AAPL": {"kelly": 0.2, "volatility": 0.05},
        "MSFT": {"kelly": 0.3, "volatility": 0.10},
    }

    allocation = optimizer.optimize_capital_allocation(positions)

    assert allocation.allocated_capital > 0
    assert allocation.unused_capital >= 0
    assert allocation.utilization_rate > 0

    history = optimizer.get_position_history()
    assert len(history) >= 2


def test_estimate_max_drawdown_caps():
    optimizer = CapitalUtilizationOptimizer(total_capital=1000.0)
    positions = {"AAPL": 900.0}

    dd = optimizer.estimate_max_drawdown(positions, volatility_avg=1.0)

    assert dd <= 1.0
