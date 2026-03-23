from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import json
import logging
import sqlite3

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    ticker: str
    optimal_size: float
    kelly_fraction: float
    max_position_limit: float
    risk_adjusted_size: float
    volatility: float
    portfolio_weight: float
    timestamp: datetime


@dataclass
class CapitalAllocation:
    total_capital: float
    allocated_capital: float
    unused_capital: float
    utilization_rate: float
    positions: Dict[str, float]
    diversification_score: float
    timestamp: datetime


class CapitalUtilizationOptimizer:
    def __init__(
        self,
        total_capital: float = 100000.0,
        max_position_pct: float = 0.05,
        min_position_size: float = 100.0,
        db_path: str = None,
    ):
        self.total_capital = total_capital
        self.max_position_pct = max_position_pct
        self.min_position_size = min_position_size
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        if not self.db_path:
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS position_sizes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    optimal_size REAL NOT NULL,
                    kelly_fraction REAL NOT NULL,
                    max_position_limit REAL NOT NULL,
                    risk_adjusted_size REAL NOT NULL,
                    volatility REAL NOT NULL,
                    portfolio_weight REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS capital_allocations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_capital REAL NOT NULL,
                    allocated_capital REAL NOT NULL,
                    unused_capital REAL NOT NULL,
                    utilization_rate REAL NOT NULL,
                    diversification_score REAL NOT NULL,
                    positions_json TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")

    def calculate_kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        if avg_win <= 0 or avg_loss <= 0:
            return 0.0

        loss_rate = 1.0 - win_rate
        kelly = (win_rate * avg_win - loss_rate * avg_loss) / avg_win
        kelly = max(0.01, min(0.50, kelly))

        return kelly

    def calculate_optimal_position_size(
        self,
        ticker: str,
        kelly_fraction: float = 0.25,
        volatility: float = 0.02,
        max_position_override: Optional[float] = None,
    ) -> PositionSize:
        risk_adjusted_kelly = kelly_fraction * (1.0 - volatility)
        optimal_size = self.total_capital * risk_adjusted_kelly

        max_limit = max_position_override or (
            self.total_capital * self.max_position_pct
        )

        risk_adjusted_size = min(optimal_size, max_limit)
        risk_adjusted_size = max(self.min_position_size, risk_adjusted_size)

        portfolio_weight = risk_adjusted_size / self.total_capital

        position = PositionSize(
            ticker=ticker,
            optimal_size=optimal_size,
            kelly_fraction=kelly_fraction,
            max_position_limit=max_limit,
            risk_adjusted_size=risk_adjusted_size,
            volatility=volatility,
            portfolio_weight=portfolio_weight,
            timestamp=datetime.now(),
        )

        self._persist_position_size(position)
        return position

    def optimize_capital_allocation(
        self,
        positions: Dict[str, Dict],
        rebalance: bool = False,
    ) -> CapitalAllocation:
        allocated_positions = {}
        total_allocated = 0.0

        for ticker, params in positions.items():
            kelly = params.get("kelly", 0.25)
            volatility = params.get("volatility", 0.02)

            pos_size = self.calculate_optimal_position_size(ticker, kelly, volatility)
            allocated_positions[ticker] = pos_size.risk_adjusted_size
            total_allocated += pos_size.risk_adjusted_size

        diversification_score = self._calculate_diversification_score(
            allocated_positions
        )

        unused_capital = max(0.0, self.total_capital - total_allocated)
        utilization_rate = (
            total_allocated / self.total_capital if self.total_capital > 0 else 0.0
        )

        allocation = CapitalAllocation(
            total_capital=self.total_capital,
            allocated_capital=total_allocated,
            unused_capital=unused_capital,
            utilization_rate=utilization_rate,
            positions=allocated_positions,
            diversification_score=diversification_score,
            timestamp=datetime.now(),
        )

        self._persist_capital_allocation(allocation)
        return allocation

    def _calculate_diversification_score(self, positions: Dict[str, float]) -> float:
        if not positions:
            return 0.0

        total = sum(positions.values())
        if total <= 0:
            return 0.0

        hhi = sum((size / total) ** 2 for size in positions.values())

        return hhi

    def _persist_position_size(self, position: PositionSize) -> None:
        if not self.db_path:
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO position_sizes
                (ticker, optimal_size, kelly_fraction, max_position_limit,
                 risk_adjusted_size, volatility, portfolio_weight)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    position.ticker,
                    position.optimal_size,
                    position.kelly_fraction,
                    position.max_position_limit,
                    position.risk_adjusted_size,
                    position.volatility,
                    position.portfolio_weight,
                ),
            )

            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Error persisting position size: {e}")

    def _persist_capital_allocation(self, allocation: CapitalAllocation) -> None:
        if not self.db_path:
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            positions_json = json.dumps(allocation.positions)

            cursor.execute(
                """
                INSERT INTO capital_allocations
                (total_capital, allocated_capital, unused_capital, utilization_rate,
                 diversification_score, positions_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    allocation.total_capital,
                    allocation.allocated_capital,
                    allocation.unused_capital,
                    allocation.utilization_rate,
                    allocation.diversification_score,
                    positions_json,
                ),
            )

            conn.commit()
            conn.close()
        except (sqlite3.Error, Exception) as e:
            logger.error(f"Error persisting capital allocation: {e}")

    def get_position_history(self, ticker: str = None) -> List[Dict]:
        if not self.db_path:
            return []

        records = []

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            if ticker:
                cursor.execute(
                    """
                    SELECT ticker, optimal_size, kelly_fraction, risk_adjusted_size,
                           volatility, portfolio_weight, timestamp
                    FROM position_sizes
                    WHERE ticker = ?
                    ORDER BY timestamp DESC
                """,
                    (ticker,),
                )
            else:
                cursor.execute(
                    """
                    SELECT ticker, optimal_size, kelly_fraction, risk_adjusted_size,
                           volatility, portfolio_weight, timestamp
                    FROM position_sizes
                    ORDER BY timestamp DESC
                """
                )

            for row in cursor.fetchall():
                records.append(
                    {
                        "ticker": row[0],
                        "optimal_size": row[1],
                        "kelly_fraction": row[2],
                        "risk_adjusted_size": row[3],
                        "volatility": row[4],
                        "portfolio_weight": row[5],
                        "timestamp": row[6],
                    }
                )

            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Error retrieving position history: {e}")

        return records

    def estimate_max_drawdown(
        self, positions: Dict[str, float], volatility_avg: float
    ) -> float:
        utilization = (
            sum(positions.values()) / self.total_capital
            if self.total_capital > 0
            else 0
        )

        estimated_drawdown = 3.0 * volatility_avg * utilization

        return min(1.0, estimated_drawdown)
