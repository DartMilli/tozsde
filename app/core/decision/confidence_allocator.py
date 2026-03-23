from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List
import logging
import sqlite3

logger = logging.getLogger(__name__)


class ConfidenceBucket(Enum):
    STRONG = "STRONG"
    NORMAL = "NORMAL"
    WEAK = "WEAK"


@dataclass
class ConfidenceAllocation:
    strategy: str
    confidence_score: float
    bucket: ConfidenceBucket
    multiplier: float
    allocated_capital: float
    base_capital: float
    bucket_allocation_date: datetime


@dataclass
class BucketStatistics:
    bucket: ConfidenceBucket
    count: int
    avg_confidence: float
    total_capital: float
    avg_multiplier: float


class ConfidenceBucketAllocator:
    def __init__(self, db_path: str = None, base_capital: float = 1000.0):
        self.db_path = db_path
        self.base_capital = base_capital
        self._init_db()

    def _init_db(self) -> None:
        if not self.db_path:
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS confidence_allocations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    bucket TEXT NOT NULL,
                    multiplier REAL NOT NULL,
                    allocated_capital REAL NOT NULL,
                    base_capital REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")

    def classify_confidence_bucket(self, confidence_score: float) -> ConfidenceBucket:
        if not 0.0 <= confidence_score <= 1.0:
            raise ValueError(
                f"Confidence score must be between 0.0 and 1.0, got {confidence_score}"
            )

        if confidence_score >= 0.75:
            return ConfidenceBucket.STRONG
        if confidence_score >= 0.5:
            return ConfidenceBucket.NORMAL
        return ConfidenceBucket.WEAK

    def get_multiplier(self, bucket: ConfidenceBucket) -> float:
        multipliers = {
            ConfidenceBucket.STRONG: 1.5,
            ConfidenceBucket.NORMAL: 1.0,
            ConfidenceBucket.WEAK: 0.5,
        }
        return multipliers[bucket]

    def allocate_capital(
        self, strategy: str, confidence_score: float
    ) -> ConfidenceAllocation:
        bucket = self.classify_confidence_bucket(confidence_score)
        multiplier = self.get_multiplier(bucket)
        allocated_capital = self.base_capital * multiplier

        allocation = ConfidenceAllocation(
            strategy=strategy,
            confidence_score=confidence_score,
            bucket=bucket,
            multiplier=multiplier,
            allocated_capital=allocated_capital,
            base_capital=self.base_capital,
            bucket_allocation_date=datetime.now(),
        )

        self._persist_allocation(allocation)
        return allocation

    def allocate_capital_by_bucket(
        self,
        strategies: Dict[str, float],
        rebalance_rate: float = 0.0,
    ) -> Dict[str, ConfidenceAllocation]:
        allocations = {}

        for strategy, confidence in strategies.items():
            if not isinstance(confidence, (int, float)):
                raise ValueError(f"Confidence score for {strategy} must be numeric")

            allocation = self.allocate_capital(strategy, confidence)

            if rebalance_rate > 0.0:
                allocation.allocated_capital *= 1.0 + rebalance_rate

            allocations[strategy] = allocation

        return allocations

    def _persist_allocation(self, allocation: ConfidenceAllocation) -> None:
        if not self.db_path:
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO confidence_allocations
                (strategy, confidence_score, bucket, multiplier, allocated_capital, base_capital)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    allocation.strategy,
                    allocation.confidence_score,
                    allocation.bucket.value,
                    allocation.multiplier,
                    allocation.allocated_capital,
                    allocation.base_capital,
                ),
            )

            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Error persisting allocation: {e}")

    def get_bucket_statistics(self) -> Dict[ConfidenceBucket, BucketStatistics]:
        if not self.db_path:
            return {}

        stats = {}

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for bucket in ConfidenceBucket:
                cursor.execute(
                    """
                    SELECT COUNT(*), AVG(confidence_score), SUM(allocated_capital), AVG(multiplier)
                    FROM confidence_allocations
                    WHERE bucket = ?
                """,
                    (bucket.value,),
                )

                result = cursor.fetchone()
                if result and result[0] > 0:
                    stats[bucket] = BucketStatistics(
                        bucket=bucket,
                        count=result[0],
                        avg_confidence=result[1],
                        total_capital=result[2],
                        avg_multiplier=result[3],
                    )

            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Error retrieving bucket statistics: {e}")

        return stats

    def get_allocation_history(self, strategy: str = None) -> List[Dict]:
        if not self.db_path:
            return []

        records = []

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            if strategy:
                cursor.execute(
                    """
                    SELECT strategy, confidence_score, bucket, multiplier, allocated_capital, timestamp
                    FROM confidence_allocations
                    WHERE strategy = ?
                    ORDER BY timestamp DESC
                """,
                    (strategy,),
                )
            else:
                cursor.execute(
                    """
                    SELECT strategy, confidence_score, bucket, multiplier, allocated_capital, timestamp
                    FROM confidence_allocations
                    ORDER BY timestamp DESC
                """
                )

            for row in cursor.fetchall():
                records.append(
                    {
                        "strategy": row[0],
                        "confidence_score": row[1],
                        "bucket": row[2],
                        "multiplier": row[3],
                        "allocated_capital": row[4],
                        "timestamp": row[5],
                    }
                )

            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Error retrieving allocation history: {e}")

        return records

    def suggest_rebalancing(
        self, current_allocations: Dict[str, float]
    ) -> Dict[str, float]:
        suggestions = {}

        if not current_allocations:
            return suggestions

        avg_capital = sum(current_allocations.values()) / len(current_allocations)

        for strategy, capital in current_allocations.items():
            if abs(capital - avg_capital) > avg_capital * 0.1:
                suggestions[strategy] = avg_capital

        return suggestions
