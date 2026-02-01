"""
Confidence Bucket Allocator - Capital allocation based on strategy confidence levels.

This module implements confidence-based capital allocation where strategies are
classified into confidence buckets (STRONG, NORMAL, WEAK) and allocated capital
multipliers based on their decision confidence scores.

Classes:
    ConfidenceBucket: Enum for confidence levels
    ConfidenceAllocation: Dataclass for allocation result
    ConfidenceBucketAllocator: Main allocator class
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
import sqlite3
import logging

logger = logging.getLogger(__name__)


class ConfidenceBucket(Enum):
    """Confidence bucket levels with corresponding capital multipliers."""
    STRONG = "STRONG"      # confidence >= 0.75, multiplier 1.5x
    NORMAL = "NORMAL"      # 0.5 <= confidence < 0.75, multiplier 1.0x
    WEAK = "WEAK"          # confidence < 0.5, multiplier 0.5x


@dataclass
class ConfidenceAllocation:
    """Result of confidence-based capital allocation."""
    strategy: str
    confidence_score: float
    bucket: ConfidenceBucket
    multiplier: float
    allocated_capital: float
    base_capital: float
    bucket_allocation_date: datetime


@dataclass
class BucketStatistics:
    """Statistics for a confidence bucket."""
    bucket: ConfidenceBucket
    count: int
    avg_confidence: float
    total_capital: float
    avg_multiplier: float


class ConfidenceBucketAllocator:
    """
    Allocates capital to strategies based on confidence levels.
    
    Strategies are classified into three confidence buckets:
    - STRONG (≥0.75): 1.5x capital multiplier
    - NORMAL (0.5-0.75): 1.0x capital multiplier
    - WEAK (<0.5): 0.5x capital multiplier
    
    This ensures high-confidence strategies receive more capital while
    low-confidence strategies are dampened to reduce risk.
    """
    
    def __init__(self, db_path: str = None, base_capital: float = 1000.0):
        """
        Initialize the confidence bucket allocator.
        
        Args:
            db_path: Path to SQLite database for persistence
            base_capital: Base capital per strategy unit
        """
        self.db_path = db_path
        self.base_capital = base_capital
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize SQLite database for allocation tracking."""
        if not self.db_path:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
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
            """)
            
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
    
    def classify_confidence_bucket(self, confidence_score: float) -> ConfidenceBucket:
        """
        Classify a confidence score into a confidence bucket.
        
        Args:
            confidence_score: Confidence score between 0.0 and 1.0
        
        Returns:
            ConfidenceBucket: The corresponding confidence bucket
        
        Raises:
            ValueError: If confidence_score is not between 0.0 and 1.0
        """
        if not 0.0 <= confidence_score <= 1.0:
            raise ValueError(f"Confidence score must be between 0.0 and 1.0, got {confidence_score}")
        
        if confidence_score >= 0.75:
            return ConfidenceBucket.STRONG
        elif confidence_score >= 0.5:
            return ConfidenceBucket.NORMAL
        else:
            return ConfidenceBucket.WEAK
    
    def get_multiplier(self, bucket: ConfidenceBucket) -> float:
        """
        Get the capital multiplier for a confidence bucket.
        
        Args:
            bucket: The confidence bucket
        
        Returns:
            float: The capital multiplier (1.5 for STRONG, 1.0 for NORMAL, 0.5 for WEAK)
        """
        multipliers = {
            ConfidenceBucket.STRONG: 1.5,
            ConfidenceBucket.NORMAL: 1.0,
            ConfidenceBucket.WEAK: 0.5,
        }
        return multipliers[bucket]
    
    def allocate_capital(self, strategy: str, confidence_score: float) -> ConfidenceAllocation:
        """
        Allocate capital to a strategy based on its confidence score.
        
        Args:
            strategy: Strategy name
            confidence_score: Confidence score (0.0-1.0)
        
        Returns:
            ConfidenceAllocation: The allocation result
        """
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
            bucket_allocation_date=datetime.now()
        )
        
        self._persist_allocation(allocation)
        return allocation
    
    def allocate_capital_by_bucket(
        self,
        strategies: Dict[str, float],
        rebalance_rate: float = 0.0
    ) -> Dict[str, ConfidenceAllocation]:
        """
        Allocate capital to multiple strategies based on their confidence scores.
        
        Args:
            strategies: Dict of {strategy_name: confidence_score}
            rebalance_rate: Optional rebalancing rate for dynamic adjustments
        
        Returns:
            Dict of {strategy_name: ConfidenceAllocation}
        
        Raises:
            ValueError: If confidence scores are invalid
        """
        allocations = {}
        
        for strategy, confidence in strategies.items():
            if not isinstance(confidence, (int, float)):
                raise ValueError(f"Confidence score for {strategy} must be numeric")
            
            allocation = self.allocate_capital(strategy, confidence)
            
            # Apply rebalancing if specified
            if rebalance_rate > 0.0:
                allocation.allocated_capital *= (1.0 + rebalance_rate)
            
            allocations[strategy] = allocation
        
        return allocations
    
    def _persist_allocation(self, allocation: ConfidenceAllocation) -> None:
        """Persist allocation to database."""
        if not self.db_path:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO confidence_allocations
                (strategy, confidence_score, bucket, multiplier, allocated_capital, base_capital)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                allocation.strategy,
                allocation.confidence_score,
                allocation.bucket.value,
                allocation.multiplier,
                allocation.allocated_capital,
                allocation.base_capital
            ))
            
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Error persisting allocation: {e}")
    
    def get_bucket_statistics(self) -> Dict[ConfidenceBucket, BucketStatistics]:
        """
        Get statistics for each confidence bucket.
        
        Returns:
            Dict mapping ConfidenceBucket to BucketStatistics
        """
        if not self.db_path:
            return {}
        
        stats = {}
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for bucket in ConfidenceBucket:
                cursor.execute("""
                    SELECT COUNT(*), AVG(confidence_score), SUM(allocated_capital), AVG(multiplier)
                    FROM confidence_allocations
                    WHERE bucket = ?
                """, (bucket.value,))
                
                result = cursor.fetchone()
                if result and result[0] > 0:
                    stats[bucket] = BucketStatistics(
                        bucket=bucket,
                        count=result[0],
                        avg_confidence=result[1],
                        total_capital=result[2],
                        avg_multiplier=result[3]
                    )
            
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Error retrieving bucket statistics: {e}")
        
        return stats
    
    def get_allocation_history(self, strategy: str = None) -> List[Dict]:
        """
        Get allocation history from database.
        
        Args:
            strategy: Optional strategy name to filter by
        
        Returns:
            List of allocation records
        """
        if not self.db_path:
            return []
        
        records = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if strategy:
                cursor.execute("""
                    SELECT strategy, confidence_score, bucket, multiplier, allocated_capital, timestamp
                    FROM confidence_allocations
                    WHERE strategy = ?
                    ORDER BY timestamp DESC
                """, (strategy,))
            else:
                cursor.execute("""
                    SELECT strategy, confidence_score, bucket, multiplier, allocated_capital, timestamp
                    FROM confidence_allocations
                    ORDER BY timestamp DESC
                """)
            
            for row in cursor.fetchall():
                records.append({
                    'strategy': row[0],
                    'confidence_score': row[1],
                    'bucket': row[2],
                    'multiplier': row[3],
                    'allocated_capital': row[4],
                    'timestamp': row[5]
                })
            
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Error retrieving allocation history: {e}")
        
        return records
    
    def suggest_rebalancing(self, current_allocations: Dict[str, float]) -> Dict[str, float]:
        """
        Suggest rebalancing based on current allocations vs confidence scores.
        
        Args:
            current_allocations: Dict of {strategy: current_capital}
        
        Returns:
            Dict of {strategy: suggested_capital}
        """
        suggestions = {}
        
        # Get average current capital
        if not current_allocations:
            return suggestions
        
        avg_capital = sum(current_allocations.values()) / len(current_allocations)
        
        for strategy, capital in current_allocations.items():
            # If capital deviates significantly, suggest rebalancing
            if abs(capital - avg_capital) > avg_capital * 0.1:  # 10% threshold
                suggestions[strategy] = avg_capital
        
        return suggestions
