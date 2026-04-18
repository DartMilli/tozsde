"""
Static type definitions for core domain objects.

These TypedDicts formalise the implicit "dict schema" that was previously
scattered across the codebase.  They are structurally compatible with ordinary
Python dicts so all existing code continues to work unchanged; they only add
type-checker visibility and serve as the single source of truth for field names.

Usage:
    from app.domain.types import Decision, PolicyPayload, ModelVote, TradeItem
"""

from __future__ import annotations

from typing import Dict, List, Optional
from typing import TypedDict


# ---------------------------------------------------------------------------
# Recommendation / Decision objects
# ---------------------------------------------------------------------------


class Decision(TypedDict, total=False):
    """Output of build_recommendation() – the core trading decision."""

    action_code: int  # 0=HOLD, 1=BUY, 2=SELL
    action: str  # Human-readable label (e.g. "BUY")
    strength: str  # "STRONG" | "NORMAL" | "WEAK" | "NO_TRADE"
    confidence: float  # Weighted ensemble confidence [0, 1]
    wf_score: Optional[float]  # Walk-forward stability score [0, 1]
    ensemble_quality: (
        str  # "STRONG" | "NORMAL" | "WEAK" | "CHAOTIC"  (EnsembleQualityBucket values)
    )
    quality_score: float  # Composite decision quality score
    no_trade: bool  # True when policy blocked the trade
    no_trade_reason: Optional[str]  # "LOW_CONFIDENCE" | "POLICY_OVERRIDE" | …
    original_action: int  # Action code before policy override
    decision_source: Optional[str]  # "model" | "fallback" | …
    safety_blocked: Optional[bool]
    safety_reason: Optional[str]


class PolicyPayload(TypedDict, total=False):
    """Intermediate payload passed through the policy stack."""

    ticker: str
    avg_confidence: float
    avg_wf_score: Optional[float]
    ensemble_quality: str
    action_code: int
    regime: str


class ModelVote(TypedDict, total=False):
    """Single model vote inside the ensemble."""

    model_path: str
    model_name: Optional[str]
    action: int
    action_label: str
    confidence: float
    wf_score: Optional[float]
    rank: Optional[int]
    trained_at: Optional[object]  # datetime.date


class DailyCandidate(TypedDict, total=False):
    """Intermediate candidate object produced during the daily pipeline."""

    ticker: str
    decision: Decision
    payload: Dict
    allocation_amount: Optional[float]
    decision_source: Optional[str]


# ---------------------------------------------------------------------------
# Trade / Execution objects
# ---------------------------------------------------------------------------


class TradeItem(TypedDict, total=False):
    """A single finalized trade passed to the execution engine."""

    ticker: str
    decision: Decision
    payload: Dict
    allocation_amount: float


class PortfolioState(TypedDict, total=False):
    """Paper-trading portfolio snapshot."""

    cash: float
    positions: Dict[str, object]  # ticker -> PaperPosition


# ---------------------------------------------------------------------------
# Walk-forward / optimisation objects
# ---------------------------------------------------------------------------


class WalkForwardSummary(TypedDict, total=False):
    """Summary returned from run_walk_forward()."""

    ticker: str
    wf_fitness: float
    raw_fitness: float
    normalized_score: float
    best_params: Dict
    folds: List[Dict]
    discarded_ratio: Optional[float]
    mean_oos_sharpe: Optional[float]
