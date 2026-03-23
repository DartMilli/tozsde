"""Settings dataclass for dependency injection.

All runtime configuration fields from the legacy Config live here.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Settings:
    ENABLE_FLASK: bool
    ENABLE_RL: bool
    RL_TRAINING_MODE: bool
    RL_TIME_SPLIT_ENABLED: bool
    RL_TRAIN_END_DATE: Optional[str]
    RL_VAL_END_DATE: Optional[str]
    RL_TEST_END_DATE: Optional[str]
    RL_TRAIN_RATIO: float
    RL_VAL_RATIO: float
    ENABLE_RELIABILITY: bool
    RELIABILITY_SOURCE: str
    ENABLE_CONFIDENCE_CALIBRATION: bool
    EXECUTION_MODE: str
    EXECUTION_POLICY: str
    ENABLE_DRIFT_DETECTION: bool
    ENABLE_DRAWDOWN_GUARD: bool
    ENABLE_EXECUTION_STRESS: bool
    PIPELINE_AUDIT_MODE: bool
    EDGE_DIAGNOSTICS_MODE: bool
    MIN_OOS_SHARPE: float
    MAX_RELATIVE_GAP: float
    AGGREGATION_MODE: str
    ALLOW_NO_MODEL_FALLBACK: bool
    LOGGING_LEVEL: Optional[str]
    PI_MODE: bool

    DATA_DIR: Path
    LOG_DIR: Path
    MODEL_DIR: Path
    DIAGNOSTICS_DIR: Path
    REPORTS_DIR: Path
    CHART_DIR: Path
    TENSORBOARD_DIR: Path
    MODEL_RELIABILITY_DIR: Path
    DECISION_LOG_DIR: Path
    DECISION_OUTCOME_DIR: Path
    HISTORY_DIR: Path
    DB_PATH: Path
    PARAMS_FILE_PATH: Path
    FAILED_DAYS_FILE_PATH: Path
    MODEL_TEST_RESULT_FILE_PATH: Path

    SECRET_KEY: str
    EMAIL_USER: Optional[str]
    EMAIL_PASSWORD: Optional[str]
    ADMIN_API_KEY: str

    CONFIDENCE_NO_TRADE_THRESHOLD: float
    NOTIFY_EMAIL: Optional[str]
    STRONG_CONFIDENCE_THRESHOLD: float
    WEAK_CONFIDENCE_THRESHOLD: float
    STRONG_WF_THRESHOLD: float
    ACTION_LABELS: Dict[str, Dict[int, str]]
    LANG: str
    EMAIL_MAX_BODY_CHARS: int
    EMAIL_MAX_DETAIL_LINES: int

    INITIAL_CAPITAL: float
    RISK: float
    TRANSACTION_FEE_PCT: float
    MIN_SLIPPAGE_PCT: float
    SPREAD_PCT: float
    RELIABILITY_PERIOD_DAYS: int

    P6_DRAWNDOWN_PENALTY_FACTOR: float
    P6_VOLATILITY_PENALTY_FACTOR: float
    P6_SAFETY_OVERRIDE_PENALTY: float
    ENABLE_POSITION_SIZING: bool
    P6_POSITION_MAX_PCT: float
    P6_MIN_CONF_FACTOR: float
    P6_MIN_WF_FACTOR: float
    P6_SAFETY_DISCOUNT: float
    P6_TRUST_MIN_SAMPLES: int
    P6_TRUST_WEIGHT_DEFAULT: float
    P6_REWARD_DRAWDOWN_PENALTY: float
    P6_REWARD_INSTABILITY_PENALTY: float
    P6_REWARD_OVERCONFIDENCE_PENALTY: float
    P6_WF_STABILITY_BASELINE: float
    P6_MAX_DRAWDOWN_BASELINE: float
    P6_MAX_DRAWDOWN_TOLERANCE: float
    P6_EFFECTIVENESS_IMPROVEMENT_MIN: float
    P6_MAX_SAFETY_OVERRIDE_RATE: float

    COOLDOWN_DAYS: int
    COOLDOWN_MAX_TRADES: int
    DRAWDOWN_LOOKBACK: int
    MAX_VIX_THRESHOLD: float

    TRAIN_WINDOW_MONTHS: int
    TEST_WINDOW_MONTHS: int
    WINDOW_STEP_MONTHS: int
    WF_STABILITY_CONSTANT: int

    START_DATE: str
    END_DATE: str

    EXCLUDED_TICKERS: List[str]
    OPTIMIZER_GENERATIONS: int
    OPTIMIZER_POPULATION: int
    RL_TIMESTEPS: int

    BEAR_MARKET_LOOKBACK_DAYS: int
    BEAR_MARKET_SMA_PERIOD: int
    RISK_FREE_FALLBACK: float
    WARMUP_DAYS: int
    DRAWDOWN_PENALTY_THRESHOLD: float
    DRAWDOWN_PENALTY_MULTIPLIER: float

    GA_CXPB: float
    GA_MUTPB: float
    CORRELATION_PENALTY_STRENGTH: float
    VOLATILITY_SOFT_CAP: float
    VOLATILITY_BUCKET_THRESHOLDS: Dict[str, float]
    ENSEMBLE_QUALITY_THRESHOLDS: Dict[str, float]

    TICKERS: Optional[List[str]]
