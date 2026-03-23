import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from .settings import Settings
from .pi_config import detect_pi_mode, apply_pi_config


def build_settings(
    env_file: Optional[Path] = None, ensure_dirs: bool = True
) -> Settings:
    base_dir = Path(__file__).resolve().parents[2]
    # Load .env from repository root by default
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv(base_dir / ".env")

    data_dir = base_dir / "app" / "data"
    log_dir = base_dir / "logs"
    model_dir = base_dir / "models"
    diagnostics_dir = base_dir / "diagnostics"
    reports_dir = base_dir / "reports"
    chart_dir = base_dir / "charts"
    tensorboard_dir = base_dir / "tensorboard"
    model_reliability_dir = log_dir / "model_reliability"
    decision_log_dir = base_dir / "decision_logs"
    decision_outcome_dir = base_dir / "decision_outcomes"
    history_dir = base_dir / "decision_history"
    db_path = data_dir / "market_data.db"

    pi_mode = detect_pi_mode()

    settings = Settings(
        ENABLE_FLASK=os.getenv("ENABLE_FLASK", "false").lower() == "true",
        ENABLE_RL=os.getenv("ENABLE_RL", "false").lower() == "true",
        RL_TRAINING_MODE=os.getenv("RL_TRAINING_MODE", "false").lower() == "true",
        RL_TIME_SPLIT_ENABLED=os.getenv("RL_TIME_SPLIT_ENABLED", "true").lower()
        == "true",
        RL_TRAIN_END_DATE=os.getenv("RL_TRAIN_END_DATE"),
        RL_VAL_END_DATE=os.getenv("RL_VAL_END_DATE"),
        RL_TEST_END_DATE=os.getenv("RL_TEST_END_DATE"),
        RL_TRAIN_RATIO=float(os.getenv("RL_TRAIN_RATIO", "0.7")),
        RL_VAL_RATIO=float(os.getenv("RL_VAL_RATIO", "0.15")),
        ENABLE_RELIABILITY=False,
        RELIABILITY_SOURCE=os.getenv("RELIABILITY_SOURCE", "db").lower(),
        ENABLE_CONFIDENCE_CALIBRATION=os.getenv(
            "ENABLE_CONFIDENCE_CALIBRATION", "false"
        ).lower()
        == "true",
        EXECUTION_MODE=os.getenv("EXECUTION_MODE", "paper").lower(),
        EXECUTION_POLICY=os.getenv("EXECUTION_POLICY", "next_open").lower(),
        ENABLE_DRIFT_DETECTION=os.getenv("ENABLE_DRIFT_DETECTION", "true").lower()
        == "true",
        ENABLE_DRAWDOWN_GUARD=os.getenv("ENABLE_DRAWDOWN_GUARD", "false").lower()
        == "true",
        ENABLE_EXECUTION_STRESS=os.getenv("ENABLE_EXECUTION_STRESS", "true").lower()
        == "true",
        PIPELINE_AUDIT_MODE=os.getenv("PIPELINE_AUDIT_MODE", "false").lower() == "true",
        EDGE_DIAGNOSTICS_MODE=os.getenv("EDGE_DIAGNOSTICS_MODE", "false").lower()
        == "true",
        MIN_OOS_SHARPE=float(os.getenv("MIN_OOS_SHARPE", "0.4")),
        MAX_RELATIVE_GAP=float(os.getenv("MAX_RELATIVE_GAP", "0.4")),
        AGGREGATION_MODE=os.getenv("AGGREGATION_MODE", "latest_only").lower(),
        ALLOW_NO_MODEL_FALLBACK=os.getenv("ALLOW_NO_MODEL_FALLBACK", "true").lower()
        == "true",
        LOGGING_LEVEL=os.getenv("LOGGING_LEVEL"),
        PI_MODE=pi_mode,
        DATA_DIR=data_dir,
        LOG_DIR=log_dir,
        MODEL_DIR=model_dir,
        DIAGNOSTICS_DIR=diagnostics_dir,
        REPORTS_DIR=reports_dir,
        CHART_DIR=chart_dir,
        TENSORBOARD_DIR=tensorboard_dir,
        MODEL_RELIABILITY_DIR=model_reliability_dir,
        DECISION_LOG_DIR=decision_log_dir,
        DECISION_OUTCOME_DIR=decision_outcome_dir,
        HISTORY_DIR=history_dir,
        DB_PATH=db_path,
        PARAMS_FILE_PATH=data_dir / "optimized_params.json",
        FAILED_DAYS_FILE_PATH=data_dir / "failed_to_download.json",
        MODEL_TEST_RESULT_FILE_PATH=data_dir / "model_test_result.json",
        SECRET_KEY=os.getenv("SECRET_KEY", "dev_key_do_not_use_in_prod"),
        EMAIL_USER=os.getenv("EMAIL_USER"),
        EMAIL_PASSWORD=os.getenv("EMAIL_PASSWORD"),
        ADMIN_API_KEY=os.getenv("ADMIN_API_KEY", "admin_key_12345"),
        CONFIDENCE_NO_TRADE_THRESHOLD=0.25,
        NOTIFY_EMAIL=os.getenv("NOTIFY_EMAIL"),
        STRONG_CONFIDENCE_THRESHOLD=0.75,
        WEAK_CONFIDENCE_THRESHOLD=0.4,
        STRONG_WF_THRESHOLD=0.7,
        ACTION_LABELS={
            "hu": {0: "TARTAS", 1: "VETEL", 2: "ELADAS"},
            "en": {0: "HOLD", 1: "BUY", 2: "SELL"},
        },
        LANG="en",
        EMAIL_MAX_BODY_CHARS=int(os.getenv("EMAIL_MAX_BODY_CHARS", "5000")),
        EMAIL_MAX_DETAIL_LINES=int(os.getenv("EMAIL_MAX_DETAIL_LINES", "20")),
        INITIAL_CAPITAL=10000,
        RISK=0.02,
        TRANSACTION_FEE_PCT=0.001,
        MIN_SLIPPAGE_PCT=0.0005,
        SPREAD_PCT=0.0005,
        RELIABILITY_PERIOD_DAYS=int(os.getenv("RELIABILITY_PERIOD_DAYS", "30")),
        P6_DRAWNDOWN_PENALTY_FACTOR=0.5,
        P6_VOLATILITY_PENALTY_FACTOR=0.5,
        P6_SAFETY_OVERRIDE_PENALTY=0.02,
        ENABLE_POSITION_SIZING=True,
        P6_POSITION_MAX_PCT=0.2,
        P6_MIN_CONF_FACTOR=0.1,
        P6_MIN_WF_FACTOR=0.5,
        P6_SAFETY_DISCOUNT=0.25,
        P6_TRUST_MIN_SAMPLES=5,
        P6_TRUST_WEIGHT_DEFAULT=1.0,
        P6_REWARD_DRAWDOWN_PENALTY=0.1,
        P6_REWARD_INSTABILITY_PENALTY=0.1,
        P6_REWARD_OVERCONFIDENCE_PENALTY=0.1,
        P6_WF_STABILITY_BASELINE=0.7,
        P6_MAX_DRAWDOWN_BASELINE=0.15,
        P6_MAX_DRAWDOWN_TOLERANCE=1.1,
        P6_EFFECTIVENESS_IMPROVEMENT_MIN=0.0,
        P6_MAX_SAFETY_OVERRIDE_RATE=0.2,
        COOLDOWN_DAYS=5,
        COOLDOWN_MAX_TRADES=2,
        DRAWDOWN_LOOKBACK=3,
        MAX_VIX_THRESHOLD=30.0,
        TRAIN_WINDOW_MONTHS=12,
        TEST_WINDOW_MONTHS=3,
        WINDOW_STEP_MONTHS=2,
        WF_STABILITY_CONSTANT=10,
        START_DATE=os.getenv("START_DATE", "2020-01-01"),
        END_DATE=os.getenv("END_DATE", datetime.today().strftime("%Y-%m-%d")),
        EXCLUDED_TICKERS=["OTP.BD", "MOL.BD", "RICHTER.BD"],
        OPTIMIZER_GENERATIONS=30,
        OPTIMIZER_POPULATION=50,
        RL_TIMESTEPS=int(os.getenv("RL_TIMESTEPS", "100000")),
        BEAR_MARKET_LOOKBACK_DAYS=400,
        BEAR_MARKET_SMA_PERIOD=200,
        RISK_FREE_FALLBACK=0.045,
        WARMUP_DAYS=100,
        DRAWDOWN_PENALTY_THRESHOLD=-20,
        DRAWDOWN_PENALTY_MULTIPLIER=0.5,
        GA_CXPB=0.7,
        GA_MUTPB=0.2,
        CORRELATION_PENALTY_STRENGTH=0.5,
        VOLATILITY_SOFT_CAP=0.03,
        VOLATILITY_BUCKET_THRESHOLDS={"LOW": 0.015, "NORMAL": 0.03, "HIGH": 0.06},
        ENSEMBLE_QUALITY_THRESHOLDS={"STRONG": 0.6, "NORMAL": 0.3, "WEAK": 0.1},
        TICKERS=None,
    )

    if settings.PI_MODE:
        settings = apply_pi_config(settings)

    if ensure_dirs:
        for d in [
            settings.DATA_DIR,
            settings.LOG_DIR,
            settings.MODEL_DIR,
            settings.DIAGNOSTICS_DIR,
            settings.REPORTS_DIR,
            settings.HISTORY_DIR,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    return settings
