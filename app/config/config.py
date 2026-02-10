import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# 1. Abszolút gyökér meghatározása (P0)
# Ez a fájl: app/core/config.py -> szülő: app/core -> szülő: app -> szülő: ROOT
BASE_DIR = Path(__file__).resolve().parents[2]

# 2. Környezeti változók betöltése
load_dotenv(BASE_DIR / ".env")

EXCLUDED_TICKERS = ["OTP.BD", "MOL.BD", "RICHTER.BD"]


class Config:
    ENABLE_FLASK = False
    ENABLE_RL = os.getenv("ENABLE_RL", "false").lower() == "true"
    RL_TRAINING_MODE = os.getenv("RL_TRAINING_MODE", "false").lower() == "true"
    ENABLE_RELIABILITY = False
    ENABLE_CONFIDENCE_CALIBRATION = (
        os.getenv("ENABLE_CONFIDENCE_CALIBRATION", "false").lower() == "true"
    )
    EXECUTION_MODE = os.getenv("EXECUTION_MODE", "paper").lower()
    ENABLE_DRIFT_DETECTION = (
        os.getenv("ENABLE_DRIFT_DETECTION", "true").lower() == "true"
    )
    ENABLE_DRAWDOWN_GUARD = (
        os.getenv("ENABLE_DRAWDOWN_GUARD", "false").lower() == "true"
    )
    ALLOW_NO_MODEL_FALLBACK = (
        os.getenv("ALLOW_NO_MODEL_FALLBACK", "true").lower() == "true"
    )

    LOGGING_LEVEL = os.getenv("LOGGING_LEVEL")

    # Útvonalak
    DATA_DIR = BASE_DIR / "app" / "data"
    LOG_DIR = BASE_DIR / "logs"
    MODEL_DIR = BASE_DIR / "models"

    CHART_DIR = BASE_DIR / "charts"
    TENSORBOARD_DIR = BASE_DIR / "tensorboard"
    MODEL_RELIABILITY_DIR = LOG_DIR / "model_reliability"

    DECISION_LOG_DIR = BASE_DIR / "decision_logs"
    DECISION_OUTCOME_DIR = BASE_DIR / "decision_outcomes"

    HISTORY_DIR = BASE_DIR / "decision_history"

    # Adatbázis (P1 - Egységes DB)
    DB_PATH = DATA_DIR / "market_data.db"

    # Common file paths (Path objects)
    PARAMS_FILE_PATH = DATA_DIR / "optimized_params.json"
    FAILED_DAYS_FILE_PATH = DATA_DIR / "failed_to_download.json"
    MODEL_TEST_RESULT_FILE_PATH = DATA_DIR / "model_test_result.json"

    # Biztonság
    SECRET_KEY = os.getenv("SECRET_KEY", "dev_key_do_not_use_in_prod")
    EMAIL_USER = os.getenv("EMAIL_USER")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")  # App password!
    ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "admin_key_12345")  # Default for testing

    # -------------------- RECOMMENDER --------------------
    CONFIDENCE_NO_TRADE_THRESHOLD = 0.25
    NOTIFY_EMAIL = os.getenv("NOTIFY_EMAIL")
    STRONG_CONFIDENCE_THRESHOLD = 0.75
    WEAK_CONFIDENCE_THRESHOLD = 0.4
    STRONG_WF_THRESHOLD = 0.7

    ACTION_LABELS = {
        "hu": {0: "TARTÁS", 1: "VÉTEL", 2: "ELADÁS"},
        "en": {0: "HOLD", 1: "BUY", 2: "SELL"},
    }

    LANG = "en"

    EMAIL_MAX_BODY_CHARS = int(os.getenv("EMAIL_MAX_BODY_CHARS", "5000"))
    EMAIL_MAX_DETAIL_LINES = int(os.getenv("EMAIL_MAX_DETAIL_LINES", "20"))

    # Kereskedési beállítások (P2 - Tranzakciós költségek)
    INITIAL_CAPITAL = 10000
    RISK = 0.02  # A portfólió 2%-át kockáztatjuk egy trade-en
    TRANSACTION_FEE_PCT = 0.001  # 0.1% jutalék
    MIN_SLIPPAGE_PCT = 0.0005  # 0.05% csúszás
    SPREAD_PCT = 0.0005  # 0.05%

    RELIABILITY_PERIOD_DAYS = 30

    # -------------------- PHASE 6 --------------------
    # P6.1 Decision effectiveness
    P6_DRAWNDOWN_PENALTY_FACTOR = 0.5
    P6_VOLATILITY_PENALTY_FACTOR = 0.5
    P6_SAFETY_OVERRIDE_PENALTY = 0.02

    # P6.2 Position sizing
    ENABLE_POSITION_SIZING = True
    P6_POSITION_MAX_PCT = 0.2
    P6_MIN_CONF_FACTOR = 0.1
    P6_MIN_WF_FACTOR = 0.5
    P6_SAFETY_DISCOUNT = 0.25

    # P6.3 Model trust
    P6_TRUST_MIN_SAMPLES = 5
    P6_TRUST_WEIGHT_DEFAULT = 1.0

    # P6.4 Reward shaping
    P6_REWARD_DRAWDOWN_PENALTY = 0.1
    P6_REWARD_INSTABILITY_PENALTY = 0.1
    P6_REWARD_OVERCONFIDENCE_PENALTY = 0.1

    # P6.5 Promotion gate
    P6_WF_STABILITY_BASELINE = 0.7
    P6_MAX_DRAWDOWN_BASELINE = 0.15
    P6_MAX_DRAWDOWN_TOLERANCE = 1.1
    P6_EFFECTIVENESS_IMPROVEMENT_MIN = 0.0
    P6_MAX_SAFETY_OVERRIDE_RATE = 0.2

    # P4.5 SAFETY
    COOLDOWN_DAYS = 5
    COOLDOWN_MAX_TRADES = 2
    DRAWDOWN_LOOKBACK = 3

    MAX_VIX_THRESHOLD = 30.0  # Efelett pánik van a piacon

    # Walk-Forward paraméterek (P5)
    TRAIN_WINDOW_MONTHS = 24  # 2 év tanulás
    TEST_WINDOW_MONTHS = 6  # 6 hónap tesztelés
    WINDOW_STEP_MONTHS = 3  # 3 havonta lépünk előre

    # tanulási adatok és optimalizációs paraméterek
    START_DATE = "2020-01-01"
    END_DATE = datetime.today().strftime("%Y-%m-%d")

    EXCLUDED_TICKERS = EXCLUDED_TICKERS
    OPTIMIZER_GENERATIONS = 30
    OPTIMIZER_POPULATION = 50
    RL_TIMESTEPS = 100_000

    BEAR_MARKET_LOOKBACK_DAYS = 400

    BEAR_MARKET_SMA_PERIOD = 200

    RISK_FREE_FALLBACK = 0.045

    WARMUP_DAYS = 100

    DRAWDOWN_PENALTY_THRESHOLD = -20

    DRAWDOWN_PENALTY_MULTIPLIER = 0.5

    GA_CXPB = 0.7

    GA_MUTPB = 0.2

    CORRELATION_PENALTY_STRENGTH = 0.5

    VOLATILITY_SOFT_CAP = 0.03

    VOLATILITY_BUCKET_THRESHOLDS = {"LOW": 0.015, "NORMAL": 0.03, "HIGH": 0.06}

    ENSEMBLE_QUALITY_THRESHOLDS = {"STRONG": 0.6, "NORMAL": 0.3, "WEAK": 0.1}

    # Lazy-loaded TICKERS (avoid circular import)
    _TICKERS = None
    _SUPPORTED_TICKERS_LOADED = False

    @classmethod
    def get_supported_tickers(cls):
        """Lazily load supported tickers to avoid circular import with data_loader"""
        if cls._TICKERS is None:
            from app.data_access.data_loader import get_supported_ticker_list

            all_tickers = get_supported_ticker_list()
            cls._TICKERS = [t for t in all_tickers if t not in EXCLUDED_TICKERS]
            cls._SUPPORTED_TICKERS_LOADED = True
        return cls._TICKERS

    @property
    def TICKERS(self):
        """Property to access TICKERS (for backward compatibility)"""
        return self.get_supported_tickers()

    @classmethod
    def ensure_dirs(cls):
        mandatory = [
            cls.DATA_DIR,
            cls.LOG_DIR,
            cls.MODEL_DIR,
            cls.HISTORY_DIR,
        ]

        optional = []
        if cls.ENABLE_FLASK:
            optional.append(cls.CHART_DIR)

        if cls.ENABLE_RL:
            optional.append(cls.TENSORBOARD_DIR)

        if cls.ENABLE_RELIABILITY:
            optional.append(cls.MODEL_RELIABILITY_DIR)
            optional.append(cls.DECISION_LOG_DIR)
            optional.append(cls.DECISION_OUTCOME_DIR)

        for d in mandatory + optional:
            d.mkdir(parents=True, exist_ok=True)


def _enforce_secret_key_policy():
    env = os.getenv("FLASK_ENV") or os.getenv("ENV") or "development"
    env = env.lower()
    if env in {"production", "prod"}:
        if not Config.SECRET_KEY or Config.SECRET_KEY == "dev_key_do_not_use_in_prod":
            raise RuntimeError("SECRET_KEY must be set in production environment")


# Inicializáláskor futtatjuk
try:
    from app.config.pi_config import apply_pi_config

    apply_pi_config(config_cls=Config, ensure_dirs=False)
except Exception:
    pass

_enforce_secret_key_policy()

Config.ensure_dirs()
