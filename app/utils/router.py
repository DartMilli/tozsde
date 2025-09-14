from pathlib import Path

# Project root: utils -> app -> <root>
BASE_DIR = Path(__file__).resolve().parents[2]

# Data & artifact directories
DATA_DIR = BASE_DIR / "app" / "data"
MODEL_DIR = BASE_DIR / "models"
CHARTS_DIR = BASE_DIR / "charts"
TENSORBOARD_DIR = BASE_DIR / "tensorboard"

# Ensure directories exist
for p in [DATA_DIR, MODEL_DIR, CHARTS_DIR, TENSORBOARD_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Common file paths (Path objects)
PRICE_DB_PATH = DATA_DIR / "price_cache.db"
PORTFOLIO_DB_PATH = DATA_DIR / "portfolio.db"
PARAMS_FILE_PATH = DATA_DIR / "optimized_params.json"
RECOMMENDATION_DB_PATH = DATA_DIR / "recommendations.db"
FAILED_DAYS_FILE_PATH = DATA_DIR / "failed_to_download.json"
MODEL_TEST_RESULT_FILE_PATH = DATA_DIR / "model_test_result.json"
