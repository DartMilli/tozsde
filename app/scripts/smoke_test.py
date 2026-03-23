"""
Smoke Test Script (Sprint 12)

Validates:
- Config/SECRET_KEY policy
- Database connectivity
- Core tables present
- Basic imports

Usage:
  python -m app.scripts.smoke_test
"""

import json
import sys
from datetime import datetime

from app.config.build_settings import build_settings
from app.infrastructure.repositories import DataManagerRepository as DataManager
from app.data_access.data_loader import get_supported_ticker_list
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)

settings = build_settings()
dm = DataManager(settings=settings)

REQUIRED_TABLES = {
    "ohlcv",
    "recommendations",
    "trades",
    "decision_history",
    "pipeline_metrics",
}


def run_smoke_test() -> dict:
    results = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "ok": True,
        "checks": {},
    }

    # Check 1: supported tickers load
    try:
        tickers = (
            list(settings.TICKERS)
            if settings.TICKERS
            else list(get_supported_ticker_list())
        )
        excluded = set(getattr(settings, "EXCLUDED_TICKERS", []))
        tickers = [t for t in tickers if t not in excluded]
        results["checks"]["tickers_loaded"] = {
            "ok": True,
            "count": len(tickers),
        }
    except Exception as e:
        results["checks"]["tickers_loaded"] = {"ok": False, "error": str(e)}
        results["ok"] = False

    # Check 2: DB connectivity + tables
    try:
        # dm should be injected from DI root
        with dm.connection() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            tables = {r[0] for r in rows}
        missing = sorted(list(REQUIRED_TABLES - tables))
        results["checks"]["db_tables"] = {
            "ok": len(missing) == 0,
            "missing": missing,
        }
        if missing:
            results["ok"] = False
    except Exception as e:
        results["checks"]["db_tables"] = {"ok": False, "error": str(e)}
        results["ok"] = False

    return results


def main() -> int:
    results = run_smoke_test()
    print(json.dumps(results, indent=2, default=str))
    if results.get("ok"):
        logger.info("Smoke test PASSED")
        return 0
    logger.error("Smoke test FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
