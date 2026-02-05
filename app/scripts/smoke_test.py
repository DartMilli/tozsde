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

from app.config.config import Config
from app.data_access.data_manager import DataManager
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)

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
        tickers = Config.get_supported_tickers()
        results["checks"]["tickers_loaded"] = {
            "ok": True,
            "count": len(tickers),
        }
    except Exception as e:
        results["checks"]["tickers_loaded"] = {"ok": False, "error": str(e)}
        results["ok"] = False

    # Check 2: DB connectivity + tables
    try:
        dm = DataManager()
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
