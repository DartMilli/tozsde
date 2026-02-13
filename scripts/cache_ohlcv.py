import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config.config import Config
from app.data_access.data_loader import ensure_data_cached
from app.data_access.data_manager import DataManager


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cache OHLCV data without creating decisions."
    )
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    args = parser.parse_args()

    end = args.end_date or Config.END_DATE
    start = args.start_date or Config.START_DATE

    if not ensure_data_cached(args.ticker, start=start, end=end):
        print("cache_status: incomplete")
        return 1

    dm = DataManager()
    cached = dm.load_ohlcv(args.ticker)
    last_date = cached.index[-1].date().isoformat() if not cached.empty else "unknown"
    print("cache_status: ok")
    print(f"rows: {len(cached)}")
    print(f"last_date: {last_date}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
