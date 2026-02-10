import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config.config import Config
from app.data_access.data_loader import load_data
from app.data_access.data_manager import DataManager


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cache OHLCV data without creating decisions."
    )
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    args = parser.parse_args()

    end = args.end_date or datetime.today().strftime("%Y-%m-%d")
    if args.start_date:
        start = args.start_date
    else:
        start = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")

    df = load_data(args.ticker, start=start, end=end)
    if df is None or df.empty:
        print("cache_status: no_data")
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
