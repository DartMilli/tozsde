import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sqlite3
from app.config.config import Config


def main() -> int:
    parser = argparse.ArgumentParser(description="Check outcomes linkage to decisions.")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    args = parser.parse_args()

    query = """
        SELECT dh.id, dh.ticker, dh.timestamp, dh.action_code, o.id AS outcome_id
        FROM decision_history dh
        LEFT JOIN outcomes o ON o.decision_id = dh.id
        WHERE 1=1
    """
    params = []
    if args.ticker:
        query += " AND dh.ticker = ?"
        params.append(args.ticker)
    if args.start_date:
        query += " AND date(dh.timestamp) >= date(?)"
        params.append(args.start_date)
    if args.end_date:
        query += " AND date(dh.timestamp) <= date(?)"
        params.append(args.end_date)

    with sqlite3.connect(str(Config.DB_PATH)) as conn:
        rows = conn.execute(query, params).fetchall()

    total = len(rows)
    trade_rows = [r for r in rows if r[3] != 0]
    missing = [r for r in trade_rows if r[4] is None]

    print("total_decisions:", total)
    print("trade_decisions:", len(trade_rows))
    print("missing_outcomes:", len(missing))

    if missing:
        print("missing_sample:")
        for row in missing[:10]:
            decision_id, ticker, ts, action_code, _ = row
            print(
                f"  id={decision_id} ticker={ticker} ts={ts} action_code={action_code}"
            )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
