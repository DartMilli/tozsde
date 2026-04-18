import sqlite3, pandas as pd, sys

db_path = r"c:\tozsde\app\data\market_data.db"
conn = sqlite3.connect(str(db_path))

# Get all rows with start date
df_full = pd.read_sql(
    "SELECT * FROM ohlcv WHERE ticker=? AND date >= ? ORDER BY date ASC",
    conn,
    params=("VOO", "2025-03-01"),
    parse_dates=["date"],
)
df_full.set_index("date", inplace=True)
print(f"Full read (start=2025-03-01): {len(df_full)} rows")
nat_mask = df_full.index.isna()
print(f"NaT in index: {nat_mask.sum()}")

# Get raw values for NaT rows
c = conn.execute(
    "SELECT rowid, date FROM ohlcv WHERE ticker=? AND date >= ? ORDER BY date ASC",
    ("VOO", "2025-03-01"),
)
all_raw = c.fetchall()
print(f"Raw rows from DB: {len(all_raw)}")

# Find unique date formats
formats_seen = set()
for rowid, d in all_raw:
    formats_seen.add(len(d))

print("Length variants:", formats_seen)

# show first few of each length
by_len = {}
for rowid, d in all_raw:
    L = len(d)
    if L not in by_len:
        by_len[L] = []
    if len(by_len[L]) < 3:
        by_len[L].append((rowid, d))

for L, examples in sorted(by_len.items()):
    print(f"  len={L}: {examples}")

conn.close()
