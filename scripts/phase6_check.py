import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config.config import Config
from app.analysis.phase6_validator import Phase6Validator


def main():
    conn = sqlite3.connect(str(Config.DB_PATH))
    cur = conn.cursor()

    cur.execute(
        "SELECT report_json, computed_at FROM validation_reports ORDER BY computed_at DESC LIMIT 2"
    )
    rows = cur.fetchall()
    if len(rows) == 2:
        r1 = json.loads(rows[0][0]) if rows[0][0] else {}
        r2 = json.loads(rows[1][0]) if rows[1][0] else {}
        print("validation_reports_equal:", r1 == r2)
    else:
        print("validation_reports_equal: insufficient data")

    cur.execute(
        "SELECT metrics_json, computed_at FROM decision_effectiveness_rolling ORDER BY computed_at DESC LIMIT 2"
    )
    rows = cur.fetchall()
    if rows:
        metrics = [json.loads(r[0]) if r[0] else {} for r in rows]
        print("decision_effectiveness_rolling:", metrics)
    else:
        print("decision_effectiveness_rolling: none")

    cur.execute(
        "SELECT model_path, trust_weight, MAX(computed_at) FROM model_trust_metrics GROUP BY model_path ORDER BY trust_weight DESC LIMIT 10"
    )
    rows = cur.fetchall()
    print("model_trust_weights:", rows)

    cur.execute(
        "SELECT model_id, status, updated_at FROM model_registry ORDER BY updated_at DESC LIMIT 5"
    )
    rows = cur.fetchall()
    print("model_registry_latest:", rows)

    validator = Phase6Validator()
    run1 = validator.run("VOO")
    run2 = validator.run("VOO")
    print("phase6_validator_equal:", run1 == run2)
    print("phase6_validator_run:", run1)

    conn.close()


if __name__ == "__main__":
    main()
