#!/usr/bin/env bash
set -euo pipefail

TICKER=${TICKER:-VOO}
START_DATE=${START_DATE:-2022-01-01}
END_DATE=${END_DATE:-2023-12-31}

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

"$PYTHON_BIN" "$ROOT_DIR/main.py" validate --ticker "$TICKER" --start-date "$START_DATE" --end-date "$END_DATE"
"$PYTHON_BIN" "$ROOT_DIR/scripts/phase6_check.py" --ticker "$TICKER" --start-date "$START_DATE" --end-date "$END_DATE"
