# Tozsde Trading System

EN: This is the root landing page. The full bilingual documentation lives in docs/README.md.
HU: Ez a gyoker README. A teljes (EN + HU) dokumentacio a docs/README.md fajlban van.

## Quick Start (Windows IDE)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Note: initialize.sh is not used on Windows. Use the commands above in your IDE terminal.

## Common Commands
```bash
python main.py daily
python main.py run-paper-history --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
python main.py validate --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
```

## Docs
- docs/README.md
- docs/INDEX.md
- docs/testing/TEST_STATUS_REPORT.md
