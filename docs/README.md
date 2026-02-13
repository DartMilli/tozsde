# Tozsde Trading System

## English

### Overview
Tozsde is a Python trading system that runs a daily decision pipeline, records auditable decisions and outcomes in SQLite, and provides backtesting, historical paper runs, and validation tooling (Phase 5 and Phase 6). It supports paper execution, model ensembles, position sizing, and reliability analysis, with reporting and operational tooling for monitoring and maintenance.

### Feature Map (Detailed)
- Daily pipeline: data load, signals, decision policy, allocation, persistence, notifications.
- Paper execution: portfolio state tracking and outcome evaluation.
- Historical paper runner: deterministic backfills for a date range; fallback HOLD decisions if no RL models are present.
- Validation: decision quality, confidence calibration, walk-forward stability, safety stress, and Phase 6 checks.
- Backtesting and audit: replay, audit trails, reward shaping analysis, and reporting outputs.
- Ops tooling: health checks, backups, error reporting, cron scheduling, and log management.

### Quick Start (Windows IDE)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Note: initialize.sh is not used on Windows. Use the commands above in your IDE terminal.

### CLI Usage (Project Root)
```bash
python main.py daily
python main.py daily --ticker VOO
python main.py weekly
python main.py monthly
python main.py walk-forward VOO
python main.py train-rl VOO
python main.py run-paper-history --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
python main.py validate --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
```

### Data and Storage
- SQLite persists decisions, outcomes, portfolio state, and validation metrics.
- Market data is loaded via the data loader and stored as OHLCV.
- Decision history captures model votes, audits, and position sizing for traceability.

### Validation and Reporting
- Phase 5 aggregates decision quality, calibration, walk-forward stability, and safety stress.
- Phase 6 checks effectiveness, position sizing monotonicity, model trust, reward shaping, and promotion gates.
- Append validation results to the test report:

```bash
python scripts/run_tests_with_report.py --skip-tests --with-validation --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
```

### Admin API (Selected Endpoints)
Admin endpoints require the X-Admin-Key header (see Config.ADMIN_API_KEY).

- GET /admin/health
- GET /admin/performance/summary?days=30
- GET /admin/performance/drawdown?days=90
- GET /admin/performance/rolling?days=90&window=30
- GET /admin/errors/summary
- GET /admin/capital/status

### Tests
- Latest test status and coverage: docs/testing/TEST_STATUS_REPORT.md
- Run locally:

```bash
pytest
```

- One-command full suite:

```bash
python scripts/run_all_tests.py
```

### Project Layout (Current)
- app/: application modules (decision, backtesting, analysis, data access, services)
- scripts/: developer utilities and reporting helpers
- tests/: pytest suite
- docs/: documentation and test reports

### Documentation
- docs/INDEX.md for navigation
- docs/SPRINTS.md for sprint history and architecture narrative
- docs/TROUBLESHOOTING_GUIDE.md for operational issues
- docs/deployment for Raspberry Pi setup

### CI workflows (GitHub Actions)
- .github/workflows/phase6_check.yml: runs Phase 5 + Phase 6 validation in CI.
	- Usage: GitHub -> Actions -> "phase6-check" -> Run workflow.
- .github/workflows/train_models.yml: model training (minimal/full) when required.
	- Usage: GitHub -> Actions -> "train-models" -> Run workflow (mode minimal or full).

Note: deploy_rpi.sh does not run RL training; Pi cron covers daily/weekly/GA monthly tasks only.

### Notes
- Historical paper runner uses a fallback HOLD decision if no RL model files are present.
- Validation depends on outcomes being recorded; without outcomes, effectiveness and trust metrics report no_data.

## Magyar

### Attekintes
A Tozsde egy Python alapú kereskedési rendszer, amely napi döntési pipeline-t futtat, auditálható döntéseket és outcome-okat ment SQLite-ba, valamint backtestinget, historikus paper futtatást és validációs toolingot ad (Phase 5 és Phase 6). Támogatja a paper végrehajtást, model ensemble-t, pozícióméretezést és megbízhatóság-elemzést, monitorozási és karbantartási eszközökkel.

### Funkciotérkép (részletes)
- Napi pipeline: adatbetöltés, jelgenerálás, policy, allokáció, mentés, értesítés.
- Paper execution: portfolio state és outcome számítás.
- Historikus paper runner: determinisztikus visszatöltés; fallback HOLD döntés, ha nincs RL modell.
- Validáció: döntési minőség, kalibráció, walk-forward stabilitás, safety stress, Phase 6 ellenőrzések.
- Backtesting és audit: replay, audit nyomvonalak, reward shaping elemzés, riportok.
- Ops tooling: health check, backup, error reporting, cron ütemezés, log menedzsment.

### Gyors Start (Windows IDE)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Megjegyzes: initialize.sh nem szukseges Windows alatt. Hasznald a fenti parancsokat az IDE terminalban.

### CLI Hasznalat (projekttoba)
```bash
python main.py daily
python main.py daily --ticker VOO
python main.py weekly
python main.py monthly
python main.py walk-forward VOO
python main.py train-rl VOO
python main.py run-paper-history --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
python main.py validate --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
```

### Adat es tarolas
- SQLite tarolja a döntéseket, outcome-okat, portfolio state-et és validációs metrikákat.
- Piaci adatok a data loaderen keresztül jönnek, OHLCV-kent mentve.
- Decision history tartalmazza a model vote-okat, auditot és pozícióméretezést.

### Validacio es riport
- Phase 5: döntési minőség, kalibráció, WF stabilitás, safety stress.
- Phase 6: hatékonyság, pozícióméretezés monotonitás, model trust, reward shaping, promotion gate.
- Validáció beírása a teszt riportba:

```bash
python scripts/run_tests_with_report.py --skip-tests --with-validation --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
```

### Admin API (kiemelt endpointok)
Az admin endpointokhoz X-Admin-Key header szükséges (Config.ADMIN_API_KEY).

- GET /admin/health
- GET /admin/performance/summary?days=30
- GET /admin/performance/drawdown?days=90
- GET /admin/performance/rolling?days=90&window=30
- GET /admin/errors/summary
- GET /admin/capital/status

### Tesztek
- Legfrissebb teszt statusz: docs/testing/TEST_STATUS_REPORT.md
- Lokalis futtatas:

```bash
pytest
```

### Projekt szerkezet (jelenlegi)
- app/: alkalmazas modulok (decision, backtesting, analysis, data access, services)
- scripts/: fejlesztoi segedeszkozok
- tests/: pytest suite
- docs/: dokumentacio es teszt riportok

### Dokumentacio
- docs/INDEX.md navigacio
- docs/SPRINTS.md sprint tortenet es architektura
- docs/TROUBLESHOOTING_GUIDE.md hibakereses
- docs/deployment Raspberry Pi telepites

### Megjegyzesek
- A historikus paper runner fallback HOLD döntest ad, ha nincs RL modell.
- Validáció outcome-ok nélkül no_data-t ad az effectiveness/trust metrikákra.
