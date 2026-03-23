Recommended: Run governance/validation via main.py:

	python main.py governance --mode diagnostics
	python main.py governance --mode validation
	python main.py governance --mode full

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

### Governance Runner
Unified quant validation and governance entry point:

```bash
python app/governance/quant_runner.py --mode diagnostics
python app/governance/quant_runner.py --mode validation
python app/governance/quant_runner.py --mode full
```

Reports are written to `reports/<timestamp>/` and include summary, validation, diagnostics, tests, checklist, and a run log.

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
- docs/validation_framework.md for quant validation and governance

### CI workflows (GitHub Actions)
- .github/workflows/phase6_check.yml: runs Phase 5 + Phase 6 validation in CI.
	- Usage: GitHub -> Actions -> "phase6-check" -> Run workflow.
- .github/workflows/train_models.yml: model training (minimal/full) when required.
	- Usage: GitHub -> Actions -> "train-models" -> Run workflow (mode minimal or full).

Note: deploy_rpi.sh supports optional RL training and optional RL cron via environment variables.

### Notes
- Historical paper runner uses a fallback HOLD decision if no RL model files are present.
- Validation depends on outcomes being recorded; without outcomes, effectiveness and trust metrics report no_data.

## Magyar

### Attekintes
A Tozsde egy Python alapu kereskedesi rendszer, amely napi dontesi pipeline-t futtat, auditalhato donteseket es outcome-okat ment SQLite-ba, valamint backtestinget, historikus paper futtatast es validacios toolingot ad (Phase 5 es Phase 6). Tamogatja a paper vegrehajtast, model ensemble-t, poziciomeretezest es megbizhatosag-elemzest, monitorozasi es karbantartasi eszkozokkel.

### Funkcioterkep (reszletes)
- Napi pipeline: adatbetoltes, jelgeneralas, policy, allokacio, mentes, ertesites.
- Paper execution: portfolio state es outcome szamitas.
- Historikus paper runner: determinisztikus visszatoltes; fallback HOLD dontes, ha nincs RL modell.
- Validacio: dontesi minoseg, kalibracio, walk-forward stabilitas, safety stress, Phase 6 ellenorzesek.
- Backtesting es audit: replay, audit nyomvonalak, reward shaping elemzes, riportok.
- Ops tooling: health check, backup, error reporting, cron utemezes, log menedzsment.

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
- SQLite tarolja a donteseket, outcome-okat, portfolio state-et es validacios metrikakat.
- Piaci adatok a data loaderen keresztul jonnek, OHLCV-kent mentve.
- Decision history tartalmazza a model vote-okat, auditot es poziciomeretezest.

### Validacio es riport
- Phase 5: dontesi minoseg, kalibracio, WF stabilitas, safety stress.
- Phase 6: hatekonysag, poziciomeretezes monotonitas, model trust, reward shaping, promotion gate.
- Validacio beirasa a teszt riportba:

```bash
python scripts/run_tests_with_report.py --skip-tests --with-validation --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
```

### Admin API (kiemelt endpointok)
Az admin endpointokhoz X-Admin-Key header szukseges (Config.ADMIN_API_KEY).

- GET /admin/health
- GET /admin/performance/summary?days=30
- GET /admin/performance/drawdown?days=90
- GET /admin/performance/rolling?days=90&window=30
- GET /admin/errors/summary
- GET /admin/capital/status

### Tesztek
- Legfrissebb teszt statusz es coverage: docs/testing/TEST_STATUS_REPORT.md
- Lokalis futtatas:

```bash
pytest
```

- Teljes suite:

```bash
python scripts/run_all_tests.py
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
- A historikus paper runner fallback HOLD dontest ad, ha nincs RL modell.
- Validacio outcome-ok nelkul no_data-t ad az effectiveness/trust metrikakra.
