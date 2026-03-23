# Frequently Asked Questions (FAQ) - EN + HU

## Installation and Setup

### Q1 (EN): What do I need to run the project?
**Answer (EN):** A Python 3.x environment, a working virtual environment, and internet access for market data. Use requirements.txt and the project venv.
**Valasz (HU):** Python 3.x, mukodo virtualis kornyezet, valamint internetkapcsolat a piaci adatokhoz. Hasznald a requirements.txt-t es a projekt venv-et.

### Q2 (EN): Do I need a Raspberry Pi?
**Answer (EN):** No. Raspberry Pi is optional for always-on deployment.
**Valasz (HU):** Nem. A Raspberry Pi csak opcionalis 0-24 uzemhez.

### Q3 (EN): How do I install dependencies?
**Answer (EN):**
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
**Valasz (HU):** A fenti parancsokkal hozd letre es aktivald a venv-et, majd telepitsd a fuggosegeket.

## Running the System

### Q4 (EN): What is the primary CLI entry point?
**Answer (EN):** main.py in the project root.
**Valasz (HU):** A fo CLI belepesi pont a main.py.

### Q5 (EN): How do I run the daily pipeline?
**Answer (EN):**
```bash
python main.py daily
python main.py daily --ticker VOO
```
**Valasz (HU):** A fenti parancsokkal futtasd a napi pipeline-t.

### Q6 (EN): How do I run a historical paper backfill?
**Answer (EN):**
```bash
python main.py run-paper-history --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
```
**Valasz (HU):** A fenti parancsokkal futtasd a historikus paper runner-t.

### Q7 (EN): What happens if there are no RL model files?
**Answer (EN):** Historical paper runs generate fallback HOLD decisions and mark decision_source as fallback. No trade is executed.
**Valasz (HU):** Historikus futasnal fallback HOLD dontes keszul, decision_source= fallback, nincs trade.

## Validation and Reporting

### Q8 (EN): How do I run Phase 5 validation?
**Answer (EN):**
```bash
python main.py validate --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
```
**Valasz (HU):** A fenti parancs futtatja a Phase 5 validaciot.

### Q9 (EN): How do I append validation results to the test report?
**Answer (EN):**
```bash
python scripts/run_tests_with_report.py --skip-tests --with-validation --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
```
**Valasz (HU):** A fenti parancs a validaciot a teszt riportba irja.

### Q10 (EN): Why do effectiveness or model trust show no_data?
**Answer (EN):** Those metrics require outcomes and model_votes. If no outcomes are recorded or decisions have no model votes, the analyzers return no_data.
**Valasz (HU):** Outcome-ok es model vote-ok szuksegesek. Ha hianyoznak, no_data jelenik meg.

## Admin API

### Q11 (EN): How do I call admin endpoints?
**Answer (EN):** Use X-Admin-Key header and /admin paths.
**Valasz (HU):** X-Admin-Key header es /admin endpointok szuksegesek.

```bash
curl http://localhost:5000/admin/health -H "X-Admin-Key: <key>"
```

### Q12 (EN): Where is the list of endpoints?
**Answer (EN):** See docs/README.md or app/ui/admin_dashboard.py.
**Valasz (HU):** Lasd docs/README.md vagy app/ui/admin_dashboard.py.

## Data and Storage

### Q13 (EN): Where is data stored?
**Answer (EN):** SQLite database at the configured path (Config.DB_PATH).
**Valasz (HU):** SQLite adatbazisban (Config.DB_PATH).

### Q14 (EN): What is stored in decision_history?
**Answer (EN):** Decisions, audit metadata, model votes, position sizing, and decision_source for traceability.
**Valasz (HU):** Dontesek, audit meta, model vote-ok, pozicio meretezes es decision_source.

## Tests

### Q15 (EN): How do I run tests?
**Answer (EN):**
```bash
pytest
```
Or one command:
```bash
python scripts/run_all_tests.py
```
**Valasz (HU):**
```bash
pytest
```
Vagy egy parancs:
```bash
python scripts/run_all_tests.py
```

### Q16 (EN): Where do I see coverage and the latest test snapshot?
**Answer (EN):** docs/testing/TEST_STATUS_REPORT.md.
**Valasz (HU):** docs/testing/TEST_STATUS_REPORT.md.

## Deployment

### Q17 (EN): How do I deploy on Raspberry Pi?
**Answer (EN):** Follow docs/deployment/RASPBERRY_PI_SETUP_GUIDE.md and run deploy_rpi.sh.
**Valasz (HU):** Kovesd a docs/deployment/RASPBERRY_PI_SETUP_GUIDE_HU.md leirast es futtasd a deploy_rpi.sh-t.

### Q18 (EN): What about health checks on Pi?
**Answer (EN):** Admin health endpoint is /admin/health and requires X-Admin-Key. If you use health_check.sh, update its URL or headers accordingly.
**Valasz (HU):** /admin/health endpoint X-Admin-Key headerrel hasznalhato. A health_check.sh scriptet ehhez igazitsd.

## Troubleshooting

### Q19 (EN): I see ModuleNotFoundError: app
**Answer (EN):** Run from the project root and use the venv python.
**Valasz (HU):** Futass a projekt gyokerbol es a venv python-t hasznald.

### Q20 (EN): Validation report is not updating
**Answer (EN):** Use the run_tests_with_report.py script with --with-validation and ensure you are in the repo root.
**Valasz (HU):** A run_tests_with_report.py szkriptet hasznald, es ellenorizd a munkakonyvtarat.
For troubleshooting steps, see docs/TROUBLESHOOTING_GUIDE.md.

---

## Quick Reference

### Common Commands
```bash
# Run tests
pytest tests/ -v

# Coverage report
pytest --cov=app --cov-report=html

# Start Flask app
python run_dev.py

# Daily pipeline
python main.py daily --ticker VOO

# Health check (admin key required)
curl http://localhost:5000/admin/health -H "X-Admin-Key: <key>"
```

---

**Last Updated:** 2026-03-21  
**Version:** 1.1  
**Source of truth for test/coverage:** [testing/TEST_STATUS_REPORT.md](testing/TEST_STATUS_REPORT.md)
