# Frequently Asked Questions (FAQ) - EN + HU

## Installation and Setup

### Q1 (EN): What do I need to run the project?
**Answer (EN):** A Python 3.x environment, a working virtual environment, and internet access for market data. Use requirements.txt and the project venv.
**Valasz (HU):** Python 3.x, mukodo virtualis kornyezet, valamint internetkapcsolat a piaci adatokhoz. Hasznald a requirements.txt-t es a projekt venv-et.

### Q2 (EN): Do I need a Raspberry Pi?
**Answer (EN):** No. Raspberry Pi is optional for always-on deployment.
**Valasz (HU):** Nem. A Raspberry Pi csak opcionális 0-24 uzemhez.

### Q3 (EN): How do I install dependencies?
**Answer (EN):**
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
**Valasz (HU):** A fenti parancsokkal hozd letre es aktiváld a venv-et, majd telepitsd a fuggosegeket.

## Running the System

### Q4 (EN): What is the primary CLI entry point?
**Answer (EN):** main.py in the project root.
**Valasz (HU):** A fo CLI belépesi pont a main.py.

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
**Valasz (HU):** Historikus futasnal fallback HOLD döntes keszul, decision_source= fallback, nincs trade.

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
**Valasz (HU):** Lásd docs/README.md vagy app/ui/admin_dashboard.py.

## Data and Storage

### Q13 (EN): Where is data stored?
**Answer (EN):** SQLite database at the configured path (Config.DB_PATH).
**Valasz (HU):** SQLite adatbazisban (Config.DB_PATH).

### Q14 (EN): What is stored in decision_history?
**Answer (EN):** Decisions, audit metadata, model votes, position sizing, and decision_source for traceability.
**Valasz (HU):** Döntesek, audit meta, model vote-ok, pozicio meretezes es decision_source.

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
**Valasz (HU):** Kövesd a docs/deployment/RASPBERRY_PI_SETUP_GUIDE_HU.md leirast es futtasd a deploy_rpi.sh-t.

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

### Q29: "ModuleNotFoundError: No module named 'app'"
**A:**
```bash
# Ensure you're in project root
cd tozsde_webapp

# Use Python module syntax
python -m pytest tests/
python -m app.scripts.daily_pipeline
```

### Q30: "Database is locked" error
**A:**
```bash
# Close other connections
# In Python: always use context managers
with sqlite3.connect('market_data.db') as conn:
    # Your code

# Or close explicitly
conn.close()
```

### Q31: Flask app won't start - "Port 5000 already in use"
**A:**
```bash
# Find process using port 5000
netstat -ano | findstr :5000

# Kill the process (Windows)
taskkill /PID <PID> /F

# Or use different port
python run_dev.py --port 5001
```

### Q32: Tests pass locally but fail on Raspberry Pi
**A:** Common causes:
- **Python version:** Check `python3 --version`
- **Missing dependencies:** Run `pip install -r requirements.txt`
- **File permissions:** `chmod +x deploy_rpi.sh`
- **Disk space:** `df -h` (need 2GB+ free)

### Q33: Data loading fails - "No data available"
**A:**
1. **Check internet:** `ping yahoo.com`
2. **Verify ticker:** Use correct symbol (e.g., "AAPL" not "Apple")
3. **Check rate limit:** Yahoo Finance limits ~2000 req/hour
4. **Try different date range:** Some tickers have limited history

### Q34: Performance is slow
**A:** Optimization tips:
- **Database:** Run `VACUUM` and `ANALYZE`
- **Cache:** DataManager caches recent data
- **Parallel processing:** Enable multiprocessing in config
- **Disk space:** Keep > 20% free
- **Memory:** Close unused programs

### Q35: Where can I get help?
**A:**
1. **Documentation:** Check `docs/` folder
2. **Logs:** Review `logs/app.log` for errors
3. **Tests:** Run `pytest -v` to diagnose
4. **Troubleshooting:** See [TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)

---

## Advanced Topics

### Q36: How do I add a new indicator?
**A:**
```python
# In app/indicators/technical.py
def calculate_my_indicator(df, period=14):
    """Calculate My Custom Indicator."""
    # Your logic here
    return indicator_series
```

### Q37: Can I customize the decision logic?
**A:** Yes! Modify:
- `app/decision/decision_engine.py` - Main decision logic
- `app/decision/decision_policy.py` - Trading rules
- `app/decision/allocation.py` - Position sizing

### Q38: How do I add a new strategy?
**A:**
```python
# In app/decision/adaptive_strategy_selector.py
def my_custom_strategy(self, df):
    """My custom trading strategy."""
    signals = []
    # Your strategy logic
    return signals
```

### Q39: Can I integrate with other data sources?
**A:** Yes! Extend `DataLoader`:
```python
# In app/data_access/data_loader.py
def load_from_custom_source(self, ticker, start, end):
    # Your custom data source
    return ohlcv_dataframe
```

### Q40: How do I contribute improvements?
**A:**
1. **Run tests:** `pytest tests/ -v`
2. **Check coverage:** `pytest --cov=app`
3. **Follow conventions:** PEP 8 style guide
4. **Document changes:** Update `SPRINTS.md`
5. **Create PR:** Submit for review

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
python -m app.scripts.daily_pipeline

# Health check
curl http://localhost:5000/admin/health
```

### Important Files
- `requirements.txt` - Python dependencies
- `pytest.ini` - Test configuration
- `app/config/config.py` - Main configuration
- `app/config/pi_config.py` - Raspberry Pi config
- `logs/app.log` - Application logs

### Key Directories
- `app/` - Source code
- `tests/` - Test suite
- `docs/` - Documentation
- `logs/` - Log files
- `htmlcov/` - Coverage reports

---

**Last Updated:** 2026-02-02  
**Version:** 1.0 (Sprint 9)  
**Test Status:** 351/351 passing ✅  
**Coverage:** 59% 📊
