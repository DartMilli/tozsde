# Troubleshooting Guide - EN + HU

## English

### 1) Test Failures
**Symptom:** pytest exits with errors.

**Fix:**
```bash
pytest tests/test_xyz.py::test_function -v
.venv\Scripts\python.exe -m pytest
```

### 2) Module Import Errors
**Symptom:** ModuleNotFoundError: No module named app.

**Fix:**
```bash
cd c:\tozsde
$env:PYTHONPATH = "$(Get-Location)"
.venv\Scripts\python.exe -m pytest
```

### 3) Validation Report Not Updating
**Fix:**
```bash
python scripts/run_tests_with_report.py --skip-tests --with-validation --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
```

### 4) Effectiveness or Model Trust Shows no_data
**Cause:** outcomes or model_votes are missing.

**Fix:**
- Ensure paper execution produced outcomes.
- Ensure decisions include model_votes when models exist.

### 5) Admin API Returns 401
**Cause:** Missing X-Admin-Key header.

**Fix:**
```bash
curl http://localhost:5000/admin/health -H "X-Admin-Key: <key>"
```

### 6) Health Check Script Uses /api/health
**Cause:** Default health_check.sh points to /api/health.

**Fix:**
- Update the script to use /admin/health and add X-Admin-Key.
- Or provide a compatible endpoint in your deployment (if you choose).

### 7) SQLite Database Locked
**Fix:**
- Close other connections.
- Remove stale journal/wal files if they exist.

### 8) Data Loading Failures
**Fix:**
- Verify ticker symbol.
- Confirm network connectivity.
- Use cached data when offline.

### 9) Cron Jobs Not Running (Pi)
**Fix:**
```bash
crontab -l
sudo systemctl status cron
```

### 10) Flask API Port Conflict
**Fix:**
```bash
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

## Magyar

### 1) Teszthibak
**Tunet:** pytest hibat dob.

**Megoldas:**
```bash
pytest tests/test_xyz.py::test_function -v
.venv\Scripts\python.exe -m pytest
```

### 2) Modul import hiba
**Tunet:** ModuleNotFoundError: No module named app.

**Megoldas:**
```bash
cd c:\tozsde
$env:PYTHONPATH = "$(Get-Location)"
.venv\Scripts\python.exe -m pytest
```

### 3) Validacios riport nem frissul
**Megoldas:**
```bash
python scripts/run_tests_with_report.py --skip-tests --with-validation --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
```

### 4) Effectiveness vagy Model Trust no_data
**Ok:** nincs outcome vagy model_votes adat.

**Megoldas:**
- Ellenorizd, hogy a paper execution outcome-okat irt-e.
- Ellenorizd, hogy vannak model vote-ok.

### 5) Admin API 401
**Ok:** hianyzik az X-Admin-Key.

**Megoldas:**
```bash
curl http://localhost:5000/admin/health -H "X-Admin-Key: <key>"
```

### 6) Health check script /api/health-et hasznal
**Ok:** a health_check.sh alapertelmezetten /api/health-et hiv.

**Megoldas:**
- Modositsd /admin/health-re es adj hozza X-Admin-Key-t.
- Vagy tegyel kompatibilis endpointot a deployba (ha ezt valasztod).

### 7) SQLite locked
**Megoldas:**
- Zarj be mas kapcsolatokat.
- Torold a beragadt journal/wal fajlokat, ha vannak.

### 8) Adatbetoltesi hiba
**Megoldas:**
- Ellenorizd a ticker nevét.
- Ellenorizd a halozatot.
- Offline modban csak cache-t hasznalj.

### 9) Cron nem fut (Pi)
**Megoldas:**
```bash
crontab -l
sudo systemctl status cron
```

### 10) Flask port ures
**Megoldas:**
```bash
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

**Solutions:**
1. **Check sudo permissions:**
   ```bash
   sudo -v  # Test sudo access
   ```

2. **Verify Python version:**
   ```bash
   python3 --version  # Should be 3.6+
   ```

3. **Check disk space:**
   ```bash
   df -h /
   # Need at least 2GB free
   ```

4. **Run deployment steps manually:**
   ```bash
   # Follow RASPBERRY_PI_SETUP_GUIDE.md
   # Section: Manual Installation
   ```

---

### 10. **Admin Dashboard Errors**

#### Symptom: Dashboard endpoints return errors

**Solutions:**
1. **Check Flask app is running:**
   ```bash
   curl http://localhost:5000/admin/health
   ```

2. **Verify database exists:**
   ```bash
   ls -la market_data.db error_log.db
   ```

3. **Check error logs:**
   ```python
   # In code or via endpoint
   from app.infrastructure.error_reporter import ErrorReporter
   reporter = ErrorReporter(db_path="error_log.db")
   errors = reporter.get_recent_errors(limit=10)
   ```

4. **Test endpoints individually:**
   ```bash
   curl http://localhost:5000/admin/performance/summary
   curl http://localhost:5000/admin/errors/summary
   ```

---

## 🔍 Diagnostic Commands

### System Health Check
```bash
# Full system check
pytest tests/ -v
python -m app.infrastructure.health_check
curl http://localhost:5000/admin/health
```

### Database Integrity
```bash
# Check SQLite databases
sqlite3 market_data.db "PRAGMA integrity_check;"
sqlite3 decision_log.db "PRAGMA integrity_check;"
sqlite3 error_log.db "PRAGMA integrity_check;"
```

### Log Analysis
```bash
# Recent errors
tail -100 logs/app.log | grep ERROR

# Daily pipeline status
grep "Pipeline completed" logs/daily_pipeline.log | tail -10

# Error statistics
python -c "from app.infrastructure.error_reporter import ErrorReporter; r=ErrorReporter(db_path='error_log.db'); print(r.get_error_statistics())"
```

---

## 📞 Getting Help

### 1. **Check Logs First**
```bash
# Application logs
tail -f logs/app.log

# Specific module logs
tail -f logs/decision_engine.log
tail -f logs/data_loader.log
```

### 2. **Run Tests with Verbose Output**
```bash
pytest tests/ -vv --tb=short
```

### 3. **Check Documentation**
- [SPRINTS.md](../SPRINTS.md) - Complete feature history
- [README.md](../README.md) - Project overview
- [RASPBERRY_PI_SETUP_GUIDE.md](../deployment/RASPBERRY_PI_SETUP_GUIDE.md) - Deployment guide

### 4. **Common Log Locations**
- **Application:** `logs/app.log`
- **Daily Pipeline:** `logs/daily_pipeline.log`
- **Errors:** `logs/error.log`
- **System (Pi):** `/var/log/syslog`

---

## 🛠️ Advanced Debugging

### Enable Debug Mode
```python
# In config.py or pi_config.py
DEBUG = True
LOG_LEVEL = "DEBUG"
```

### Profile Performance
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### Database Query Debugging
```python
import sqlite3

conn = sqlite3.connect('market_data.db')
conn.set_trace_callback(print)  # Log all SQL queries

# Your code here
conn.close()
```

---

## ✅ Prevention Best Practices

1. **Always use virtual environment**
2. **Run tests before deployment**
3. **Monitor disk space (>20% free)**
4. **Check logs daily**
5. **Keep backups (weekly)**
6. **Use health checks (5-min intervals)**
7. **Document configuration changes**
8. **Test on paper trading first**

---

## 📊 Status Indicators

### Healthy System
- ✅ All tests passing (359/359)
- ✅ API responds < 500ms
- ✅ No critical errors (24h)
- ✅ Disk space > 20%
- ✅ Memory usage < 80%
- ✅ Daily pipeline completes
- ✅ Cron jobs executing

### Warning Signs
- ⚠️ Test failures increasing
- ⚠️ API response time > 1s
- ⚠️ Critical errors present
- ⚠️ Disk space < 20%
- ⚠️ Memory usage > 80%
- ⚠️ Pipeline taking > 10 min

### Critical Issues
- 🚨 System crash/restart
- 🚨 Database corruption
- 🚨 No trades for 3+ days
- 🚨 Error rate > 100/hour
- 🚨 Disk full
- 🚨 Memory leak detected

---

**Last Updated:** 2026-02-02  
**Version:** 1.0 (Sprint 9)
