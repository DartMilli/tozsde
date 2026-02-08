# Troubleshooting Guide - ToZsDE Trading System

## 📋 Quick Reference

This guide helps diagnose and fix common issues with the ToZsDE trading system.

---

## 🚨 Common Issues & Solutions

### 1. **Test Failures**

#### Symptom: `pytest` exits with errors
```bash
FAILED tests/test_xyz.py::test_function
```

**Solutions:**
1. **Run specific test to see full error:**
   ```bash
   pytest tests/test_xyz.py::test_function -v
   ```

2. **Check Python environment:**
   ```bash
   .venv/Scripts/python.exe --version  # Should be 3.6+
   ```

3. **Reinstall dependencies:**
   ```bash
   .venv/Scripts/pip.exe install -r requirements.txt
   ```

4. **Database issues:** Some tests require SQLite database. Check temp file permissions.

---

### 2. **Module Import Errors**

#### Symptom: `ModuleNotFoundError: No module named 'app'`

**Solutions:**
1. **Ensure you're in project root:**
   ```bash
   pwd  # Should show: .../tozsde_webapp
   ```

2. **Check PYTHONPATH:**
   ```bash
   # Run from project root
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
   $env:PYTHONPATH = "$(Get-Location)"      # Windows PowerShell
   ```

3. **Use virtual environment:**
   ```bash
   .venv/Scripts/python.exe -m pytest  # Always use this
   ```

#### Symptom: Validation report not updating

**Solutions:**
1. **Run validation with report integration:**
   ```bash
   python scripts/run_tests_with_report.py --with-validation --skip-tests --ticker VOO --start-date 2020-01-01 --end-date 2024-01-01
   ```
2. **Ensure you are in repo root:**
   ```bash
   cd c:\tozsde
   ```

---

### 3. **Database Errors**

#### Symptom: `sqlite3.OperationalError: database is locked`

**Solutions:**
1. **Close other connections:**
   ```python
   # In code: always use context managers
   with sqlite3.connect(db_path) as conn:
       # Your code here
       pass  # Connection auto-closes
   ```

2. **Check file permissions:**
   ```bash
   # Linux/Mac
   ls -la market_data.db
   chmod 644 market_data.db
   
   # Windows
   icacls market_data.db
   ```

3. **Clear lock files:**
   ```bash
   rm market_data.db-journal  # If exists
   rm market_data.db-wal      # If exists
   ```

---

### 4. **Flask API Not Starting**

#### Symptom: `Address already in use: Port 5000`

**Solutions:**
1. **Find process using port:**
   ```bash
   # Windows
   netstat -ano | findstr :5000
   taskkill /PID <PID> /F
   
   # Linux/Mac
   lsof -ti:5000 | xargs kill -9
   ```

2. **Use different port:**
   ```python
   # In main.py or config
   app.run(host='0.0.0.0', port=5001)
   ```

3. **Check systemd service:**
   ```bash
   # On Raspberry Pi
   sudo systemctl status tozsde
   sudo systemctl restart tozsde
   ```

---

### 5. **Cron Jobs Not Running**

#### Symptom: Daily pipeline doesn't execute

**Solutions:**
1. **Check cron status:**
   ```bash
   # Linux/Raspberry Pi
   crontab -l  # List cron jobs
   systemctl status cron
   ```

2. **Check logs:**
   ```bash
   # Daily pipeline log
   tail -f logs/daily_pipeline.log
   
   # System cron log
   grep CRON /var/log/syslog
   ```

3. **Test manually:**
   ```bash
   cd /opt/tozsde_webapp
   .venv/bin/python -m app.scripts.daily_pipeline
   ```

4. **Verify cron syntax:**
   ```cron
   # Daily at 6:00 AM
   0 6 * * * cd /opt/tozsde_webapp && .venv/bin/python -m app.scripts.daily_pipeline
   ```

---

### 6. **Data Loading Failures**

#### Symptom: `No data available for ticker XYZ`

**Solutions:**
1. **Check internet connection:**
   ```bash
   ping yahoo.com
   curl -I https://query1.finance.yahoo.com
   ```

2. **Verify ticker symbol:**
   ```python
   # Correct: "AAPL", "MSFT", "SPY"
   # Wrong: "Apple", "Microsoft", "S&P500"
   ```

3. **Check rate limiting:**
   - Yahoo Finance limits: ~2000 requests/hour
   - Add delays: `time.sleep(1)` between requests

4. **Use cache:**
   ```python
   # DataManager caches recent data automatically
   # Check: market_data.db
   ```

---

### 7. **Performance Issues**

#### Symptom: System running slow

**Solutions:**
1. **Check disk space:**
   ```bash
   df -h  # Linux/Mac
   Get-PSDrive C  # Windows
   ```

2. **Monitor memory:**
   ```bash
   free -h  # Linux/Mac
   Get-Process python  # Windows
   ```

3. **Optimize database:**
   ```bash
   sqlite3 market_data.db "VACUUM;"
   sqlite3 market_data.db "ANALYZE;"
   ```

4. **Clear old logs:**
   ```bash
   find logs/ -type f -mtime +30 -delete  # >30 days
   ```

---

### 8. **Backtesting Errors**

#### Symptom: Backtester returns empty results

**Solutions:**
1. **Check date range:**
   ```python
   # Ensure sufficient history
   start_date = "2020-01-01"  # Not too recent
   end_date = "2023-12-31"
   ```

2. **Verify data availability:**
   ```python
   from app.data_access.data_manager import DataManager
   dm = DataManager()
   data = dm.get_ohlcv("AAPL", start_date, end_date)
   print(len(data))  # Should be > 0
   ```

3. **Check strategy configuration:**
   ```python
   # Ensure strategy is properly configured
   strategy_config = {...}  # Must have all required keys
   ```

---

### 9. **Deployment Issues (Raspberry Pi)**

#### Symptom: `deploy_rpi.sh` fails

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
