# Frequently Asked Questions (FAQ)

## 📚 Table of Contents
- [Installation & Setup](#installation--setup)
- [Testing & Development](#testing--development)
- [Deployment](#deployment)
- [API Usage](#api-usage)
- [Trading & Strategies](#trading--strategies)
- [Troubleshooting](#troubleshooting)

---

## Installation & Setup

### Q1: What are the minimum system requirements?
**A:** 
- **Python:** 3.6 or higher
- **RAM:** 2GB minimum (4GB recommended)
- **Disk:** 5GB free space
- **OS:** Windows 10+, Linux (Ubuntu 18.04+), macOS 10.14+
- **Internet:** Stable connection for data fetching

### Q2: Do I need a Raspberry Pi to run this?
**A:** No. The system works on any computer. Raspberry Pi is optional for 24/7 deployment:
- **Development:** Use Windows/Mac/Linux desktop
- **Production (optional):** Deploy to Raspberry Pi for always-on trading

### Q3: How do I install dependencies?
**A:**
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install all requirements
pip install -r requirements.txt
```

### Q4: What Python version should I use?
**A:** Python 3.6 - 3.9 recommended. Tested with:
- ✅ Python 3.6.6
- ✅ Python 3.7.x
- ✅ Python 3.8.x
- ⚠️ Python 3.10+ (some dependencies may need updates)

### Q5: Can I run this on macOS with M1/M2 chip?
**A:** Yes, but use Rosetta 2 for compatibility:
```bash
arch -x86_64 python3 -m venv .venv
arch -x86_64 .venv/bin/pip install -r requirements.txt
```

---

## Testing & Development

### Q6: How do I run all tests?
**A:**
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_backtester.py -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

### Q7: Tests are failing. What should I do?
**A:**
1. **Check virtual environment:** Ensure `.venv` is activated
2. **Update dependencies:** `pip install -r requirements.txt --upgrade`
3. **Check database:** Some tests need SQLite files
4. **See troubleshooting:** Check [TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)

### Q8: How do I add a new test?
**A:**
```python
# In tests/test_my_module.py
import pytest
from app.my_module import MyClass

def test_my_function():
    obj = MyClass()
    result = obj.my_function(param=123)
    assert result == expected_value
```

### Q9: What's the current test coverage?
**A:** **59%** (as of Sprint 9). Coverage report:
- Run: `pytest --cov=app --cov-report=html`
- View: Open `htmlcov/index.html`

### Q10: How do I debug a failing test?
**A:**
```bash
# Run single test with verbose output
pytest tests/test_file.py::test_function -vv

# Drop into debugger on failure
pytest tests/test_file.py --pdb

# Print all output
pytest tests/test_file.py -s
```

---

## Deployment

### Q11: How do I deploy to Raspberry Pi?
**A:** Follow [RASPBERRY_PI_SETUP_GUIDE.md](deployment/RASPBERRY_PI_SETUP_GUIDE.md):
```bash
# On Raspberry Pi
bash deploy_rpi.sh
```

### Q12: Can I run this without Flask/web UI?
**A:** Yes! Use standalone mode:
```python
# In main.py
from app.scripts.daily_pipeline import DailyPipeline

pipeline = DailyPipeline()
pipeline.run_full_pipeline()
```

### Q13: How do I set up the cron job?
**A:**
```bash
# Edit crontab
crontab -e

# Add daily pipeline (runs at 6 AM)
0 6 * * * cd /opt/tozsde_webapp && .venv/bin/python -m app.scripts.daily_pipeline
```

### Q14: How do I monitor the system?
**A:** Use health check endpoint:
```bash
# Check system health
curl http://localhost:5000/admin/health

# Or via Python
from app.infrastructure.health_check import HealthChecker
checker = HealthChecker()
status = checker.check_all()
```

### Q15: Can I deploy to cloud (AWS/Azure/GCP)?
**A:** Yes! System requirements:
- **VM:** 2 CPU, 4GB RAM minimum
- **Storage:** 20GB SSD
- **Python:** 3.6+ pre-installed
- **Firewall:** Allow port 5000 (or use nginx proxy)
- **Process manager:** Use systemd or supervisord

---

## API Usage

### Q16: How do I access the Admin Dashboard?
**A:**
```bash
# Start Flask app
python run_dev.py

# Open browser
http://localhost:5000/admin/health
```

### Q17: What API endpoints are available?
**A:** Sprint 9 endpoints:

**Health:**
- `GET /admin/health` - System health check

**Performance:**
- `GET /admin/performance/summary?days=30` - Performance metrics
- `GET /admin/performance/detailed?days=90` - Detailed analytics
- `GET /admin/performance/chart-data?days=180` - Chart data

**Errors:**
- `GET /admin/errors/summary` - Error statistics
- `GET /admin/errors/recent?limit=50` - Recent errors
- `GET /admin/errors/critical` - Critical errors only
- `POST /admin/errors/export` - Export error log

**Capital:**
- `GET /admin/capital/status` - Current capital status
- `GET /admin/capital/history?days=30` - Capital history
- `GET /admin/capital/allocation` - Current allocation
- `GET /admin/capital/projection` - Future projection

### Q18: How do I authenticate API requests?
**A:** Currently no authentication (localhost only). For production:
```python
# Add authentication in app/ui/admin_dashboard.py
from flask_httpauth import HTTPBasicAuth

auth = HTTPBasicAuth()

@auth.verify_password
def verify_password(username, password):
    # Check credentials
    return username == "admin" and password == "secret"

@dashboard_bp.route('/admin/health')
@auth.login_required
def health():
    # ...
```

### Q19: Can I use the API from JavaScript?
**A:** Yes:
```javascript
// Fetch performance data
fetch('http://localhost:5000/admin/performance/summary?days=30')
  .then(response => response.json())
  .then(data => {
    console.log('Total Return:', data.total_return);
    console.log('Sharpe Ratio:', data.sharpe_ratio);
  });
```

### Q20: How do I export data via API?
**A:**
```bash
# Export error log
curl -X POST http://localhost:5000/admin/errors/export \
  -H "Content-Type: application/json" \
  -d '{"format": "csv", "severity": "CRITICAL"}' \
  -o errors.csv
```

---

## Trading & Strategies

### Q21: What trading strategies are supported?
**A:** Multiple strategies (Sprint 5):
- **Trend Following:** Momentum, breakout detection
- **Mean Reversion:** Buy dips, sell rallies
- **Risk Parity:** Equal risk allocation
- **Adaptive:** Market regime-based switching

### Q22: How does the system decide what to buy/sell?
**A:** Decision pipeline:
1. **Data Loading:** Fetch OHLCV from Yahoo Finance
2. **Indicator Calculation:** RSI, MACD, Bollinger Bands, etc.
3. **Regime Detection:** Bull/bear/sideways market
4. **Strategy Selection:** Choose best strategy for current regime
5. **Position Sizing:** Calculate allocation using Kelly Criterion
6. **Risk Management:** Check correlation, diversification
7. **Signal Generation:** BUY/SELL/HOLD recommendations

### Q23: Can I backtest strategies?
**A:** Yes:
```python
from app.backtesting.backtester import Backtester
from app.config.config import Config

config = Config()
backtester = Backtester(config)

results = backtester.run_backtest(
    tickers=["AAPL", "MSFT", "GOOGL"],
    start_date="2020-01-01",
    end_date="2023-12-31",
    initial_capital=10000
)

print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
```

### Q24: How often does the system trade?
**A:** Configurable:
- **Daily Pipeline:** Runs once per day (default: 6 AM)
- **Rebalancing:** Weekly (default: Monday)
- **Emergency Exits:** Real-time monitoring (if enabled)

### Q25: What's the typical return?
**A:** **Disclaimer:** Past performance doesn't guarantee future results. Backtesting shows:
- **Avg Annual Return:** 8-15% (varies by strategy)
- **Sharpe Ratio:** 0.8-1.5
- **Max Drawdown:** 15-25%
- **Win Rate:** 55-65%

### Q26: How does risk management work?
**A:** Multi-layered approach:
- **Position Sizing:** Kelly Criterion (Sprint 2)
- **Correlation Limits:** Max 0.7 between assets (Sprint 7)
- **Drawdown Protection:** Exit if > 20% loss (Sprint 6)
- **Diversification Score:** Minimum 0.6 required
- **Capital Allocation:** Risk parity weighting

### Q27: Can I paper trade first?
**A:** Yes! Set in config:
```python
# In app/config/config.py
PAPER_TRADING = True  # No real money
REAL_TRADING = False
```

### Q28: What brokers are supported?
**A:** Currently:
- **Data Source:** Yahoo Finance (yfinance)
- **Broker Integration:** Not implemented (Sprint 10+ roadmap)
- **Manual Trading:** System provides recommendations, you execute manually

---

## Troubleshooting

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
