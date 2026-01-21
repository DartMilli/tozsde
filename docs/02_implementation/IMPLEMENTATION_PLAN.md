# IMPLEMENTATION PLAN (Q1 2026)
**Priority:** Critical gaps blocking production maturity  
**Timeline:** 8 weeks of development

---

## SPRINT 1 (Weeks 1–2): TEST SUITE & QUALITY ASSURANCE

### 1.1 Create Test Infrastructure
**Effort:** 4 hours | **Impact:** HIGH (unblocks refactoring)

**What:** Build `tests/` directory with pytest fixtures and conftest.

**Where:** Create new files:
```
tests/
  __init__.py
  conftest.py          # Fixtures (mock DB, config, sample data)
  test_indicators.py   # Technical indicators
  test_fitness.py      # Genetic optimizer fitness functions
  test_backtester.py   # Backtesting core logic
  test_walk_forward.py # Walk-forward windows
  test_data_manager.py # SQLite DAL operations
  test_allocation.py   # Capital allocation logic
```

**Code stub — `tests/conftest.py`:**
```python
import pytest
import tempfile
import os
from pathlib import Path
from app.config.config import Config
from app.data_access.data_manager import DataManager

@pytest.fixture(scope="session")
def test_db():
    """Create temporary test database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        Config.DATA_DIR = Path(tmpdir)
        Config.DB_PATH = Path(tmpdir) / "test.db"
        dm = DataManager()
        dm.initialize_tables()
        yield dm

@pytest.fixture
def sample_ohlcv():
    """Sample OHLCV data for testing."""
    return {
        "ticker": "TEST",
        "date": "2025-01-20",
        "open": 100.0,
        "high": 105.0,
        "low": 99.0,
        "close": 102.0,
        "volume": 1000000.0
    }
```

**Steps:**
1. `mkdir tests && touch tests/__init__.py`
2. Create `conftest.py` with fixtures above
3. Add `pytest` to test requirements if not present

---

### 1.2 Unit Tests — Indicators Module
**Effort:** 6 hours | **Impact:** MEDIUM (validates core logic)

**Where:** `tests/test_indicators.py`

**What to test:**
- SMA calculation correctness (vs. manual validation)
- Edge cases (short windows, NaN values)
- Performance (large datasets)

**Code stub:**
```python
import pytest
import numpy as np
import pandas as pd
from app.indicators.technical import sma

def test_sma_basic():
    """SMA of [1,2,3,4,5] over 3 periods = [NaN, NaN, 2, 3, 4]."""
    data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sma(data, 3)
    expected = pd.Series([np.nan, np.nan, 2.0, 3.0, 4.0])
    pd.testing.assert_series_equal(result, expected)

def test_sma_window_larger_than_data():
    """SMA with window > data length should return all NaN."""
    data = pd.Series([1.0, 2.0, 3.0])
    result = sma(data, 10)
    assert result.isna().all()

def test_sma_constant_series():
    """SMA of constant series should be the constant."""
    data = pd.Series([5.0] * 10)
    result = sma(data, 3)
    assert (result[2:] == 5.0).all()
```

---

### 1.3 Unit Tests — Fitness & Backtester
**Effort:** 8 hours | **Impact:** HIGH (validates profit calculations)

**Where:** `tests/test_fitness.py`, `tests/test_backtester.py`

**What to test:**
- Fitness function edge cases (zero trades, all losses)
- Backtester Sharpe ratio calculation
- Walk-forward window boundaries

**Code stub — `tests/test_backtester.py`:**
```python
from app.backtesting.backtester import Backtester
import pandas as pd
import numpy as np

def test_backtester_no_trades():
    """Backtester with no trades should return 0% return."""
    df = pd.DataFrame({
        "close": [100, 101, 102, 103, 104],
        "signal": [0, 0, 0, 0, 0]  # All HOLD
    })
    bt = Backtester(df, "TEST")
    results = bt.backtest()
    assert results["total_return"] == 0.0

def test_backtester_transaction_costs():
    """Verify transaction costs are applied."""
    df = pd.DataFrame({
        "close": [100, 101, 102, 103, 104],
        "signal": [0, 1, 0, 0, 0]  # Buy then hold
    })
    bt = Backtester(df, "TEST")
    results = bt.backtest()
    # Profit should be reduced by ~0.2% (buy + sell fees)
    assert results["total_return"] < 0.01  # ~1% gross, <0.8% net
```

---

### 1.4 Integration Test — Daily Pipeline
**Effort:** 4 hours | **Impact:** HIGH (validates end-to-end flow)

**Where:** `tests/test_daily_pipeline.py`

**What to test:**
- `run_daily()` completes without errors
- Recommendations saved to DB
- Email formatted correctly

**Code stub:**
```python
import pytest
from unittest.mock import patch, MagicMock
from main import run_daily
from app.data_access.data_manager import DataManager

@patch("app.notifications.mailer.send_email")
def test_run_daily_success(mock_send_email, test_db):
    """Daily pipeline should complete and send email."""
    dm = test_db
    dm.initialize_tables()
    
    # Mock external calls
    with patch("app.data_access.data_loader.load_data") as mock_load:
        mock_load.return_value = pd.DataFrame({...})  # Mock OHLCV
        
        result = run_daily()
        
        # Verify email sent
        assert mock_send_email.called
        
        # Verify recommendations in DB
        recs = dm.get_today_recommendations()
        assert len(recs) > 0
```

---

## SPRINT 2 (Weeks 3–4): LEARNING SYSTEM & RL ACTIVATION

### 2.1 Enable RL Module Safely
**Effort:** 3 hours | **Impact:** HIGH (enables P8 features)

**Where:** [config.py](app/config/config.py#L20)

**Change:**
```python
# BEFORE
ENABLE_RL = False

# AFTER
ENABLE_RL = os.getenv("ENABLE_RL", "false").lower() == "true"
```

**Add to `.env`:**
```env
ENABLE_RL=true
RL_TRAINING_MODE=false  # true only during optimization weeks
```

---

### 2.2 Performance Drift Detection
**Effort:** 6 hours | **Impact:** MEDIUM (enables proactive monitoring)

**Where:** Create `app/decision/drift_detector.py`

**What:** Track model performance degradation across rolling windows.

**Code stub:**
```python
import json
from datetime import datetime, timedelta
from app.config.config import Config
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)

class PerformanceDriftDetector:
    """Detects when model performance degrades beyond threshold."""
    
    def __init__(self, lookback_days: int = 30):
        self.lookback_days = lookback_days
        self.drift_threshold = 0.15  # 15% performance drop triggers alert
    
    def check_drift(self, ticker: str, current_score: float) -> dict:
        """
        Compare current performance vs. historical average.
        Returns: {drifting: bool, drift_pct: float, alert_level: str}
        """
        historical_scores = self._load_historical_scores(ticker)
        
        if not historical_scores:
            return {"drifting": False, "drift_pct": 0.0, "alert_level": "NONE"}
        
        avg_score = sum(historical_scores) / len(historical_scores)
        drift_pct = (avg_score - current_score) / max(avg_score, 0.01)
        
        if drift_pct > self.drift_threshold:
            alert_level = "CRITICAL" if drift_pct > 0.3 else "WARNING"
            logger.warning(
                f"{ticker}: Performance drift detected! "
                f"Avg={avg_score:.3f}, Current={current_score:.3f}, "
                f"Drift={drift_pct:.1%} ({alert_level})"
            )
            return {
                "drifting": True,
                "drift_pct": drift_pct,
                "alert_level": alert_level
            }
        
        return {"drifting": False, "drift_pct": 0.0, "alert_level": "NONE"}
    
    def _load_historical_scores(self, ticker: str) -> list:
        """Load last N days of reliability scores."""
        # Placeholder: query from DB or JSON
        return []
```

**Integrate into `audit_builder.py`:**
```python
# Add to build_audit_summary()
drift_detector = PerformanceDriftDetector()
for ticker in tickers:
    drift_info = drift_detector.check_drift(ticker, reliability_score)
    audit["drift_alerts"].append({
        "ticker": ticker,
        **drift_info
    })
```

---

### 2.3 RL Strategy Selection Activation
**Effort:** 4 hours | **Impact:** MEDIUM (enables intelligent strategy switching)

**Where:** [rl_inference.py](app/models/rl_inference.py) + [main.py](main.py)

**Change in `main.py` `run_daily()`:**
```python
# OLD: Use fixed parameters
from app.optimization.genetic_optimizer import load_best_params

params = load_best_params(ticker)
...

# NEW: Let RL select strategy if enabled
if Config.ENABLE_RL:
    from app.models.rl_inference import RLModelEnsembleRunner
    runner = RLModelEnsembleRunner(ticker, df)
    decisions, rl_state = runner.run_ensemble(df)
else:
    # Fall back to genetic algorithm
    params = load_best_params(ticker)
    ...
```

---

## SPRINT 3 (Weeks 5–6): PORTFOLIO OPTIMIZATION

### 3.1 Implement Risk Parity Allocation
**Effort:** 8 hours | **Impact:** HIGH (reduces portfolio volatility)

**Where:** Create `app/decision/risk_parity.py`

**What:** Equal risk contribution across all tickers.

**Code stub:**
```python
import numpy as np
import pandas as pd
from app.config.config import Config
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)

class RiskParityAllocator:
    """Allocates capital such that each position contributes equally to risk."""
    
    def __init__(self, lookback_days: int = 60):
        self.lookback_days = lookback_days
    
    def allocate(self, decisions: list, price_history: dict) -> list:
        """
        decisions: [{ticker, action, confidence, ...}]
        price_history: {ticker: [prices]}
        
        Returns: decisions with allocation_amount, allocation_pct updated
        """
        # 1. Compute volatility for each ticker
        volatilities = {}
        for ticker, prices in price_history.items():
            returns = np.diff(prices) / prices[:-1]
            volatilities[ticker] = np.std(returns)
        
        # 2. Compute risk-parity weights (inverse volatility normalized)
        tradeable = [d for d in decisions if d["signal"] != "HOLD"]
        if not tradeable:
            return decisions
        
        inv_vol = np.array([1.0 / max(volatilities.get(d["ticker"], 0.1), 0.01) 
                           for d in tradeable])
        weights = inv_vol / inv_vol.sum()
        
        # 3. Allocate capital
        capital = Config.INITIAL_CAPITAL * 0.95  # Reserve 5% cash
        
        for idx, decision in enumerate(tradeable):
            allocation = capital * weights[idx]
            decision["allocation_amount"] = round(allocation, 2)
            decision["allocation_pct"] = round(weights[idx], 4)
            logger.info(
                f"{decision['ticker']}: {weights[idx]:.1%} "
                f"(vol={volatilities.get(decision['ticker'], 0):.3f})"
            )
        
        return decisions
```

---

### 3.2 Add Correlation Threshold Enforcement
**Effort:** 6 hours | **Impact:** MEDIUM (prevents concentrated bets)

**Where:** Update `app/decision/allocation.py`

**Add method:**
```python
def enforce_correlation_limits(
    decisions: list, 
    price_history: dict, 
    max_correlation: float = 0.7
) -> list:
    """
    Remove or reduce allocation for highly correlated assets.
    max_correlation: If two tickers >0.7 correlated, keep only higher-confidence one.
    """
    if len(decisions) < 2:
        return decisions
    
    # Compute correlation matrix
    prices_df = pd.DataFrame(price_history)
    corr_matrix = prices_df.corr()
    
    # Find correlated pairs
    high_corr_pairs = []
    for i, ticker1 in enumerate(corr_matrix.columns):
        for j, ticker2 in enumerate(corr_matrix.columns):
            if i < j and corr_matrix.iloc[i, j] > max_correlation:
                high_corr_pairs.append((ticker1, ticker2, corr_matrix.iloc[i, j]))
    
    # For each correlated pair, reduce the lower-confidence one
    for ticker1, ticker2, corr in high_corr_pairs:
        d1 = next((d for d in decisions if d["ticker"] == ticker1), None)
        d2 = next((d for d in decisions if d["ticker"] == ticker2), None)
        
        if d1 and d2:
            conf1 = d1.get("confidence", 0)
            conf2 = d2.get("confidence", 0)
            
            weaker = d2 if conf1 > conf2 else d1
            weaker["allocation_pct"] *= 0.5
            weaker["allocation_amount"] *= 0.5
            logger.warning(
                f"Correlation {ticker1}–{ticker2}: {corr:.2%}. "
                f"Reduced {weaker['ticker']} allocation."
            )
    
    return decisions
```

---

### 3.3 Rebalancing Rules
**Effort:** 6 hours | **Impact:** MEDIUM (maintains target allocation)

**Where:** Create `app/decision/rebalancer.py`

**Code stub:**
```python
class PortfolioRebalancer:
    """Triggers rebalancing when drift exceeds threshold."""
    
    def __init__(self, rebalance_threshold: float = 0.20):
        """rebalance_threshold: Drift % to trigger rebalancing (e.g., 20%)."""
        self.rebalance_threshold = rebalance_threshold
    
    def should_rebalance(self, 
                        current_weights: dict,  # {ticker: current_pct}
                        target_weights: dict    # {ticker: target_pct}
                        ) -> dict:
        """
        Compare current vs target weights.
        Returns: {should_rebalance: bool, drift_per_ticker: dict, drift_avg: float}
        """
        drifts = {}
        for ticker in target_weights:
            target = target_weights[ticker]
            current = current_weights.get(ticker, 0)
            drift = abs(current - target) / max(target, 0.01)
            drifts[ticker] = drift
        
        avg_drift = np.mean(list(drifts.values()))
        
        if avg_drift > self.rebalance_threshold:
            logger.info(
                f"Portfolio drift {avg_drift:.1%} > threshold "
                f"{self.rebalance_threshold:.1%}. Rebalancing triggered."
            )
            return {
                "should_rebalance": True,
                "drift_per_ticker": drifts,
                "drift_avg": avg_drift
            }
        
        return {
            "should_rebalance": False,
            "drift_per_ticker": drifts,
            "drift_avg": avg_drift
        }
```

---

## SPRINT 4 (Weeks 7–8): ADMIN DASHBOARD & MONITORING

### 4.1 Add Admin Dashboard Route
**Effort:** 6 hours | **Impact:** MEDIUM (enables manual monitoring)

**Where:** Add to [app/ui/app.py](app/ui/app.py)

**Code stub:**
```python
from flask import jsonify
from app.data_access.data_manager import DataManager
from app.backtesting.history_store import HistoryStore
from datetime import datetime, timedelta

@app.route("/admin/dashboard")
def admin_dashboard():
    """Admin-only dashboard with system health & recent decisions."""
    # Check auth (e.g., API key)
    key = request.headers.get("X-Admin-Key")
    if key != Config.ADMIN_API_KEY:
        return {"error": "Unauthorized"}, 401
    
    dm = DataManager()
    hs = HistoryStore()
    
    # System health
    today = datetime.today().strftime("%Y-%m-%d")
    recommendations = dm.get_today_recommendations()
    recent_history = hs.load_range(
        start_date=(datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d"),
        end_date=today
    )
    
    return jsonify({
        "status": "healthy" if recommendations else "no_recs",
        "date": today,
        "recommendations_today": len(recommendations),
        "recent_decisions": recent_history,
        "model_reliability_avg": dm.get_model_reliability_avg(today),
        "portfolio_drift": compute_portfolio_drift(),
        "last_update": dm.get_last_update_timestamp()
    })

@app.route("/admin/force-rebalance", methods=["POST"])
def force_rebalance():
    """Manual trigger for portfolio rebalancing."""
    key = request.headers.get("X-Admin-Key")
    if key != Config.ADMIN_API_KEY:
        return {"error": "Unauthorized"}, 401
    
    # Trigger rebalance logic
    result = trigger_rebalance_now()
    return jsonify({"status": "rebalance_initiated", **result})
```

---

### 4.2 Add System Monitoring Metrics
**Effort:** 4 hours | **Impact:** LOW–MEDIUM (enables alerting)

**Where:** Create `app/infrastructure/metrics.py`

**Code stub:**
```python
import json
from datetime import datetime
from app.config.config import Config
from pathlib import Path

class SystemMetrics:
    """Tracks system health metrics for monitoring."""
    
    def __init__(self):
        self.metrics_dir = Config.LOG_DIR / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)
    
    def log_pipeline_execution(self, 
                               ticker: str, 
                               status: str,  # "success", "error", "timeout"
                               duration_sec: float,
                               details: dict = None):
        """Log daily pipeline execution."""
        metric = {
            "timestamp": datetime.utcnow().isoformat(),
            "ticker": ticker,
            "status": status,
            "duration_sec": duration_sec,
            "details": details or {}
        }
        
        path = self.metrics_dir / f"{datetime.utcnow().strftime('%Y-%m')}.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(metric) + "\n")
    
    def get_recent_metrics(self, hours: int = 24) -> dict:
        """Get recent metrics for dashboard."""
        # Placeholder: load from JSONL and aggregate
        return {
            "success_rate": 0.98,
            "avg_duration_sec": 45.2,
            "errors_24h": 1,
            "last_success": "2026-01-21 10:30:00"
        }
```

---

### 4.3 Error Alerting System
**Effort:** 4 hours | **Impact:** HIGH (prevents silent failures)

**Where:** Create `app/notifications/alerter.py`

**Code stub:**
```python
from app.infrastructure.logger import setup_logger
from app.notifications.mailer import send_email
from app.config.config import Config

logger = setup_logger(__name__)

class ErrorAlerter:
    """Sends immediate alerts for critical failures."""
    
    CRITICAL_ERRORS = [
        "DB_CONNECTION_FAILED",
        "CRON_EXECUTION_FAILED",
        "DATA_DOWNLOAD_ERROR",
        "PIPELINE_TIMEOUT"
    ]
    
    @staticmethod
    def alert(error_code: str, message: str, details: dict = None):
        """Send alert if error is critical."""
        if error_code in ErrorAlerter.CRITICAL_ERRORS:
            logger.critical(f"ALERT: {error_code} — {message}")
            
            # Send email alert
            subject = f"🚨 {error_code}"
            body = f"{message}\n\nDetails: {details or {}}"
            send_email(
                to=Config.NOTIFY_EMAIL,
                subject=subject,
                body=body
            )
        else:
            logger.warning(f"{error_code} — {message}")

# Usage in main.py
try:
    run_daily()
except Exception as e:
    ErrorAlerter.alert("CRON_EXECUTION_FAILED", str(e), {"traceback": traceback.format_exc()})
    raise
```

---

## SPRINT 5: PRODUCTION DEPLOYMENT & UI IMPROVEMENTS

### Context
After completing all P0-P9 features (Sprints 1-4), Sprint 5 focuses on:
1. **Production deployment** (8h) — Native Synology NAS deployment (no Docker)
2. **UI enhancements** (4h) — React dashboard and PyFolio integration (bonus)

---

### 5.1 Production Deployment to Synology NAS (8 hours)
**Effort:** 8 hours | **Impact:** 🔴 CRITICAL (enables live trading)

**Why NOT Docker?**
- **Resource constraints**: NAS has limited CPU/memory; container overhead is wasteful
- **Complexity**: Docker adds debugging layer; direct execution simpler on NAS
- **Native integration**: Synology DSM natively supports systemd and cron scheduling
- **Performance**: No container cold-start delays for daily pipelines
- **Maintenance**: Fewer dependencies; simpler troubleshooting with native OS tools

**Architecture (systemd + timers instead of Docker):**

```
┌─────────────────────────────────────────────────────────┐
│           Synology NAS / Linux Server                   │
├─────────────────────────────────────────────────────────┤
│  /volume1/tozsde_webapp/                                │
│  ├── app/                      (application code)       │
│  ├── tests/                    (test suite)             │
│  ├── venv/                     (Python 3.8+ venv)       │
│  ├── config/                   (settings)               │
│  └── logs/                     (JSONL metrics/errors)   │
├─────────────────────────────────────────────────────────┤
│           systemd Services (4 services)                 │
│                                                          │
│  tozsde-api.service     → Flask API on port 5000        │
│  tozsde-daily.service   → Daily 6 AM market pipeline    │
│  tozsde-optimize.service → Quarterly GA optimization    │
│  tozsde-reliability.service → Weekly backtest audit     │
├─────────────────────────────────────────────────────────┤
│           systemd Timers (3 timers)                     │
│                                                          │
│  tozsde-daily.timer     → Daily @ 6:00 AM              │
│  tozsde-quarterly.timer → 1st of month @ 1:00 AM       │
│  tozsde-weekly.timer    → Monday @ 4:00 AM             │
├─────────────────────────────────────────────────────────┤
│           Monitoring & Logging                          │
│                                                          │
│  /var/log/tozsde/       → Centralized logging           │
│  health_check.sh        → 5-min liveness probes         │
│  metrics.jsonl          → System health metrics         │
│  logrotate rules        → Automatic log rotation        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Task 5.1a: Pre-Deployment Checklist (1h)**
- [ ] Review [P9_PRODUCTION_DEPLOYMENT.md](P9_PRODUCTION_DEPLOYMENT.md) completely
- [ ] Verify target NAS/server: SSH access, Python 3.8+, sudo permissions
- [ ] Plan maintenance window during off-market hours
- [ ] Backup existing configuration and data
- [ ] Create dedicated `tozsde` system user on NAS

**Task 5.1b: Create Application Directory (1h)**
```bash
# SSH into NAS
ssh user@nas.local

# Create app directory with proper permissions
sudo mkdir -p /volume1/tozsde_webapp/{logs,config,data}
sudo chown tozsde:tozsde /volume1/tozsde_webapp -R

# Clone application
cd /volume1/tozsde_webapp
git clone <repository> .
```

**Task 5.1c: Setup Python Virtual Environment (1h)**
```bash
# Create isolated Python environment
cd /volume1/tozsde_webapp
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Test import
python -c "import app; print('✓ Application imports successfully')"
```

**Task 5.1d: Create systemd Service Files (2h)**

Create 4 service files in `/etc/systemd/system/`:

**tozsde-api.service** (Flask API - always-on)
```ini
[Unit]
Description=ToZsDE Trading API
After=network.target

[Service]
Type=notify
User=tozsde
WorkingDirectory=/volume1/tozsde_webapp
Environment="PATH=/volume1/tozsde_webapp/venv/bin"
ExecStart=/volume1/tozsde_webapp/venv/bin/gunicorn \
  --workers 2 \
  --bind 0.0.0.0:5000 \
  --timeout 120 \
  app.ui.app:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**tozsde-daily.service** (Daily 6 AM pipeline)
```ini
[Unit]
Description=ToZsDE Daily Market Pipeline
After=network.target

[Service]
Type=simple
User=tozsde
WorkingDirectory=/volume1/tozsde_webapp
Environment="PATH=/volume1/tozsde_webapp/venv/bin"
ExecStart=/volume1/tozsde_webapp/venv/bin/python -m app.daily_pipeline
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

**tozsde-optimize.service** (Quarterly optimization)
```ini
[Unit]
Description=ToZsDE Quarterly GA Optimization
After=network.target

[Service]
Type=simple
User=tozsde
WorkingDirectory=/volume1/tozsde_webapp
Environment="PATH=/volume1/tozsde_webapp/venv/bin"
ExecStart=/volume1/tozsde_webapp/venv/bin/python -m app.optimization.runner
StandardOutput=journal
StandardError=journal
TimeoutStartSec=7200

[Install]
WantedBy=multi-user.target
```

**tozsde-reliability.service** (Weekly backtest audit)
```ini
[Unit]
Description=ToZsDE Weekly Reliability Audit
After=network.target

[Service]
Type=simple
User=tozsde
WorkingDirectory=/volume1/tozsde_webapp
Environment="PATH=/volume1/tozsde_webapp/venv/bin"
ExecStart=/volume1/tozsde_webapp/venv/bin/python -m app.backtesting.audit_runner
StandardOutput=journal
StandardError=journal
TimeoutStartSec=3600

[Install]
WantedBy=multi-user.target
```

**Install services:**
```bash
sudo cp /volume1/tozsde_webapp/config/systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable tozsde-*.service
sudo systemctl start tozsde-api.service
```

**Task 5.1e: Create systemd Timer Files (1.5h)**

Create 3 timer files in `/etc/systemd/system/`:

**tozsde-daily.timer** (Runs daily service at 6:00 AM)
```ini
[Unit]
Description=Daily ToZsDE Market Pipeline Timer
Requires=tozsde-daily.service

[Timer]
OnCalendar=*-*-* 06:00:00
Persistent=true
OnBootSec=5min

[Install]
WantedBy=timers.target
```

**tozsde-quarterly.timer** (Runs optimize service 1st of month at 1:00 AM)
```ini
[Unit]
Description=Quarterly ToZsDE GA Optimization Timer
Requires=tozsde-optimize.service

[Timer]
OnCalendar=*-*-01 01:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

**tozsde-weekly.timer** (Runs reliability service every Monday at 4:00 AM)
```ini
[Unit]
Description=Weekly ToZsDE Reliability Audit Timer
Requires=tozsde-reliability.service

[Timer]
OnCalendar=Mon *-*-* 04:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

**Install and enable timers:**
```bash
sudo cp /volume1/tozsde_webapp/config/systemd/*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable tozsde-*.timer
sudo systemctl start tozsde-*.timer

# Verify timers
sudo systemctl list-timers tozsde-*
```

**Task 5.1f: Setup Logging & Monitoring (1h)**

**Configure log rotation** (`/etc/logrotate.d/tozsde`):
```
/volume1/tozsde_webapp/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 tozsde tozsde
}
```

**Create health check script** (`/volume1/tozsde_webapp/scripts/health_check.sh`):
```bash
#!/bin/bash
# Run every 5 minutes via cron

HEALTH_URL="http://localhost:5000/api/health"
TIMEOUT=10

# Check API health
if ! timeout $TIMEOUT curl -f "$HEALTH_URL" > /dev/null 2>&1; then
    echo "❌ API health check failed"
    systemctl restart tozsde-api.service
    exit 1
fi

# Check disk space
DISK_USAGE=$(df /volume1 | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 85 ]; then
    echo "⚠️ Disk usage critical: ${DISK_USAGE}%"
    exit 1
fi

echo "✓ System healthy"
exit 0
```

**Add health check to crontab:**
```bash
# Edit crontab
sudo crontab -e

# Add this line (runs every 5 minutes)
*/5 * * * * /volume1/tozsde_webapp/scripts/health_check.sh
```

**Task 5.1g: Testing & Validation (0.5h)**

```bash
# Test API
curl http://nas.local:5000/api/health
# Expected: 200 OK with health status

# Test daily service (manual run)
sudo systemctl start tozsde-daily.service
sleep 5
sudo journalctl -u tozsde-daily.service -n 30

# Verify timers
sudo systemctl list-timers --all
# Expected: Three tozsde-* timers listed with next scheduled times

# Test logging
tail -f /volume1/tozsde_webapp/logs/*.log

# Check systemd journal
sudo journalctl -u tozsde-* -n 100 --follow
```

**Troubleshooting:**

| Issue | Solution |
|-------|----------|
| Service won't start | `journalctl -u tozsde-api.service -n 100` |
| Timer not running | `systemctl is-enabled tozsde-daily.timer` |
| Import errors | `cd /volume1/tozsde_webapp && source venv/bin/activate && python -m app` |
| Port 5000 in use | `sudo lsof -i :5000` then `systemctl restart tozsde-api.service` |

**Reference:** [P9_PRODUCTION_DEPLOYMENT.md](P9_PRODUCTION_DEPLOYMENT.md)

---

### 5.2 UI/UX Enhancements (4 hours - bonus)
**Effort:** 4 hours | **Impact:** 🟢 LOW (optional nice-to-have)

#### Feature 5.2a: React Dashboard (2h)
```bash
# Migrate to React 18 with real-time updates
npx create-react-app app/ui/frontend
npm install chart.js react-chartjs-2 axios

# Build:
# - Real-time recommendation display
# - Portfolio visualization (Plotly)
# - Parameter tuning interface
```

#### Feature 5.2b: PyFolio Advanced Reporting (2h)
**Where:** `app/reporting/pyfolio_report.py`

```python
import pyfolio as pf
import pandas as pd

def generate_pyfolio_report(returns_series: pd.Series, 
                           positions: pd.DataFrame) -> dict:
    """
    Generate comprehensive performance analysis using PyFolio.
    
    Args:
        returns_series: Daily returns pd.Series
        positions: Position history DataFrame
    
    Returns:
        Dictionary with {sharpe_ratio, max_drawdown, calmar_ratio, ...}
    """
    try:
        rolling_sharpe = pf.stats.rolling_sharpe(returns_series, rolling_window=252)
        rolling_max_dd = pf.stats.rolling_max_drawdown(returns_series, rolling_window=252)
        
        return {
            "rolling_sharpe": rolling_sharpe.to_dict(),
            "rolling_max_dd": rolling_max_dd.to_dict(),
            "calmar_ratio": pf.stats.calmar_ratio(returns_series),
            "stability": pf.stats.stability_of_timeseries(returns_series)
        }
    except Exception as e:
        logger.warning(f"PyFolio report failed: {e}")
        return {}
```

---

## EFFORT SUMMARY

| Sprint | Focus | Duration | Effort | Impact |
|--------|-------|----------|--------|--------|
| **1** | Tests & QA | 2 weeks | 22h | 🔴 Critical |
| **2** | RL & Drift Detection | 2 weeks | 13h | 🔴 High |
| **3** | Portfolio Optimization | 2 weeks | 20h | 🟡 Medium |
| **4** | Admin & Monitoring | 2 weeks | 14h | 🟡 Medium |
| **5** | Production Deployment | 1 week | 8h | 🔴 Critical |
| **5+** | UI & Analysis (Bonus) | — | 12h | 🟢 Low |
| | **TOTAL** | **9 weeks** | **~89 hours** | |

---

## IMPLEMENTATION PRIORITY

### **🔴 DO FIRST (Weeks 1–4)**
1. **Unit tests** (Sprint 1) — Unblocks safe refactoring
2. **Enable RL** (Sprint 2) — Unlocks learning system
3. **Drift detection** (Sprint 2) — Proactive risk management

### **🟡 DO SECOND (Weeks 5–6)**
4. **Risk parity** (Sprint 3) — Reduces portfolio risk
5. **Correlation limits** (Sprint 3) — Prevents concentration

### **🟢 DO LAST (Weeks 7–8)**
6. **Admin dashboard** (Sprint 4) — Manual control
7. **Error alerting** (Sprint 4) — Visibility
8. **Interactive UI** (Sprint 5) — Polish (optional)

---

## QUICK START CHECKLIST

```bash
# Week 1 — Set up tests
mkdir tests
touch tests/__init__.py tests/conftest.py tests/test_indicators.py
# ... implement fixtures and first test

# Week 2 — Test core modules
# ... implement backtester, fitness, walk-forward tests

# Week 3 — Enable RL
# Update CONFIG: ENABLE_RL = true
# Start training agents on 2-year history

# Week 4 — Add drift detection
# Create drift_detector.py
# Integrate into audit_builder.py

# Weeks 5–6 — Portfolio features
# Implement risk_parity.py, correlation limits, rebalancing

# Weeks 7–8 — Admin features
# Add /admin/dashboard route, metrics tracking, error alerts
```

---

## SUCCESS CRITERIA

- [ ] 100+ tests passing (P9: 50% → 90%)
- [ ] RL module enabled and generating strategy selections
- [ ] Performance drift alerts sent when models degrade >15%
- [ ] Risk parity allocations reducing portfolio volatility by 10%+
- [ ] Admin dashboard showing system health in real-time
- [ ] Zero silent failures in daily pipeline (100% exception logging)
- [ ] Production readiness: **P0–P9 average 90%+**

---

**Next:** Pick Sprint 1 tasks and start with test infrastructure setup.  
**Questions?** Review specific sprint details above for code stubs and integration points.
