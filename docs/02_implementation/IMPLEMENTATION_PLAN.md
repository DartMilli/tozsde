# IMPLEMENTATION PLAN (Q1 2026)
**Priority:** Production deployment & operational excellence  
**Timeline:** 9 weeks of development

**STATUS:** ✅ **SPRINTS 1-4 COMPLETE** (139/139 tests passing) | **SPRINT 5 READY**

---

## 🎉 Completion Summary

| Sprint | Component | Tests | Status | Date |
|--------|-----------|-------|--------|------|
| 1 | Core Infrastructure | 63 | ✅ COMPLETE | Jan 22, 2026 |
| 2 | Enhanced Decision Making | Integrated | ✅ COMPLETE | Jan 22, 2026 |
| 3 | Portfolio Optimization | 51 | ✅ COMPLETE | Jan 22, 2026 |
| 4 | Hardening & Monitoring | 25 | ✅ COMPLETE | Jan 23, 2026 |
| **TOTAL** | **Test Suite** | **139** | **✅ PASSING (100%)** | **Jan 23, 2026** |
| **NEXT** | **Production Deployment** | — | 🔄 PLANNED | — |

---

## SPRINT 1 (Weeks 1–2): TEST SUITE & QUALITY ASSURANCE ✅ COMPLETE

**Status:** ✅ **COMPLETE** - 63/63 tests passing

**Achievements:**
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

## SPRINT 2 (Weeks 3–4): LEARNING SYSTEM & RL ACTIVATION ✅ COMPLETE

**Status:** ✅ **COMPLETE** - Integrated without adding new test files

**Achievements:**
- ✅ 2.1: RL Module Safely Enabled (environment-driven configuration)
- ✅ 2.2: Performance Drift Detection (SQLite-based, 15%/30% thresholds)
- ✅ 2.3: RL Strategy Selection (voting ensemble with fallback)

**Key Implementation Details:**
- Modified [config.py](app/config/config.py): `ENABLE_RL` and `RL_TRAINING_MODE` environment variables
- Added `.env`: Safe defaults (`ENABLE_RL=false`, `RL_TRAINING_MODE=false`)
- Created [drift_detector.py](app/decision/drift_detector.py): SQLite-based drift detection
- Modified [audit_builder.py](app/reporting/audit_builder.py): Added `drift_status` field
- Modified [main.py](main.py): RL consultation block with graceful fallback

### 2.1 Enable RL Module Safely ✅
**Effort:** 3 hours | **Status:** COMPLETE

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

## SPRINT 3 (Weeks 5–6): PORTFOLIO OPTIMIZATION ✅ COMPLETE

**Status:** ✅ **COMPLETE** - 51/51 tests passing (114/114 cumulative)

**Achievements:**
- ✅ 3.1: Risk Parity Allocation (13 tests)
- ✅ 3.2: Correlation Threshold Enforcement (10 tests)
- ✅ 3.3: Rebalancing Rules (28 tests)

**Documentation:** [SPRINT3_COMPLETION.md](../03_testing/SPRINT3_COMPLETION.md)

### 3.1 Implement Risk Parity Allocation ✅ COMPLETE
**Effort:** 8 hours | **Status:** COMPLETE | **Tests:** 13/13 PASSING

**Where:** [app/decision/risk_parity.py](app/decision/risk_parity.py) (181 lines)

**Implementation:** Inverse-volatility weighting
- Lower volatility → Higher allocation
- Annualized volatility (252 trading days)
- 1% volatility floor to prevent extreme allocations

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

### 3.2 Add Correlation Threshold Enforcement ✅ COMPLETE
**Effort:** 6 hours | **Status:** COMPLETE | **Tests:** 10/10 PASSING

**Where:** [app/decision/allocation.py](app/decision/allocation.py) - Added `enforce_correlation_limits()` (85 lines)

**Implementation:** Correlation-based allocation reduction
- Detects correlated pairs (>0.7 correlation threshold)
- Reduces lower-confidence asset by 50%
- Keeps higher-confidence asset at full allocation
- Metadata: `correlation_adjustment` flag

**Test Coverage:** [test_correlation_limits.py](../tests/test_correlation_limits.py) (10 tests)
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

### 3.3 Rebalancing Rules ✅ COMPLETE
**Effort:** 6 hours | **Status:** COMPLETE | **Tests:** 28/28 PASSING

**Where:** [app/decision/rebalancer.py](app/decision/rebalancer.py) (240 lines)

**Test Coverage:** [test_rebalancer.py](../tests/test_rebalancer.py) (28 tests)

---

## SPRINT 4 (Weeks 7–8): ADMIN DASHBOARD & MONITORING ⏳ PENDING
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

**Storage:** SQLite-based metrics (same database as recommendations)

**Code stub:**
```python
from datetime import datetime
from app.config.config import Config
from app.data_access.data_manager import DataManager

class SystemMetrics:
    """Tracks system health metrics in SQLite for monitoring."""
    
    def __init__(self):
        self.dm = DataManager()
        self._ensure_metrics_table()
    
    def _ensure_metrics_table(self):
        """Create metrics table if not exists."""
        with self.dm.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    ticker TEXT NOT NULL,
                    status TEXT NOT NULL,
                    duration_sec REAL,
                    error_message TEXT,
                    execution_date DATE
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_date ON pipeline_metrics(execution_date)")
            conn.commit()
    
    def log_pipeline_execution(self, 
                               ticker: str, 
                               status: str,  # "success", "error", "timeout"
                               duration_sec: float,
                               error_message: str = None):
        """Log daily pipeline execution to database."""
        with self.dm.get_connection() as conn:
            conn.execute("""
                INSERT INTO pipeline_metrics 
                (ticker, status, duration_sec, error_message, execution_date)
                VALUES (?, ?, ?, ?, DATE('now'))
            """, (ticker, status, duration_sec, error_message))
            conn.commit()
    
    def get_recent_metrics(self, hours: int = 24) -> dict:
        """Get recent metrics for dashboard."""
        with self.dm.get_connection() as conn:
            cursor = conn.execute("""
                SELECT status, COUNT(*) as count, AVG(duration_sec) as avg_duration
                FROM pipeline_metrics
                WHERE timestamp >= datetime('now', '-' || ? || ' hours')
                GROUP BY status
            """, (hours,))
            
            results = cursor.fetchall()
            total = sum(r[1] for r in results)
            success_count = next((r[1] for r in results if r[0] == "success"), 0)
            error_count = next((r[1] for r in results if r[0] == "error"), 0)
            avg_duration = next((r[2] for r in results if r[0] == "success"), 0) or 0
            
            success_rate = success_count / total if total > 0 else 0
            
            return {
                "success_rate": success_rate,
                "avg_duration_sec": avg_duration,
                "errors_24h": error_count,
                "total_executions": total
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

## SPRINT 4 (Weeks 7–8): HARDENING & MONITORING ✅ COMPLETE

**Status:** ✅ **COMPLETE** - 25/25 tests passing
**Completion Date:** Jan 23, 2026
**Total Project Tests:** 139/139 (100%)

### Components

#### 4.1 Admin Dashboard Routes (9 tests) ✅
**File:** `app/ui/app.py`

**Endpoints:**
- `GET /admin/dashboard` - Full system overview (health + recent decisions)
- `GET /admin/metrics` - Detailed metrics query (filterable by hours/date)
- `GET /admin/health` - Health status check
- `POST /admin/force-rebalance` - Manual rebalance trigger

**Authentication:** `X-Admin-Key` header validation, returns 401 for invalid keys

**Test Coverage:**
- 5 tests: Authentication enforcement (all 4 endpoints reject invalid keys)
- 4 tests: Route functionality (dashboard, metrics, health, rebalance)

#### 4.2 System Metrics - SQLite Based (16 tests) ✅
**Architecture:** SQL strictly isolated to DataManager only

**Files:**
- `app/infrastructure/metrics.py` (130 lines) - Pure delegation layer
- `app/data_access/data_manager.py` (extended) - SQL operations

**Methods:**
```python
log_pipeline_execution(ticker, status, duration_sec, error_message)
log_backtest_execution(ticker, wf_score, trades_count, profit_factor)
get_recent_metrics(hours)  # Last N hours aggregation
get_daily_summary(date)    # Date-specific statistics
get_health_status()        # System health snapshot
```

**Database Schema:**
```sql
CREATE TABLE pipeline_metrics (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    ticker TEXT NOT NULL,
    status TEXT NOT NULL,
    duration_sec REAL,
    error_message TEXT,
    execution_date DATE
)
CREATE INDEX idx_metrics_date ON pipeline_metrics(execution_date)
CREATE INDEX idx_metrics_ticker ON pipeline_metrics(ticker)
```

**Health Status:**
- Healthy: < 5% errors
- Degraded: 5-10% errors
- Critical: > 10% errors

**Test Coverage:**
- 2 tests: Database initialization & indexing
- 4 tests: Pipeline execution logging (success/error/multiple)
- 2 tests: Backtest result logging
- 5 tests: Metrics aggregation & calculations
- 3 tests: Daily summaries & ticker lists
- 3 tests: Health status classification (healthy/degraded/critical)

**Key Pattern - Isolated Test Databases:**
```python
@pytest.fixture
def fresh_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        dm = DataManager(db_path=Path(tmpdir) / "test.db")
        dm.initialize_tables()
        yield dm
    # Cleanup automatic
```

#### 4.3 Error Alerting (pre-validated) ✅
**File:** `app/notifications/alerter.py` (262 lines)

**Functionality:**
- Error severity classification (CRITICAL, WARNING, INFO)
- Immediate email alerts for critical errors
- Comprehensive error logging

**Error Classifications:**
- CRITICAL: DB failures, execution failures, invalid states → Email alert
- WARNING: Data issues, recoverable errors → Log only
- INFO: Normal operations → Log only

### Key Achievements SPRINT 4

✅ **Architecture Decision: SQL Segregation**
- All SQL queries confined to `DataManager` only
- No SQL in `metrics.py` or `app.py`
- Clean separation of concerns
- Highly testable

✅ **Test Isolation Strategy**
- Function-scoped `fresh_db` fixtures
- Each test gets isolated database
- Zero cross-test contamination
- All 25 tests pass independently

✅ **Production Readiness**
- API authentication on all endpoints
- Health monitoring in place
- Metrics tracking operational
- Error alerting configured
- Zero regressions to previous sprints

### Test Files
- `tests/test_admin_dashboard.py` - 9 tests (routes & auth)
- `tests/test_metrics.py` - 16 tests (metrics & health)

### Result
✅ **SPRINT 4 COMPLETE** - 25/25 tests passing
✅ **PRODUCTION READY** - All deliverables tested & validated

---

## SPRINT 5: PRODUCTION DEPLOYMENT (Raspberry Pi 4/5)

### Context
After completing all P0-P9 features (Sprints 1-4), Sprint 5 focuses on:
1. **Production deployment to Raspberry Pi** (8h) — Complete setup from unboxing to live trading
2. **UI enhancements** (4h) — React dashboard and PyFolio integration (bonus)

---

### 5.1 Complete Raspberry Pi Deployment (8 hours)

**Effort:** 8 hours | **Impact:** 🔴 CRITICAL (enables live trading)

**Why Raspberry Pi?**
- **Efficient**: Low power consumption (5-27W vs NAS 30-100W)
- **Simple**: No Docker overhead, direct systemd + cron scheduling
- **Reliable**: Proven Linux ARM64 platform, rock-solid stability
- **Cost-effective**: €50-80 hardware investment
- **24/7 capable**: Can run indefinitely without thermal issues

**Supported Models:**
- Raspberry Pi 4B (2GB+ RAM recommended, 8GB ideal)
- Raspberry Pi 5 (8GB+ RAM recommended) — *Faster, more cores*
- OS: **Raspberry Pi OS Lite** (64-bit, Debian-based)

**Architecture (cron + systemd):**

```
┌──────────────────────────────────────────────────────────┐
│         Raspberry Pi 4/5 (64-bit ARM)                    │
├──────────────────────────────────────────────────────────┤
│  /home/pi/tozsde_webapp/                                 │
│  ├── app/                      (application code)        │
│  ├── tests/                    (test suite)              │
│  ├── venv/                     (Python 3.11+ venv)       │
│  ├── config/                   (settings, secrets)       │
│  ├── logs/                     (JSONL metrics/errors)    │
│  ├── scripts/                  (deploy, health checks)   │
│  └── deploy_rpi.sh             (ONE-CLICK INSTALLER!)   │
├──────────────────────────────────────────────────────────┤
│  systemd Services (Flask API)                            │
│                                                           │
│  tozsde-api.service → Flask API on port 5000             │
│                      (always-on, auto-restart)           │
├──────────────────────────────────────────────────────────┤
│  Cron Jobs (scheduled tasks)                             │
│                                                           │
│  6:00 AM   → Daily market pipeline                       │
│  1:00 AM   → 1st of month: GA optimization               │
│  4:00 AM   → Monday: Weekly backtest audit               │
├──────────────────────────────────────────────────────────┤
│  Monitoring                                              │
│                                                           │
│  health_check.sh (runs every 5 min)                      │
│  metrics.jsonl → System performance logs                 │
│  logrotate → Auto cleanup old logs                       │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

---

#### **5.1 - ONE-CLICK DEPLOYMENT SCRIPT**

**All setup in ONE file:** `deploy_rpi.sh` (self-contained, idempotent)

**Task 5.1a: Initial Hardware Setup (physical, ~15 min)**

1. **Unbox Raspberry Pi 4/5** + accessories:
   - Micro HDMI cable (2x for Pi 4, 1x for Pi 5)
   - USB-C power supply (27W minimum for Pi 5, 15W for Pi 4)
   - Micro SD card (at least 64GB, faster is better)
   - Optional: Heat sinks, fan for thermal management

2. **Flash OS to SD Card** (on any computer with USB reader):
   ```bash
   # Download Raspberry Pi Imager: https://www.raspberrypi.com/software/
   # Or use ddrescue/Balena Etcher
   
   # Option: Command line (macOS/Linux)
   unzip ~/Downloads/2024-01-15-raspios-bookworm-arm64-lite.img.zip
   diskutil list  # Find /dev/diskX
   sudo dd if=2024-01-15-raspios-bookworm-arm64-lite.img of=/dev/rdiskX bs=4m
   ```

3. **Boot Raspberry Pi:**
   - Insert flashed SD card
   - Connect HDMI, keyboard, mouse
   - Connect power
   - Wait ~30 seconds for initial boot

4. **Initial Configuration (one-time):**
   ```bash
   # Login: pi / raspberry
   
   # Expand filesystem to use full SD card
   sudo raspi-config
   # → Choose: Advanced Options → Expand Filesystem
   # → Reboot
   
   # Update system
   sudo apt-get update
   sudo apt-get upgrade -y
   
   # Enable SSH for remote access
   sudo raspi-config
   # → Interface Options → SSH → Yes
   # Note IP address: hostname -I
   ```

5. **From here on, everything is SSH remote or one script!**

---

**Task 5.1b: Copy Application & Run Deploy Script (automated)**

```bash
# FROM YOUR LAPTOP/DESKTOP:

# 1. SSH into Pi (first time, or use IP from hostname -I)
ssh pi@raspberrypi.local  # or ssh pi@192.168.x.x

# 2. Clone repo or upload files
git clone <your-repo> ~/tozsde_webapp
cd ~/tozsde_webapp

# 3. RUN ONE-LINE DEPLOY (does everything!)
bash deploy_rpi.sh

# That's it! Script will:
# ✓ Install system dependencies (Python 3.11, pip, cron, curl)
# ✓ Create Python venv
# ✓ Install requirements.txt
# ✓ Create systemd service (Flask API)
# ✓ Setup cron jobs (daily, weekly, monthly tasks)
# ✓ Configure health checks
# ✓ Setup log rotation
# ✓ Start services
# ✓ Verify everything works
```

---

**Task 5.1c: Deploy Script Contents**

Create this file: **`deploy_rpi.sh`**

```bash
#!/bin/bash
##############################################################################
# ToZsDE Raspberry Pi Automated Deployment Script
# One-click setup: System config → Python env → Services → Monitoring
# Usage: bash deploy_rpi.sh
##############################################################################

set -e  # Exit on error

APP_DIR="/home/pi/tozsde_webapp"
VENV_DIR="$APP_DIR/venv"
LOGS_DIR="$APP_DIR/logs"
USER="pi"
GROUP="pi"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_header() {
    echo -e "\n${YELLOW}=== $1 ===${NC}\n"
}

# ==============================================================================
# STEP 1: System Dependencies
# ==============================================================================
print_header "Installing System Dependencies"

sudo apt-get update
sudo apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    pip \
    git \
    curl \
    build-essential \
    libssl-dev \
    libffi-dev \
    logrotate

print_status "System dependencies installed"

# ==============================================================================
# STEP 2: Create Application Directory & Ownership
# ==============================================================================
print_header "Setting Up Application Directory"

if [ ! -d "$APP_DIR" ]; then
    print_error "App directory $APP_DIR not found!"
    exit 1
fi

mkdir -p "$LOGS_DIR"
mkdir -p "$APP_DIR/config/systemd"
mkdir -p "$APP_DIR/scripts"

# Set permissions
sudo chown -R $USER:$GROUP "$APP_DIR"
chmod 755 "$APP_DIR"
chmod 755 "$LOGS_DIR"

print_status "Application directory ready: $APP_DIR"

# ==============================================================================
# STEP 3: Python Virtual Environment
# ==============================================================================
print_header "Setting Up Python Virtual Environment"

if [ ! -d "$VENV_DIR" ]; then
    python3.11 -m venv "$VENV_DIR"
    print_status "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate venv for this script
source "$VENV_DIR/bin/activate"

# Upgrade pip, setuptools, wheel
pip install --upgrade pip setuptools wheel

# Install requirements
if [ -f "$APP_DIR/requirements.txt" ]; then
    pip install -r "$APP_DIR/requirements.txt"
    print_status "Python dependencies installed from requirements.txt"
else
    print_error "requirements.txt not found in $APP_DIR"
    exit 1
fi

# Test import
if python -c "import app; print('✓ App imports OK')" 2>/dev/null; then
    print_status "Application imports successfully"
else
    print_error "Application import failed"
    exit 1
fi

# ==============================================================================
# STEP 4: Create systemd Service (Flask API)
# ==============================================================================
print_header "Creating systemd Service"

cat > "$APP_DIR/config/systemd/tozsde-api.service" << 'EOF'
[Unit]
Description=ToZsDE Trading API
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pi
Group=pi
WorkingDirectory=/home/pi/tozsde_webapp
Environment="PATH=/home/pi/tozsde_webapp/venv/bin"
Environment="PYTHONUNBUFFERED=1"
ExecStart=/home/pi/tozsde_webapp/venv/bin/python -m app.ui.app
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
StandardOutputMaxMem=16M

[Install]
WantedBy=multi-user.target
EOF

# Install service
sudo cp "$APP_DIR/config/systemd/tozsde-api.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable tozsde-api.service

print_status "systemd service created and enabled"

# ==============================================================================
# STEP 5: Create Cron Jobs (Scheduled Tasks)
# ==============================================================================
print_header "Setting Up Cron Jobs"

# Create crontab entries (idempotent)
CRON_CMD="source /home/pi/tozsde_webapp/venv/bin/activate && cd /home/pi/tozsde_webapp && python -m app.daily_pipeline"

# Remove old cron entries if exist
(crontab -l 2>/dev/null | grep -v "tozsde_webapp" | grep -v "^$") | crontab - 2>/dev/null || true

# Add new cron entries
(crontab -l 2>/dev/null; echo "# ToZsDE Daily Pipeline - 6:00 AM every day") | crontab -
(crontab -l 2>/dev/null; echo "0 6 * * * $CRON_CMD >> /home/pi/tozsde_webapp/logs/cron_daily.log 2>&1") | crontab -

# Weekly audit (Monday 4:00 AM)
WEEKLY_CMD="source /home/pi/tozsde_webapp/venv/bin/activate && cd /home/pi/tozsde_webapp && python -m app.backtesting.audit_runner"
(crontab -l 2>/dev/null; echo "0 4 * * 1 $WEEKLY_CMD >> /home/pi/tozsde_webapp/logs/cron_weekly.log 2>&1") | crontab -

# Monthly optimization (1st of month, 1:00 AM)
MONTHLY_CMD="source /home/pi/tozsde_webapp/venv/bin/activate && cd /home/pi/tozsde_webapp && python -m app.optimization.runner"
(crontab -l 2>/dev/null; echo "0 1 1 * * $MONTHLY_CMD >> /home/pi/tozsde_webapp/logs/cron_monthly.log 2>&1") | crontab -

print_status "Cron jobs configured:"
echo "  • Daily pipeline: 6:00 AM"
echo "  • Weekly audit: Monday 4:00 AM"
echo "  • Monthly optimization: 1st of month 1:00 AM"

# ==============================================================================
# STEP 6: Create Health Check Script
# ==============================================================================
print_header "Creating Health Check Script"

cat > "$APP_DIR/scripts/health_check.sh" << 'EOF'
#!/bin/bash
# Health check for Flask API
# Add to crontab: */5 * * * * /home/pi/tozsde_webapp/scripts/health_check.sh

HEALTH_URL="http://localhost:5000/api/health"
TIMEOUT=5
LOG_FILE="/home/pi/tozsde_webapp/logs/health_check.log"

# Check if API is responding
if ! timeout $TIMEOUT curl -f -s "$HEALTH_URL" > /dev/null 2>&1; then
    echo "[$(date)] ❌ Health check FAILED - restarting service" >> "$LOG_FILE"
    systemctl restart tozsde-api.service
    sleep 5
    if timeout $TIMEOUT curl -f -s "$HEALTH_URL" > /dev/null 2>&1; then
        echo "[$(date)] ✓ Service recovered" >> "$LOG_FILE"
    else
        echo "[$(date)] ❌ Service still down after restart" >> "$LOG_FILE"
    fi
else
    echo "[$(date)] ✓ Health check OK" >> "$LOG_FILE"
fi

# Check disk space
DISK_USAGE=$(df -h /home | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 85 ]; then
    echo "[$(date)] ⚠️  Disk usage critical: ${DISK_USAGE}%" >> "$LOG_FILE"
fi
EOF

chmod +x "$APP_DIR/scripts/health_check.sh"

# Add health check to crontab (every 5 minutes)
(crontab -l 2>/dev/null; echo "*/5 * * * * /home/pi/tozsde_webapp/scripts/health_check.sh >> /dev/null 2>&1") | crontab -

print_status "Health check script created and scheduled (every 5 min)"

# ==============================================================================
# STEP 7: Setup Log Rotation
# ==============================================================================
print_header "Configuring Log Rotation"

cat > /tmp/tozsde_logrotate << 'EOF'
/home/pi/tozsde_webapp/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 pi pi
}
EOF

sudo mv /tmp/tozsde_logrotate /etc/logrotate.d/tozsde
sudo chmod 644 /etc/logrotate.d/tozsde

print_status "Log rotation configured (daily, keep 7 days)"

# ==============================================================================
# STEP 8: Start Services
# ==============================================================================
print_header "Starting Services"

sudo systemctl start tozsde-api.service
sleep 2

if sudo systemctl is-active --quiet tozsde-api.service; then
    print_status "Flask API service started successfully"
else
    print_error "Failed to start Flask API service"
    sudo journalctl -u tozsde-api.service -n 20
    exit 1
fi

# ==============================================================================
# STEP 9: Verification
# ==============================================================================
print_header "Verification & Testing"

# Wait for API to be ready
echo "Waiting for API to be ready..."
for i in {1..10}; do
    if curl -f -s http://localhost:5000/api/health > /dev/null 2>&1; then
        print_status "✓ API responding on port 5000"
        break
    fi
    if [ $i -eq 10 ]; then
        print_error "API not responding after 10 attempts"
        exit 1
    fi
    sleep 1
done

# Show cron jobs
echo -e "\n${YELLOW}Scheduled Cron Jobs:${NC}"
crontab -l | grep tozsde

# Show service status
echo -e "\n${YELLOW}systemd Service Status:${NC}"
sudo systemctl status tozsde-api.service --no-pager

# Show recent logs
echo -e "\n${YELLOW}Recent Logs:${NC}"
sudo journalctl -u tozsde-api.service -n 10 --no-pager

# ==============================================================================
# SUCCESS!
# ==============================================================================
print_header "DEPLOYMENT COMPLETE! 🎉"

echo "✓ System dependencies installed"
echo "✓ Python venv created & activated"
echo "✓ Application requirements installed"
echo "✓ Flask API service running (port 5000)"
echo "✓ Cron jobs scheduled (daily, weekly, monthly)"
echo "✓ Health checks active (every 5 min)"
echo "✓ Log rotation configured"
echo ""
echo "NEXT STEPS:"
echo "  1. Access Flask API: curl http://raspberrypi.local:5000/api/health"
echo "  2. View logs: sudo journalctl -u tozsde-api.service -f"
echo "  3. Check cron: crontab -l"
echo "  4. Verify timers: sudo systemctl list-timers"
echo ""
echo "To SSH again: ssh pi@raspberrypi.local"
echo "App directory: /home/pi/tozsde_webapp"
echo ""
```

---

**Task 5.1d: Run the Deployment (from Raspberry Pi)**

```bash
# SSH into Pi
ssh pi@raspberrypi.local

# Navigate to app directory
cd ~/tozsde_webapp

# Run deployment script (ONE LINE!)
bash deploy_rpi.sh

# Expected output:
# [✓] System dependencies installed
# [✓] Virtual environment created
# [✓] Python dependencies installed
# [✓] systemd service created and enabled
# [✓] Cron jobs configured
# [✓] Health check script created
# [✓] Log rotation configured
# [✓] Flask API service started successfully
# [✓] API responding on port 5000
# DEPLOYMENT COMPLETE! 🎉
```

---

**Task 5.1e: Verification & First Run**

```bash
# Check Flask API is running
curl http://raspberrypi.local:5000/api/health
# Expected: {"status": "healthy", ...}

# View live logs
sudo journalctl -u tozsde-api.service -f

# Check cron jobs
crontab -l

# Test manual cron execution
source ~/tozsde_webapp/venv/bin/activate
cd ~/tozsde_webapp
python -m app.daily_pipeline
# Check logs/cron_daily.log for output

# View disk space (important on Pi!)
df -h /home

# Check Pi temperature (should stay <80°C)
/opt/vc/bin/vcgencmd measure_temp
```

---

**Task 5.1f: Troubleshooting**

| Issue | Solution |
|-------|----------|
| **SSH won't connect** | Check: `hostname -I` on Pi, ping it, check firewall |
| **API won't start** | `sudo journalctl -u tozsde-api.service -n 50` |
| **Port 5000 in use** | `sudo lsof -i :5000`, kill it, restart service |
| **Cron not running** | `crontab -l`, check logs: `/home/pi/tozsde_webapp/logs/cron_*.log` |
| **High CPU/slow** | `top`, check if backtest running, consider reducing walk-forward periods |
| **Disk full (75%+)** | Remove old logs: `rm /home/pi/tozsde_webapp/logs/cron_*.log.gz` |
| **Network timeout** | Check Rpi network: `ping 8.8.8.8`, restart WiFi: `sudo systemctl restart networking` |

---

#### **Summary: Everything Automated**

✅ **One script does everything:**

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
- System dependencies installed
- Python venv created & activated
- Application requirements installed
- Flask API service running (port 5000)
- Cron jobs scheduled (daily, weekly, monthly)
- Health checks active (every 5 min)
- Log rotation configured

**That's it! System is now LIVE. 🚀**

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
