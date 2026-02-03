# Bug Fix & Test Coverage Improvement Plan

**Created:** 2026-02-02  
**Phase:** Sprint 10 Complete  
**Scope:** Code quality enhancement  
**Target:** 83% coverage achieved, zero critical issues

---

## 📋 Executive Summary

Sprint 10 is complete. Test coverage increased from 59% to **83%** with **625 passing tests** and **1 skipped integration test**, while maintaining code stability.

**Actual outcome:** 59% → 83% coverage, +274 tests, 0 critical issues.

---

## 🐛 Known Issues - Prioritized Repair Plan

### Priority 1: CRITICAL (Must fix before hardware deployment)

#### Issue #1: AdminDashboard Blueprint Not Registered
**Status:** ✅ FIXED (2026-02-02)

**What was done:**
- Added blueprint import to `app/ui/app.py`
- Registered blueprint: `app.register_blueprint(admin_bp)`
- All 12 endpoints now accessible

**Verification:**
```bash
curl http://localhost:5000/admin/health
# Should return: {"status": "healthy", "timestamp": "..."}
```

---

### Priority 2: HIGH (Breaks some tests, but not production)

#### Issue #2: Portfolio Correlation Manager - 6 Failing Tests
**File:** `tests/test_portfolio_correlation_manager.py`  
**Failing tests:** 6/8  
**Root cause:** Tests expect real correlation matrix data in test database

**Problem Analysis:**
```python
# tests/test_portfolio_correlation_manager.py
def test_calculate_diversification_score_multi_asset(test_db):
    # test_db fixture doesn't have correlation data
    # Returns empty/zero correlations
    # Assertions fail because diversification score is 0
```

**Solution Plan:**

**Step 1: Create test fixtures with mock correlation data**
```python
# tests/fixtures/correlation_data.py
@pytest.fixture
def sample_correlation_matrix():
    """Create realistic correlation matrix for testing."""
    return {
        ('AAPL', 'MSFT'): 0.65,  # Positive correlation
        ('AAPL', 'GOOGL'): 0.58,
        ('MSFT', 'GOOGL'): 0.72,
        ('AAPL', 'GOLD'): -0.15,  # Negative correlation
        ('MSFT', 'GOLD'): -0.20,
        ('GOOGL', 'GOLD'): -0.10,
    }

@pytest.fixture
def populated_test_db(sample_correlation_matrix):
    """Create test database with correlation data."""
    db = sqlite3.connect(':memory:')
    cursor = db.cursor()
    
    # Create correlation table
    cursor.execute('''
        CREATE TABLE correlations (
            ticker1 TEXT,
            ticker2 TEXT,
            correlation REAL,
            date TEXT
        )
    ''')
    
    # Insert sample data
    for (t1, t2), corr in sample_correlation_matrix.items():
        cursor.execute(
            'INSERT INTO correlations VALUES (?, ?, ?, ?)',
            (t1, t2, corr, '2026-01-01')
        )
    
    db.commit()
    return db
```

**Step 2: Update failing tests to use new fixture**
```python
def test_calculate_diversification_score_multi_asset(populated_test_db):
    manager = PortfolioCorrelationManager(populated_test_db)
    
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    score = manager.calculate_diversification_score(tickers)
    
    assert 0.5 <= score <= 1.0  # Realistic range
```

**Step 3: Run and verify all 8 tests pass**
```bash
pytest tests/test_portfolio_correlation_manager.py -v
# Expected: 8/8 passing
```

**Effort estimate:** 3-4 hours  
**Impact:** +8 passing tests, -6 failures

---

#### Issue #3: Deprecation Warnings (pandas Series dtype)
**File:** `app/backtesting/backtester.py:68`  
**Warning:** `DeprecationWarning: The default dtype for empty Series will be 'object'...`

**Problem:**
```python
# Current (line 68)
"equity_curve": pd.Series(),  # dtype not specified

# In future pandas version, will default to 'object' instead of 'float64'
```

**Solution:**
```python
# Fixed version
"equity_curve": pd.Series(dtype=float),  # Explicit dtype
```

**Implementation:**
```bash
# Search and replace
grep -n "pd.Series()" app/backtesting/backtester.py
# Replace with: pd.Series(dtype=float)
```

**Verification:**
```bash
pytest tests/test_backtester.py -W error::DeprecationWarning
# Should pass with no warnings
```

**Effort estimate:** 30 minutes  
**Impact:** Eliminates all 10 deprecation warnings

---

#### Issue #4: Error in test_sprint9_modules.py (if any)
**Status:** VERIFY

Before coverage improvements, ensure Sprint 9 tests still all pass:
```bash
pytest tests/test_sprint9_modules.py -v
# Expected: 17/17 passing
```

**Effort estimate:** 15 minutes (if no issues found)

---

### Priority 3: MEDIUM (Coverage issues, but core features work)

#### Issue #5: Low Coverage on Critical Modules
**Scope:** 5 modules with < 20% coverage

| Module | Current | Target | Gap |
|--------|---------|--------|-----|
| plotter.py | 11% | 40% | +29% |
| backtest_audit.py | 6% | 30% | +24% |
| genetic_optimizer.py | 16% | 50% | +34% |
| data_loader.py | 16% | 50% | +34% |
| data_cleaner.py | 18% | 40% | +22% |

**Note:** These are visualization and advanced features. Lower priority than core logic.

---

## 📈 Test Coverage Improvement Strategy

### Phase 1: Foundation (Weeks 1-2) - Target 65%

#### Goal: Increase coverage on core decision & backtesting modules

**Module 1: AdminDashboard (13% → 40%)**

Current gaps:
- No tests for Flask endpoints
- No validation of response formats
- No error scenarios tested

**Implementation:**
```python
# tests/test_admin_dashboard_integration.py

import pytest
from app.ui.app import app
from flask import json

@pytest.fixture
def client():
    """Flask test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    """Test /admin/health endpoint."""
    response = client.get('/admin/health')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'status' in data
    assert data['status'] in ['healthy', 'warning', 'critical']
    assert 'timestamp' in data

def test_performance_summary_no_data(client):
    """Test /admin/performance/summary with no data."""
    response = client.get('/admin/performance/summary?days=30')
    
    # Should return 200 even with no data
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, dict)

def test_performance_summary_with_params(client):
    """Test /admin/performance/summary with different day ranges."""
    for days in [7, 30, 90, 365]:
        response = client.get(f'/admin/performance/summary?days={days}')
        assert response.status_code == 200
        data = json.loads(response.data)
        # Validate response structure
        assert 'period_days' in data or 'total_return' in data

def test_errors_summary(client):
    """Test /admin/errors/summary endpoint."""
    response = client.get('/admin/errors/summary')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'error_count' in data or 'summary' in data

def test_capital_status(client):
    """Test /admin/capital/status endpoint."""
    response = client.get('/admin/capital/status')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, dict)

def test_error_export_post(client):
    """Test POST /admin/errors/export."""
    response = client.post(
        '/admin/errors/export',
        json={'format': 'csv'}
    )
    
    # Should accept the request
    assert response.status_code in [200, 400]  # Success or validation error

def test_invalid_days_parameter(client):
    """Test with invalid day parameter."""
    response = client.get('/admin/performance/summary?days=invalid')
    
    # Should handle gracefully
    assert response.status_code in [200, 400]
```

**Tests to create:** 8-10  
**Effort:** 4-5 hours  
**Expected coverage:** 40%

---

**Module 2: PerformanceAnalytics (46% → 65%)**

Current gaps:
- Sharpe ratio edge cases
- Drawdown calculations
- Sortino ratio
- Custom period calculations

**Implementation:**
```python
# tests/test_performance_analytics_extended.py

import pytest
import pandas as pd
import numpy as np
from app.reporting.performance_analytics import PerformanceAnalytics

@pytest.fixture
def simple_returns():
    """Simple return series for testing."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252)  # 1 year
    returns = np.random.normal(0.0005, 0.01, 252)  # Daily returns
    return pd.Series(returns, index=dates)

@pytest.fixture
def equity_curve():
    """Equity curve from $10,000 start."""
    simple_ret = np.array([1.01, 0.99, 1.02, 0.98, 1.03])
    equity = 10000 * np.cumprod(simple_ret)
    return pd.Series(equity, index=pd.date_range('2023-01-01', periods=5))

def test_sharpe_ratio_calculation(simple_returns):
    """Test Sharpe ratio calculation."""
    analytics = PerformanceAnalytics(simple_returns)
    
    sharpe = analytics.calculate_sharpe_ratio()
    
    assert isinstance(sharpe, (int, float))
    assert sharpe > -10 and sharpe < 10  # Reasonable range

def test_sharpe_ratio_zero_returns():
    """Test Sharpe with zero returns."""
    returns = pd.Series([0.0] * 100)
    analytics = PerformanceAnalytics(returns)
    
    sharpe = analytics.calculate_sharpe_ratio()
    
    assert sharpe == 0 or np.isnan(sharpe)

def test_sortino_ratio(simple_returns):
    """Test Sortino ratio calculation."""
    analytics = PerformanceAnalytics(simple_returns)
    
    sortino = analytics.calculate_sortino_ratio()
    
    assert isinstance(sortino, (int, float))
    # Sortino should be >= Sharpe (uses only downside deviation)
    sharpe = analytics.calculate_sharpe_ratio()
    if not np.isnan(sharpe) and not np.isnan(sortino):
        assert sortino >= sharpe - 0.1  # Allow small tolerance

def test_max_drawdown(equity_curve):
    """Test max drawdown calculation."""
    analytics = PerformanceAnalytics(equity_curve.pct_change())
    
    max_dd = analytics.calculate_max_drawdown(equity_curve)
    
    assert -1.0 <= max_dd <= 0.0  # Should be negative or zero

def test_win_rate(simple_returns):
    """Test win rate calculation."""
    analytics = PerformanceAnalytics(simple_returns)
    
    win_rate = analytics.calculate_win_rate()
    
    assert 0 <= win_rate <= 1.0

def test_profit_factor(simple_returns):
    """Test profit factor calculation."""
    analytics = PerformanceAnalytics(simple_returns)
    
    profit_factor = analytics.calculate_profit_factor()
    
    assert profit_factor > 0  # Should be positive
    assert profit_factor < 100  # Reasonable upper bound

def test_custom_period_metrics(simple_returns):
    """Test metrics for custom periods."""
    analytics = PerformanceAnalytics(simple_returns)
    
    # Monthly metrics
    monthly_ret = simple_returns.resample('M').sum()
    
    for metric_func in [analytics.calculate_sharpe_ratio,
                       analytics.calculate_win_rate]:
        result = metric_func()
        assert isinstance(result, (int, float, np.number))
```

**Tests to create:** 8-10  
**Effort:** 4-5 hours  
**Expected coverage:** 65%

---

#### Module 3: DecisionEngine (45% → 70%)

Current gaps:
- Signal generation edge cases
- Market condition handling
- Allocation logic
- Error handling

**Implementation:**
```python
# tests/test_decision_engine_extended.py

def test_generate_signals_bull_market():
    """Test signal generation in bull market."""
    engine = DecisionEngine()
    indicators = {...}  # Bull market indicators
    
    signal = engine.generate_signals(indicators)
    
    assert signal in ['BUY', 'HOLD', 'SELL']

def test_generate_signals_bear_market():
    """Test signal generation in bear market."""
    engine = DecisionEngine()
    indicators = {...}  # Bear market indicators
    
    signal = engine.generate_signals(indicators)
    
    assert signal in ['BUY', 'HOLD', 'SELL']

def test_allocation_high_confidence():
    """Test allocation with high confidence signals."""
    engine = DecisionEngine()
    
    allocation = engine.calculate_allocation(confidence=0.9)
    
    assert sum(allocation.values()) == 1.0  # 100% allocated

def test_allocation_low_confidence():
    """Test allocation with low confidence."""
    engine = DecisionEngine()
    
    allocation = engine.calculate_allocation(confidence=0.3)
    
    # Should reduce exposure or hold cash
    assert sum(allocation.values()) <= 1.0

# ... more edge cases
```

**Tests to create:** 10-12  
**Effort:** 5-6 hours  
**Expected coverage:** 70%

---

### Phase 2: Enhancement (Weeks 3-4) - Target 75%

#### Goal: Improve medium-coverage modules and add missing tests

**Module 4: ErrorReporter (33% → 55%)**

Missing tests for:
- Export functionality (CSV, JSON)
- Severity filtering
- Date range queries
- Error statistics

**Implementation:** 6-8 tests, 3-4 hours

---

**Module 5: DataLoader (16% → 45%)**

Missing tests for:
- Yahoo Finance API calls (mock yfinance)
- Rate limiting handling
- Cache behavior
- Error scenarios

**Implementation:** 8-10 tests, 5-6 hours

---

**Module 6: GeneticOptimizer (16% → 45%)**

Missing tests for:
- Crossover operations
- Mutation operations
- Fitness calculations
- Convergence detection

**Implementation:** 10-12 tests, 6-8 hours

---

## 📊 Comprehensive Coverage Improvement Plan

### Baseline State (Sprint 9 end)
```
TOTAL: 351 passing tests, 59% coverage
│
├─ High (80%+): 10 modules ✅
├─ Medium (50-79%): 13 modules
└─ Low (<50%): 11 modules
```

### Actual State (Sprint 10 end)
```
TOTAL: 625 passing tests, 1 skipped, 83% coverage
│
├─ High (80%+): 20+ modules ✅
├─ Medium (50-79%): ~10 modules
└─ Low (<50%): 0 modules
```

### Detailed Coverage Improvement Table (Plan, historical)

| Module | Current | Phase 1 | Phase 2 | Target | Effort (hrs) |
|--------|---------|---------|---------|--------|-------------|
| admin_dashboard | 13% | 40% | 50% | 60% | 5 |
| performance_analytics | 46% | 60% | 70% | 75% | 5 |
| decision_engine | 45% | 65% | 75% | 80% | 6 |
| error_reporter | 33% | 40% | 55% | 60% | 4 |
| data_loader | 16% | 30% | 45% | 50% | 6 |
| genetic_optimizer | 16% | 30% | 45% | 50% | 8 |
| recommendation_builder | 32% | 50% | 65% | 70% | 5 |
| walk_forward | 34% | 50% | 65% | 70% | 4 |
| decision_builder | 41% | 55% | 70% | 75% | 4 |
| decision_logger | 40% | 55% | 70% | 75% | 5 |
| data_cleaner | 18% | 35% | 50% | 55% | 4 |
| plotter | 11% | 20% | 35% | 40% | 6 |
| backtest_audit | 6% | 15% | 30% | 35% | 5 |

**Total estimated effort:** 62 hours

### Effort Distribution by Week

**Week 1 (Issue fixes + Phase 1 start)**
- Fix portfolio correlation tests: 4 hours
- Fix deprecation warnings: 0.5 hours
- Start admin_dashboard tests: 3 hours
- Start performance_analytics tests: 3 hours
- **Total: 10.5 hours**

**Week 2 (Phase 1 completion)**
- Complete decision_engine tests: 6 hours
- Complete admin_dashboard tests: 2 hours
- Complete performance_analytics tests: 2 hours
- Bug fixes and integration: 2 hours
- **Total: 12 hours**

**Week 3 (Phase 2 start)**
- error_reporter tests: 4 hours
- data_loader tests: 6 hours
- recommendation_builder tests: 5 hours
- **Total: 15 hours**

**Week 4 (Phase 2 completion + polish)**
- genetic_optimizer tests: 8 hours
- walk_forward tests: 4 hours
- decision_builder tests: 4 hours
- Final verification & integration: 3 hours
- **Total: 19 hours**

---

## 🎯 Implementation Checklist

### Week 1: Foundation & Bug Fixes
- [ ] Fix portfolio_correlation tests (fixture-based)
- [ ] Fix pandas deprecation warnings
- [ ] Create admin_dashboard integration tests (5-8 tests)
- [ ] Create performance_analytics extended tests (5-8 tests)
- [ ] Run full test suite: `pytest tests/ -v`
- [ ] Measure coverage: `pytest --cov=app --cov-report=html`
- [ ] **Target: 351 → 365 tests, 59% → 62% coverage**

### Week 2: Phase 1 Completion
- [ ] Complete decision_engine tests (10-12 tests)
- [ ] Add error_reporter basic tests (3-4 tests)
- [ ] Add data_loader mock tests (4-5 tests)
- [ ] Integration testing
- [ ] Regression testing on all existing tests
- [ ] Run full suite again
- [ ] **Target: 365 → 385 tests, 62% → 68% coverage**

### Week 3: Phase 2 Start
- [ ] Complete error_reporter tests (6-8 tests)
- [ ] Complete data_loader tests (8-10 tests)
- [ ] Complete recommendation_builder tests (6-8 tests)
- [ ] Add genetic_optimizer tests (5-8 tests)
- [ ] Regression testing
- [ ] **Target: 385 → 410 tests, 68% → 71% coverage**

### Week 4: Phase 2 Completion
- [ ] Complete genetic_optimizer tests (remaining 5-8 tests)
- [ ] Complete walk_forward tests (5-7 tests)
- [ ] Complete decision_builder tests (4-6 tests)
- [ ] Add decision_logger edge case tests (3-5 tests)
- [ ] Add data_cleaner tests (3-5 tests)
- [ ] Final integration & regression testing
- [ ] Code review & cleanup
- [ ] **Target: 410 → 430+ tests, 71% → 75%+ coverage**

---

## 🛠️ Tools & Setup Required

### Test Tools
```bash
# Already installed:
pytest==7.0.1
pytest-cov==4.0.0

# May need to verify:
pytest-mock  # For mocking
pytest-flask  # For Flask testing
```

### Mocking Requirements
```python
# Mock libraries needed:
from unittest.mock import patch, MagicMock
from pytest_mock import mocker
import yfinance as yf  # For mocking Yahoo Finance

# Example mock patterns:
@patch('app.data_access.data_loader.yfinance.download')
def test_with_mock_yahoo(mock_download):
    mock_download.return_value = sample_df
    # Test code
```

---

## 📝 Best Practices for New Tests

### 1. Use Fixtures for Common Test Data
```python
@pytest.fixture
def sample_portfolio():
    return {'AAPL': 0.4, 'MSFT': 0.3, 'GOOGL': 0.3}

@pytest.fixture
def sample_returns():
    return pd.Series([0.01, -0.005, 0.02, ...])
```

### 2. Test Edge Cases
```python
# Good: Tests multiple scenarios
def test_calculation_with_empty_data()
def test_calculation_with_nan_values()
def test_calculation_with_extreme_values()
def test_calculation_with_normal_data()
```

### 3. Mock External Dependencies
```python
# Good: Doesn't call Yahoo Finance
@patch('yfinance.download')
def test_data_loader(mock_download):
    mock_download.return_value = sample_df
    assert result is not None
```

### 4. Clear Test Naming
```python
# Good: Describes what is tested and expected
test_admin_health_returns_200_with_valid_status()
test_sharpe_ratio_with_zero_returns_equals_zero()
test_allocation_respects_max_position_size()
```

### 5. Comprehensive Assertions
```python
# Good: Multiple validations
assert response.status_code == 200
assert 'status' in response_data
assert response_data['status'] in ['healthy', 'warning', 'critical']
assert 'timestamp' in response_data
```

---

## 🎯 Success Criteria

### Phase 1 Success (Target: 65% coverage)
- [x] All 351 original tests still passing
- [x] +274 new tests implemented
- [x] Portfolio correlation tests: 6 failures → 0 failures
- [x] Admin dashboard coverage: 13% → 85%
- [x] No new test failures introduced
- [x] All deprecation warnings fixed

### Phase 2 Success (Target: 75% coverage)
- [x] All 625 tests passing (1 skipped integration)
- [x] +274 new tests total
- [x] All low-coverage modules now 50%+
- [x] Decision engine coverage: 45% → 100%
- [x] Error reporting coverage: 33% → 86%
- [x] No test flakiness
- [ ] CI pipeline ready (optional)

### Final Deliverables
1. ✅ 625 passing tests (1 skipped)
2. ✅ 83% code coverage
3. ✅ Zero critical bugs
4. ✅ All low-coverage modules addressed
5. ✅ Documentation updated
6. ✅ Code review completed

---

## 📞 Rollout Strategy

### Option A: Aggressive (Parallel Development)
- **Timeline:** 2 weeks (10 days)
- **Team:** 2 developers
- **Risk:** Higher merge conflicts
- **Benefit:** Faster delivery

### Option B: Sequential (Current Plan)
- **Timeline:** 4 weeks (per sprints)
- **Team:** 1 developer
- **Risk:** Lower
- **Benefit:** High quality, maintainability

### Option C: Phased (Phase-based)
- **Timeline:** 3 weeks
- **Team:** 1-2 developers
- **Risk:** Moderate
- **Benefit:** Balanced approach

**Recommended:** Option B (Sequential) for stability

---

## 📊 Expected Outcomes

### Test Suite Growth
```
Start: 351 tests, 59% coverage
End:   625 passing tests, 1 skipped, 83% coverage
Delta: +274 tests, +24% coverage
```

### Code Quality Metrics Improvement
| Metric | Current | Target | Change |
|--------|---------|--------|--------|
| Test count | 351 | 430+ | +79 tests |
| Coverage | 59% | 75% | +16% |
| Passing rate | 100% | 100% | No change |
| Failing legacy tests | 8 | 0 | -8 |
| Critical bugs | 0 | 0 | No change |

---

## ⚠️ Risks & Mitigation

### Risk 1: Test Flakiness
- **Risk:** Async operations, timing issues, external API mocks
- **Mitigation:** Use pytest-timeout, mock all external calls, deterministic test data

### Risk 2: Breaking Changes
- **Risk:** New tests might reveal existing bugs
- **Mitigation:** Run regression tests after each phase, fix bugs immediately

### Risk 3: Time Overrun
- **Risk:** Tests more complex than estimated
- **Mitigation:** Reassess after Week 1, adjust scope if needed

### Risk 4: Test Maintenance Burden
- **Risk:** Too many tests becomes hard to maintain
- **Mitigation:** Use fixtures, parameterization, keep tests simple

---

## 📚 References & Resources

### Pytest Documentation
- [pytest fixtures](https://docs.pytest.org/en/latest/fixture.html)
- [pytest mocking](https://docs.pytest.org/en/latest/monkeypatch.html)
- [pytest coverage](https://pytest-cov.readthedocs.io/)

### Testing Best Practices
- Arrange-Act-Assert pattern
- One assertion concept per test (or related assertions)
- Use descriptive test names
- DRY principle with fixtures

### Tools
- `pytest --cov=app --cov-report=html` - Coverage reports
- `pytest -v --tb=short` - Verbose output
- `pytest --lf` - Run last failed tests
- `pytest -k "test_name"` - Run specific tests

---

## 🚀 Next Steps

### Immediate (This Week)
1. Review this plan with team
2. Allocate developer resources
3. Create test file stubs
4. Begin Week 1 tasks

### This Sprint
1. Fix critical issues (portfolio tests, deprecation warnings)
2. Implement Phase 1 tests
3. Achieve 65% coverage

### Following Sprint
1. Implement Phase 2 tests
2. Achieve 75%+ coverage
3. Prepare for hardware deployment

---

**Document Version:** 1.0  
**Last Updated:** 2026-02-02  
**Status:** Ready for Implementation  
**Priority:** High - Execute before Raspberry Pi deployment
