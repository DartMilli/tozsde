# IMPLEMENTATION SKELETON - CODE FILES CREATED

**Date Created:** 2026-01-21  
**Total Files:** 15 skeleton modules  
**Total Lines:** ~2000+ TODOs with structure

---

## FILE MANIFEST

### SPRINT 1: TEST SUITE (8 files)

| File | Purpose | Status | Est. Hours |
|------|---------|--------|-----------|
| `tests/__init__.py` | Test package marker | ✅ Complete | 0.5h |
| `tests/conftest.py` | Pytest fixtures & config | ✅ Skeleton | 2h |
| `tests/test_indicators.py` | SMA, indicator tests | ✅ Skeleton | 4h |
| `tests/test_fitness.py` | Fitness function tests | ✅ Skeleton | 3h |
| `tests/test_backtester.py` | Backtest logic tests | ✅ Skeleton | 4h |
| `tests/test_walk_forward.py` | Walk-forward tests | ✅ Skeleton | 3h |
| `tests/test_data_manager.py` | SQLite DAL tests | ✅ Skeleton | 3h |
| `tests/test_allocation.py` | Capital allocation tests | ✅ Skeleton | 3h |
| `tests/test_daily_pipeline.py` | Integration tests | ✅ Skeleton | 4h |

**Sprint 1 Total:** ~22 hours

---

### SPRINT 2: LEARNING SYSTEM (2 files)

| File | Purpose | Status | Est. Hours |
|------|---------|--------|-----------|
| `app/decision/drift_detector.py` | Performance drift detection | ✅ Skeleton | 4h |
| `app/notifications/alerter.py` (updated) | See below | ✅ Skeleton | 3h |

**Sprint 2 Total:** ~7 hours (+ RL activation integration)

---

### SPRINT 3: PORTFOLIO OPTIMIZATION (2 files)

| File | Purpose | Status | Est. Hours |
|------|---------|--------|-----------|
| `app/decision/risk_parity.py` | Risk parity allocation | ✅ Skeleton | 5h |
| `app/decision/rebalancer.py` | Portfolio rebalancing | ✅ Skeleton | 5h |

**Sprint 3 Total:** ~10 hours

---

### SPRINT 4: ADMIN & MONITORING (2 files)

| File | Purpose | Status | Est. Hours |
|------|---------|--------|-----------|
| `app/infrastructure/metrics.py` | System metrics tracking | ✅ Skeleton | 4h |
| `app/notifications/alerter.py` | Error alerting system | ✅ Skeleton | 4h |

**Sprint 4 Total:** ~8 hours

---

### SPRINT 5: BONUS ANALYSIS (1 file)

| File | Purpose | Status | Est. Hours |
|------|---------|--------|-----------|
| `app/reporting/pyfolio_report.py` | PyFolio integration | ✅ Skeleton | 4h |

**Sprint 5 Total:** ~4 hours

---

## SKELETON STRUCTURE OVERVIEW

Each skeleton file includes:
- ✅ **Docstrings** — Purpose, usage examples, responsibility
- ✅ **Class signatures** — Full method signatures with type hints
- ✅ **TODO markers** — Where implementation logic goes
- ✅ **Stub returns** — Placeholder return values
- ✅ **Comments** — Step-by-step implementation guidance
- ✅ **Import structure** — All required imports (some commented)

### Example skeleton pattern:

```python
def check_drift(self, ticker: str, current_score: float) -> Dict:
    """Check for performance degradation."""
    # TODO: Implement
    # 1. Load historical scores
    # 2. Compute average
    # 3. Calculate drift percentage
    # 4. Classify alert level
    
    return {
        "drifting": False,
        "drift_pct": 0.0,
        "alert_level": "NONE"
    }
```

---

## NEXT STEPS

### WEEK 1: IMPLEMENT SPRINT 1 TESTS

1. **Copy skeleton files to active development**
   ```bash
   cd tests/
   # All 9 files already in place
   ```

2. **Fill in test cases**
   - Start with `conftest.py` (fixtures)
   - Then `test_indicators.py` (simplest logic)
   - Then `test_backtester.py` (complex logic)
   - Finally `test_daily_pipeline.py` (integration)

3. **Run tests**
   ```bash
   pytest tests/ -v
   pytest tests/ --cov=app  # Coverage report
   ```

### WEEK 2-3: IMPLEMENT SPRINT 2–3

1. Fill in `drift_detector.py`
   - Implement `_load_historical_scores()`
   - Implement `check_drift()` logic
   - Test with real data

2. Fill in `risk_parity.py`
   - Implement volatility computation
   - Implement weight normalization
   - Test with sample portfolio

3. Fill in `rebalancer.py`
   - Implement drift detection
   - Implement trade generation
   - Test rebalancing logic

### WEEK 4: IMPLEMENT SPRINT 4

1. Fill in `metrics.py`
   - Implement JSONL logging
   - Implement metrics aggregation
   - Add to dashboard

2. Fill in `alerter.py`
   - Implement error classification
   - Implement email alerts
   - Add to main.py exception handling

### WEEK 5: SPRINT 5 (BONUS)

1. Fill in `pyfolio_report.py` (optional)
   - Test PyFolio dependency
   - Implement metric calculations

---

## INTEGRATION CHECKLIST

### To activate drift detection:
```python
# In app/reporting/audit_builder.py
from app.decision.drift_detector import PerformanceDriftDetector

detector = PerformanceDriftDetector()
for ticker in tickers:
    drift_info = detector.check_drift(ticker, wf_score)
    audit["drift_alerts"].append(drift_info)
```

### To activate risk parity:
```python
# In main.py run_daily()
from app.decision.risk_parity import apply_risk_parity

decisions = apply_risk_parity(decisions, price_history)
```

### To activate error alerting:
```python
# In main.py
from app.notifications.alerter import ErrorAlerter, catch_and_alert

@catch_and_alert
def run_daily():
    ...
```

### To activate metrics tracking:
```python
# In main.py
from app.infrastructure.metrics import SystemMetrics

metrics = SystemMetrics()
metrics.log_pipeline_execution(ticker, "success", duration)
```

---

## TESTING THE SKELETONS

```bash
# Verify all files created
find . -name "*.py" -path "*/tests/*" -o -name "drift_detector.py" -o -name "risk_parity.py" | sort

# Check for syntax errors
python -m py_compile tests/*.py app/decision/*.py app/infrastructure/*.py app/notifications/*.py

# Count TODOs
grep -r "TODO:" tests/ app/decision/*.py app/infrastructure/*.py app/notifications/*.py | wc -l
# Expected: ~100+ TODOs
```

---

## ESTIMATED TIMELINE

| Phase | Duration | Files | Lines | % Complete |
|-------|----------|-------|-------|-----------|
| Skeleton Creation | Done ✅ | 15 | ~2000 | 25% |
| Sprint 1 (Tests) | 1 week | 9 | ~1200 | 50% |
| Sprint 2 (RL+Drift) | 1 week | 2 | ~400 | 75% |
| Sprint 3 (Portfolio) | 1 week | 2 | ~600 | 85% |
| Sprint 4 (Admin) | 1 week | 2 | ~500 | 95% |
| Sprint 5 (PyFolio) | 1 week | 1 | ~200 | 100% |
| **TOTAL** | **~5 weeks** | **15** | **~2900** | **100%** |

---

## QUALITY GATES

- [ ] 100+ tests passing
- [ ] Code coverage >80%
- [ ] All TODOs implemented
- [ ] No hardcoded values (use Config)
- [ ] All exceptions handled
- [ ] Logging at appropriate levels
- [ ] Type hints on all functions
- [ ] Docstrings on all classes/functions

---

## FILES CREATED SUMMARY

✅ **15 skeleton files** ready for implementation  
✅ **~2000 lines** of structured code with TODOs  
✅ **100+ implementation tasks** marked clearly  
✅ **Full test coverage skeleton** for 8 test modules  
✅ **P7–P9 features** with integration points

**Status:** Ready for Week 1 implementation sprint!

