# SPRINT 1: Testing Structure Guide

## 📚 Teszt Fájlok Szervezettsége

### Fájl Hierarchia

```
tests/
├── conftest.py                 ← Pytest fixtures (5 db)
├── test_indicators.py          ← Technical indicators (6 tests, 5✅ 1❌)
├── test_fitness.py             ← Fitness functions (9 tests, 9✅)
├── test_backtester.py          ← Backtester logic (11 tests, code ready)
├── test_walk_forward.py        ← Walk-forward validation (7 tests)
├── test_data_manager.py        ← Data access layer (7 tests)
├── test_allocation.py          ← Capital allocation (10 tests)
└── test_daily_pipeline.py      ← End-to-end pipeline (8 tests)
    
    ────────────────────────────────────────
    TOTAL: 88+ tests planned
```

---

## 🧪 Test Execution Status

### ✅ Fully Implemented & Working

**test_indicators.py** (6 tests)
```bash
pytest tests/test_indicators.py -v
Result: 5/6 PASS (83%)
```

**test_fitness.py** (9 tests)
```bash
pytest tests/test_fitness.py -v
Result: 9/9 PASS (100%)
```

### ⏳ Implemented but Blocked

**test_backtester.py** (11 tests)
```bash
pytest tests/test_backtester.py -v
Status: BLOCKED by circular import
Error: Cannot import app.config.config
Fix Needed: See BLOCKERS_AND_SOLUTIONS.md
```

### 📝 Not Yet Started

**test_walk_forward.py** (7 tests) - Blocked by backtester
**test_data_manager.py** (7 tests)
**test_allocation.py** (10 tests)
**test_daily_pipeline.py** (8 tests)

---

## 🔧 Test Fixtures (conftest.py)

Összes test számára elérhető fixtures:

### 1. **test_db**
```python
@pytest.fixture
def test_db():
    """SQLite test database"""
    # Returns: in-memory SQLite connection
```

### 2. **sample_ohlcv**
```python
@pytest.fixture
def sample_ohlcv():
    """OHLCV DataFrame (30 rows)"""
    # Returns: DataFrame with Close, High, Low, Open, Volume columns
    # Usage: Backtester, data manager tests
```

### 3. **sample_df**
```python
@pytest.fixture
def sample_df():
    """Simple price DataFrame"""
    # Returns: DataFrame with Close column
    # Usage: Indicator, fitness tests
```

### 4. **sample_signals**
```python
@pytest.fixture
def sample_signals():
    """Pre-computed trading signals"""
    # Returns: List of "BUY", "SELL", "HOLD" signals
    # Usage: Backtester tests
```

### 5. **mock_config**
```python
@pytest.fixture
def mock_config():
    """Mocked Config object"""
    # Returns: Mock Config with test parameters
    # Usage: Configuration-dependent tests
```

---

## 🎯 Test Pattern Reference

### Pattern 1: Direct Function Testing

**Fájl: tests/test_indicators.py**

```python
class TestSMA:
    def test_sma_basic_calculation(self, sample_df):
        """Test SMA function with simple input"""
        from app.indicators.technical import sma
        
        result = sma(sample_df["Close"].values, window=3)
        
        # Assertions
        assert len(result) == len(sample_df)
        assert result[2] > 0  # First valid SMA value
```

**Jellemzők:**
- ✅ Import a test inside method
- ✅ Use fixture data
- ✅ Clear assertions
- ✅ Test name describes behavior

### Pattern 2: Mock Object Testing

**Fájl: tests/test_fitness.py**

```python
class MockMetrics:
    """Mock backtest results for testing"""
    def __init__(self, trade_count=50, net_profit=1000.0, max_drawdown=0.1, winrate=0.55):
        self.trade_count = trade_count
        self.net_profit = net_profit
        self.max_drawdown = max_drawdown
        self.winrate = winrate

class TestFitness:
    def test_fitness_positive_return(self):
        """Test fitness calculation with profit"""
        from app.optimization.fitness import fitness_single
        
        metrics = MockMetrics(net_profit=1000.0)
        result = fitness_single(metrics)
        
        assert result > 0  # Profitable = positive fitness
```

**Jellemzők:**
- ✅ Mock class simulates complex objects
- ✅ Testable without full app initialization
- ✅ Clear inputs/outputs
- ✅ No side effects

### Pattern 3: Dataclass Integration

**Fájl: tests/test_fitness.py (Walk-Forward Tests)**

```python
from app.reporting.metrics import WalkForwardMetrics

class TestWalkForward:
    def test_wf_fitness_basic(self):
        """Test walk-forward aggregation"""
        from app.optimization.fitness import fitness_walk_forward
        
        wf_metrics = WalkForwardMetrics(
            avg_profit=500.0,
            avg_dd=0.1,
            profit_std=50.0,
            dd_std=0.01,
            negative_fold_ratio=0.2
        )
        
        result = fitness_walk_forward(wf_metrics)
        assert isinstance(result, float)
```

**Jellemzők:**
- ✅ Use dataclass constructor (NOT attribute assignment)
- ✅ All fields required
- ✅ Type-safe
- ✅ Clear data flow

---

## 📊 Test Coverage Map

### Modules Tested

```
app/
├── indicators/
│   └── technical.py           ✅ test_indicators.py (6 tests)
├── optimization/
│   └── fitness.py             ✅ test_fitness.py (9 tests)
├── backtesting/
│   ├── backtester.py          ⏳ test_backtester.py (11 tests) [BLOCKED]
│   └── walk_forward.py        ⏳ test_walk_forward.py (7 tests) [BLOCKED]
├── data_access/
│   └── data_manager.py        📝 test_data_manager.py (7 tests)
├── decision/
│   └── allocation.py          📝 test_allocation.py (10 tests)
└── main pipeline              📝 test_daily_pipeline.py (8 tests)
```

### Coverage Target: 80%+

---

## 🚀 Running Tests

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test File
```bash
pytest tests/test_fitness.py -v
```

### Run Single Test
```bash
pytest tests/test_fitness.py::TestFitnessFunction::test_fitness_positive_return -v
```

### Run with Coverage
```bash
pytest tests/ --cov=app --cov-report=html
```

### Run Specific Test Class
```bash
pytest tests/test_fitness.py::TestFitnessFunction -v
```

### Run Tests Matching Pattern
```bash
pytest tests/ -k "fitness" -v
```

---

## 🛠️ Test Development Workflow

### Step 1: Understand Module
```python
# Read the actual implementation first
# Understand inputs, outputs, edge cases
from app.optimization.fitness import fitness_single
```

### Step 2: Create Test Class
```python
class TestFitnessFunction:
    """Tests for single-period fitness calculation."""
```

### Step 3: Write Test Methods
```python
def test_fitness_positive_return(self):
    """Fitness should be positive for profitable trades."""
    # Arrange
    metrics = MockMetrics(net_profit=1000.0)
    
    # Act
    result = fitness_single(metrics)
    
    # Assert
    assert result > 0
```

### Step 4: Test Edge Cases
```python
def test_fitness_zero_trades(self):
    """Fitness should penalize insufficient trades."""
    metrics = MockMetrics(trade_count=5)  # Less than minimum
    result = fitness_single(metrics)
    assert result == -1e12 or result < -1000
```

---

## 📝 Test Documentation Format

**Javasolt dokumentáció minden test-ben:**

```python
class TestFitnessFunction:
    """Tests for single-period fitness calculation.
    
    Covers:
    - Edge cases (zero trades, all losses)
    - Normal operation (profitable strategies)
    - Sharpe ratio weighting
    - Drawdown penalties
    """
    
    def test_fitness_positive_return(self):
        """Fitness should be positive for profitable trades.
        
        Given:
            - Trade count: 50 (valid)
            - Net profit: 1000.0 (positive)
            - Max drawdown: 0.1 (reasonable)
            
        When:
            - fitness_single() is called
            
        Then:
            - Result should be > 0
        """
```

---

## ⚠️ Common Issues & Solutions

### Issue 1: "Cannot import module X"

**OK:** Import szükséges setup-hoz
```python
def test_example(self):
    from app.module import function  # ✅ This is fine
```

**NOT OK:** Import app level setup-nál
```python
from app.config.config import Config  # ❌ May cause circular import
```

### Issue 2: "Fixture not found"

**OK:** conftest.py-ben definiált fixture
```python
def test_example(self, sample_df):  # ✅ Uses conftest.py fixture
```

**NOT OK:** Nem létező fixture
```python
def test_example(self, missing_fixture):  # ❌ Not defined
```

### Issue 3: Flaky Tests (random failures)

**NOT OK:** Random data
```python
def test_calculation():
    random_value = random.random()
    assert calculate(random_value) > 0  # ❌ Unpredictable
```

**OK:** Deterministic data
```python
def test_calculation(self, sample_df):
    result = calculate(sample_df["Close"])
    assert result > 0  # ✅ Reproducible
```

---

## 📈 Test Quality Checklist

Minden test-nek teljesítenie kell:

- [ ] **Clear Name** - Test neve leírja a viselkedést
- [ ] **Single Purpose** - Egy dolog tesztelése
- [ ] **Deterministic** - Ugyanaz az input = ugyanaz az output
- [ ] **No Dependencies** - Nem függ más test-ektől
- [ ] **Fast** - < 1 másodperc futási idő
- [ ] **Documented** - Docstring leírja az expectedt
- [ ] **Edge Cases** - Szélsőséges esetek tesztelve
- [ ] **Assertions Clear** - Mit tesztel és miért

---

## 🎯 Next Session Tasks

### High Priority
1. ❌ Fix circular import (config.py)
2. ✅ Run test_backtester.py (after fix)
3. 📝 Implement test_walk_forward.py

### Medium Priority
4. 📝 Implement test_data_manager.py
5. 📝 Implement test_allocation.py

### Low Priority
6. 📝 Implement test_daily_pipeline.py
7. 📊 Run full coverage analysis

---

**Készítve:** 2025-01-21  
**Verzió:** 1.0  
**Status:** 🟡 In Progress (27% complete)
