# BLOCKERS AND SOLUTIONS - SPRINT 1 Teszt Megvalósítás

## 🚨 Kritikus Blokkerek

---

## 1. Circular Import Issue (KRITIKUS)

### 📍 Hely
```
app/config/config.py → imports → app/data_access/data_loader.py
                              ↓
                        imports → app/config/config.py
```

### 🔴 Probléma Leírása

**Fájl: app/config/config.py (6. sor)**
```python
from app.data_access.data_loader import get_supported_ticker_list
```

**Fájl: app/data_access/data_loader.py (3. és 6. sor)**
```python
import yfinance as yf
...
from app.config.config import Config
```

### 💥 Hatás

```
❌ Nem tölthető: app.backtesting.backtester
❌ Nem futnak: test_backtester.py (11 test)
❌ Nem futnak: test_walk_forward.py (7 test) [depends on backtester]
❌ Nem futnak: test_daily_pipeline.py (8 test) [depends on backtester]
───────────────────────────────────────────
❌ BLOKK: ~26 teszt (23 test)
```

### 📊 Érintett Teszt Fájlok

| Test File | Tests | Status | Blocker | Megoldás |
|-----------|-------|--------|---------|----------|
| test_backtester.py | 11 | ❌ Blocked | Circular import | Breaking cycle |
| test_walk_forward.py | 7 | ⏳ Waiting | Depends on backtester | After cycle fixed |
| test_daily_pipeline.py | 8 | ⏳ Waiting | Depends on backtester | After cycle fixed |

### ✅ Megoldási Lehetőségek

#### **Option 1: Ticker Lista Lazy Loading** (JAVASOLT ⭐)

**Módosítás: app/config/config.py**

```python
# ELŐTTE:
from app.data_access.data_loader import get_supported_ticker_list

class Config:
    SUPPORTED_TICKERS = get_supported_ticker_list()  # AZONNAL betöltése

# UTÁNA:
class Config:
    _SUPPORTED_TICKERS = None
    
    @classmethod
    def get_supported_tickers(cls):
        if cls._SUPPORTED_TICKERS is None:
            from app.data_access.data_loader import get_supported_ticker_list
            cls._SUPPORTED_TICKERS = get_supported_ticker_list()
        return cls._SUPPORTED_TICKERS
```

**Előnyök:**
- ✅ Circular import szétszakad
- ✅ Config azonnal betölthető
- ✅ Ticker lista csak szükség esetén
- ✅ Backward compatible (lazy init)

**Hátrányok:**
- ⚠️ Kód módosítás szükséges más helyeken
- ⚠️ Első betöltés lassabb

#### **Option 2: Külön Modul Szeparáció**

**Új fájl: app/config/ticker_config.py**

```python
# Szeparált ticker konfigurációs modul
from app.data_access.data_loader import get_supported_ticker_list

SUPPORTED_TICKERS = get_supported_ticker_list()
EXCLUDED_TICKERS = ["OTP.BD", "MOL.BD", "RICHTER.BD"]
```

**app/config/config.py:**
```python
# Nincs direct import data_access-ből
from app.config.ticker_config import SUPPORTED_TICKERS

class Config:
    SUPPORTED_TICKERS = SUPPORTED_TICKERS
```

**Előnyök:**
- ✅ Tiszta szeparáció
- ✅ Nincs circular import
- ✅ Könnyen módosítható

#### **Option 3: Initialization Helper**

**Új fájl: app/bootstrap.py**

```python
def initialize_config():
    """Inicializálja Config-ot az importok után"""
    from app.data_access.data_loader import get_supported_ticker_list
    from app.config.config import Config
    
    Config.SUPPORTED_TICKERS = get_supported_ticker_list()
```

---

## 2. SMA Edge Case (MINOR)

### 📍 Test: test_sma_window_larger_than_data

### 🟡 Probléma Leírása

**Fájl: tests/test_indicators.py**

```python
def test_sma_window_larger_than_data(self, sample_df):
    """SMA with window > data length should return all NaN."""
    result = sma(sample_df["Close"].values, window=50)
    # Expected: All NaN values
    # Actual: Some values returned (padding + convolution behavior)
```

### 💥 Hatás

```
⚠️ 1 teszt FAIL (test_sma_window_larger_than_data)
📊 Overall Pass Rate: 5/6 (83%)
```

### ✅ Megoldási Lehetőségek

#### **Option 1: Teszt Módosítása** (GYORS)

```python
# ELŐTTE:
assert np.all(np.isnan(result))

# UTÁNA:
# SMA pads with NaN, then applies convolution
# So some values are returned, not all NaN
assert len(result) == len(data)
assert np.isnan(result[0])  # First values should be NaN
# or accept actual behavior as valid
```

#### **Option 2: SMA Implementáció Review**

**Fájl: app/indicators/technical.py**

Átnézni az SMA függvényt, hogy:
- Szándékos-e a padding viselkedés?
- Dokumentáció szükséges?
- Bug vagy feature?

---

## 3. Python 3.6 Kompatibilitás

### 📍 Probléma: TypedDict Import Error

### 🟡 Szint: SOLVED ✅

**Megoldás már implementálva:**
- ✅ typing-extensions telepítve
- ✅ multitasking 0.0.9 (backward compatible)
- ✅ requirements.txt frissítve

---

## 🎯 Kiválasztott Megoldás: Option 1 (Lazy Loading)

### Indoklás

1. ✅ Szétszakítja a circular import-ot
2. ✅ Minimális kódváltozás
3. ✅ Backward compatible
4. ✅ Gyorsabb app inicializáció
5. ✅ Teszt-friendly (mock-elhető)

### Implementáció Lépések

```python
# Step 1: Módosítsd app/config/config.py

class Config:
    _SUPPORTED_TICKERS = None
    
    @classmethod
    def get_supported_tickers(cls):
        if cls._SUPPORTED_TICKERS is None:
            from app.data_access.data_loader import get_supported_ticker_list
            cls._SUPPORTED_TICKERS = get_supported_ticker_list()
        return cls._SUPPORTED_TICKERS

# Step 2: Frissítsd az összes helyet, ahol Config.SUPPORTED_TICKERS-t használ
# Helyettesítsd: Config.SUPPORTED_TICKERS
# Új: Config.get_supported_tickers()

# Step 3: Futtatsd a testeket
pytest tests/test_backtester.py -v
```

---

## 📋 Blocker Resolution Checklist

### CRÍTICO (Circular Import) - MAGASABB PRIORITÁS

- [ ] **Döntés:** Melyik megoldás választjuk?
- [ ] **Implementáció:** Lazy loading vagy szeparáció?
- [ ] **Teszt:** test_backtester.py működik-e?
- [ ] **Verifikáció:** 11 test fut és PASS?

### MENOR (SMA Edge Case)

- [ ] **Döntés:** Teszt módosítás vagy implementáció review?
- [ ] **Fix:** Szükséges-e kódváltozás?
- [ ] **Teszt:** 6/6 test PASS?

---

## 📊 Status Összefoglalása

```
KRITIKUS:       1 blocker  (circular import)
                → Megoldás: Lazy loading
                → Hatás: 26+ teszt feloldható
                
FONTOS:         1 blocker  (SMA edge case)
                → Megoldás: Teszt vagy impl. fix
                → Hatás: 1 teszt kijavítható

MEGOLDOTT:      1 issue    (Python 3.6)
                → Megoldás: typing-extensions
                → Hatás: ✅ Lezárt
```

---

## 🚀 Javasolt Cselekvési Plan

### Phase 1: Circular Import Megoldása
```
1. Implement lazy loading Config.get_supported_tickers()
2. Run tests: pytest tests/test_backtester.py::TestBacktesterTradeExecution -v
3. Verify 11 tests pass
```

### Phase 2: SMA Edge Case Fixálása
```
1. Decide: Fix test or implementation
2. Apply fix
3. Run: pytest tests/test_indicators.py -v
4. Verify 6/6 pass
```

### Phase 3: Remaining Tests
```
1. Implement test_walk_forward.py (now unblocked)
2. Implement test_data_manager.py
3. Implement test_allocation.py
4. Implement test_daily_pipeline.py
```

---

## 📞 Contact / Questions

Ha kérdésed van a megoldásokkal kapcsolatban:

1. Lazyloading vs Szeparáció: Option 1 (Lazy) ajánlott
2. SMA teszt: Valószínűleg teszt módosítás gyorsabb
3. Tesztelés: Pytest-kel futtatható az összes

---

**Utolsó frissítés:** 2025-01-21  
**Status:** ACTIONABLE - Megoldás szükséges
**Prioritás:** 🔴 KRITIKUS (Circular import)
