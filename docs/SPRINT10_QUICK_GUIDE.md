# Sprint 10 - Gyors Implementációs Útmutató

**Verzió:** 1.0  
**Dátum:** 2026-02-02  
**Fázis:** Sprint 10 Planning  

---

## 🎯 Cél

- **Hibák:** 5 ismert probléma megoldása
- **Coverage:** 59% → 75%+ (59 teszt +)
- **Időtartam:** 4 hét (40-50 óra)
- **Csapat:** 1 fejlesztő

---

## 🐛 5 Ismert Hiba - Prioritás szerinti

### 1. ✅ AdminDashboard Blueprint (KÉSZ - 2026-02-02)
- **Probléma:** Endpoints nem regisztrálva
- **Megoldás:** Import + `app.register_blueprint(admin_bp)`
- **Státusz:** FIXED

### 2. 🔴 Portfolio Correlation Teszt Hibák (6 failure)
- **Probléma:** Test fixture nincs korrelációs adat
- **Megoldás:** Mock fixture-val: `populated_test_db`
- **Munka:** 3-4 óra
- **Eredmény:** +8 passing test

### 3. 🟡 Pandas Deprecation Warning (10 warning)
- **Probléma:** `pd.Series()` dtype nem specifikált
- **Megoldás:** `pd.Series(dtype=float)`
- **Munka:** 30 perc
- **Fájl:** `app/backtesting/backtester.py:68`

### 4. 🟡 AdminDashboard Coverage (13%)
- **Probléma:** Flask endpoint teszt nélkül
- **Megoldás:** Integration tests: 8-10 test
- **Munka:** 4-5 óra
- **Cél:** 40% → 60%

### 5. 🟡 PerformanceAnalytics Coverage (46%)
- **Probléma:** Edge case tesztek hiányzanak
- **Megoldás:** 8-10 új test (Sharpe, Sortino, drawdown)
- **Munka:** 4-5 óra
- **Cél:** 60% → 75%

---

## 📈 Coverage Javítás - Modulok Prioritás Szerinti

### Phase 1: Hét 1-2 (TARGET: 62% → 68%)

| Modul | Jelenlegi | Cél | Tesztek | Óra |
|-------|-----------|-----|---------|-----|
| admin_dashboard | 13% | 40% | 8-10 | 5 |
| performance_analytics | 46% | 65% | 8-10 | 5 |
| decision_engine | 45% | 70% | 10-12 | 6 |

**Súlypont:** Core decision logic
**Munka:** 16 óra
**Teszt hozzáadás:** +28-32 teszt

### Phase 2: Hét 3-4 (TARGET: 68% → 75%+)

| Modul | Jelenlegi | Cél | Tesztek | Óra |
|-------|-----------|-----|---------|-----|
| error_reporter | 33% | 55% | 6-8 | 4 |
| data_loader | 16% | 45% | 8-10 | 6 |
| genetic_optimizer | 16% | 45% | 10-12 | 8 |
| recommendation_builder | 32% | 70% | 6-8 | 5 |
| walk_forward | 34% | 70% | 5-7 | 4 |
| decision_builder | 41% | 75% | 4-6 | 4 |

**Súlypont:** Advanced features & edge cases
**Munka:** 31 óra
**Teszt hozzáadás:** +45-51 teszt

---

## ⏱️ Hetenkénti Terv

### **1. HÉT** (10.5 óra)
**목표:** Bug fixek + Phase 1 start

```
Nap 1-2: Portfolio correlation tesztek javítása (4 óra)
  └─ Mock fixture: populated_test_db
  └─ 6 failing → 8 passing

Nap 2-3: Deprecation warning fix (0.5 óra)
  └─ pd.Series(dtype=float)

Nap 3-5: AdminDashboard integration tesztek (4-5 óra)
  └─ 8-10 Flask endpoint teszt
  └─ 13% → 40% coverage

Nap 5-6: Performance analytics extended tesztek (3 óra)
  └─ Sharpe, Sortino, drawdown tesztek
  └─ 46% → 60% coverage

VERIFIKÁCIÓ:
$ pytest tests/ -v
$ pytest --cov=app --cov-report=html
TARGET: 351 → 365 tests, 59% → 62%
```

### **2. HÉT** (12 óra)
**Cél:** Phase 1 completion

```
Nap 1-3: Decision engine tesztek (6 óra)
  └─ Signal generation: bull/bear/sideways
  └─ Allocation logic edge cases
  └─ Error scenarios
  └─ 45% → 70% coverage

Nap 3-4: Error reporter basic tesztek (3-4 óra)
  └─ Export functionality
  └─ Severity filtering

Nap 5-6: DataLoader mock tesztek (4-5 óra)
  └─ yfinance mock-kal
  └─ 16% → 30% coverage

VERIFIKÁCIÓ:
$ pytest tests/ -v
TARGET: 365 → 385 tests, 62% → 68%
```

### **3. HÉT** (15 óra)
**Cél:** Phase 2 start

```
Nap 1-2: Error reporter complete (4 óra)
  └─ CSV/JSON export
  └─ 33% → 55% coverage

Nap 2-3: DataLoader extended (6 óra)
  └─ Rate limiting
  └─ Cache behavior
  └─ 16% → 45% coverage

Nap 4-5: Recommendation builder tesztek (5 óra)
  └─ Recommendation generation
  └─ 32% → 70% coverage

VERIFIKÁCIÓ:
TARGET: 385 → 410 tests, 68% → 71%
```

### **4. HÉT** (19 óra)
**Cél:** Phase 2 completion + Final

```
Nap 1-2: Genetic optimizer tesztek (8 óra)
  └─ Crossover/mutation
  └─ Fitness calculation
  └─ 16% → 45% coverage

Nap 3-4: Walk forward tesztek (4 óra)
  └─ 34% → 70% coverage

Nap 4-5: Decision builder tesztek (4 óra)
  └─ 41% → 75% coverage

Nap 5-6: Végső integrációs tesztek (3 óra)
  └─ Regression tesztek
  └─ Code review

VERIFIKÁCIÓ:
TARGET: 410 → 430+ tests, 71% → 75%+
```

---

## 📋 Checklist - Heti Lebontás

### ✅ 1. HÉT Feladatok
- [ ] Portfolio correlation fixture létrehozása
- [ ] 6 failing teszt javítása
- [ ] Pandas warning fix
- [ ] AdminDashboard integration tesztek (8-10)
  - [ ] /admin/health
  - [ ] /admin/performance/summary
  - [ ] /admin/performance/detailed
  - [ ] /admin/performance/chart-data
  - [ ] /admin/errors/summary
  - [ ] /admin/errors/recent
  - [ ] /admin/errors/critical
  - [ ] /admin/capital/status
- [ ] Performance analytics extended (8-10)
  - [ ] test_sharpe_ratio_calculation
  - [ ] test_sharpe_with_zero_returns
  - [ ] test_sortino_ratio
  - [ ] test_max_drawdown
  - [ ] test_win_rate
  - [ ] test_profit_factor
- [ ] Full test suite run ✓
- [ ] Coverage report ✓

### ✅ 2. HÉT Feladatok
- [ ] Decision engine tests (10-12)
  - [ ] Bull market signal generation
  - [ ] Bear market signal generation
  - [ ] Sideways market handling
  - [ ] High confidence allocation
  - [ ] Low confidence allocation
- [ ] Error reporter basic tests (3-4)
- [ ] DataLoader mock tests (4-5)
- [ ] Regression testing
- [ ] Coverage report ✓

### ✅ 3. HÉT Feladatok
- [ ] Error reporter complete (6-8 tesztek)
- [ ] DataLoader extended (8-10 tesztek)
- [ ] Recommendation builder (6-8 tesztek)
- [ ] Genetic optimizer start (5-8 tesztek)
- [ ] Coverage report ✓

### ✅ 4. HÉT Feladatok
- [ ] Genetic optimizer complete (5-8 tesztek)
- [ ] Walk forward tests (5-7 tesztek)
- [ ] Decision builder tests (4-6 tesztek)
- [ ] Decision logger edge cases (3-5 tesztek)
- [ ] Data cleaner tests (3-5 tesztek)
- [ ] Final integration testing
- [ ] Code review
- [ ] Final coverage report ✓

---

## 🚀 Parancsok & Tools

### Tesztek futtatása
```bash
# Összes teszt
pytest tests/ -v

# Specifikus modul
pytest tests/test_admin_dashboard.py -v

# Coverage report
pytest tests/ --cov=app --cov-report=html

# Utolsó failed tesztek
pytest --lf

# Specifikus teszt név
pytest -k "test_sharpe" -v
```

### Coverage megjelenítés
```bash
# Terminal report
pytest --cov=app --cov-report=term

# HTML report
pytest --cov=app --cov-report=html
# Böngészőben: htmlcov/index.html
```

### Debugging
```bash
# Verbose + print output
pytest tests/test_file.py -vv -s

# Debugger
pytest tests/test_file.py --pdb

# Stop on first failure
pytest tests/ -x
```

---

## 📊 Előhaladás Nyomon Követés

### Tesztek & Coverage Trend
```
Week 0: 351 tests, 59% ✅
Week 1: 365 tests, 62% TARGET
Week 2: 385 tests, 68% TARGET
Week 3: 410 tests, 71% TARGET
Week 4: 430+ tests, 75%+ TARGET
```

### Heti Reportozás
```markdown
**Week N Progress Report**

✅ Completed:
- [x] Task 1
- [x] Task 2

🔄 In Progress:
- [ ] Task 3

❌ Blocked:
- [ ] Task 4 (reason)

📊 Metrics:
- Tests: XXX → XXX (+XX)
- Coverage: XX% → XX%
- Bugs fixed: N

⚠️ Issues:
- Issue 1: description
```

---

## 🎯 Success Criteria

### Végső Metrikák
| Metrika | Kezdet | Vég | Cél |
|---------|--------|-----|-----|
| Tesztek | 351 | 430+ | ✅ |
| Coverage | 59% | 75%+ | ✅ |
| Hiba ráta | 0% | 0% | ✅ |
| Legacy failures | 8 | 0 | ✅ |

### Minőség Mutatók
- ✅ Összes teszt passing (100%)
- ✅ Nincsenek kritikus hibák
- ✅ Regressions: 0
- ✅ Dokumentáció friss
- ✅ Code review passed

---

## 📞 Kapcsolat & Támogatás

### Kérdések?
- Tekintsd meg: `docs/FAQ.md` (40 Q&A)
- Hibaelhárítás: `docs/TROUBLESHOOTING_GUIDE.md`
- Full plan: `docs/BUG_FIX_COVERAGE_PLAN.md`

### Eszközök Szükséges
```bash
pytest==7.0.1
pytest-cov==4.0.0
pytest-mock  # For mocking
```

---

## 🔗 Fontos Linkek

- 📄 [Teljes terv](BUG_FIX_COVERAGE_PLAN.md)
- 🧪 [Test Status Report](02_testing/TEST_STATUS_REPORT.md)
- 📊 [Coverage Report](../htmlcov/index.html)
- 📖 [FAQ](FAQ.md)
- 🆘 [Troubleshooting](TROUBLESHOOTING_GUIDE.md)

---

**Kezdés dátuma:** 2026-02-03 (jövő hét)  
**Becsült befejezés:** 2026-03-02  
**Felelős:** Sprint 10 fejlesztő  
**Státusz:** 🟢 READY TO START
