# Changelog

All notable changes are documented here. Format: [Sprint X – YYYY-MM-DD]

---

## [Sprint 18 – 2026-04-13]

### Fixed
- **`decision_effectiveness.py` NameError** (`app/core/analysis/decision_effectiveness.py`): `get_settings` névhiba javítva → `build_settings` alias beállítva. Napló-szintű `ImportError` megszűnt.

### Added
- **Deprecation warnings** (`app/decision/*.py`, 20 fájl): `warnings.warn(..., DeprecationWarning)` hozzáadva minden shim/passthrough fájlhoz, így a direkt import azonnal figyelmeztet.
- **`SqliteDecisionRepository` 100% lefedettség** – 48 új teszt: `test_sprint18_repositories.py`
- **`DataManagerRepository` 99% lefedettség** – 96 új teszt: `test_sprint18_data_manager_repo.py`
- 8 új tesztfájl: `test_sprint18_execution_engines.py`, `test_sprint18_repositories.py`, `test_sprint18_data_manager_repo.py`, `test_sprint18_sensitivity.py`, `test_sprint18_governance.py`, `test_sprint18_analysis_shims.py`, `test_sprint18_execution_utils.py`, `test_sprint18_coverage_push.py`

### Metrics
- Tesztek: 1503 → **1820** (+317)
- Coverage: 75% → **80%**

---

## [Sprint 17 – 2026-04-10]

### Fixed / Refactored
- **DI fallbackek eltávolítva** (`app/core/decision/`, 8+ fájl): `build_settings()` közvetlen hívások helyett explicit settings injection. DeprecationWarning-ok megszűntek.
- Coverage uplift: use-case réteg, validation/governance modulok, services + execution ágak célzott tesztbővítéssel.

### Metrics
- Tesztek: ~1165 → **1503** (+338)
- Coverage: ~61% → **75%**

---

## [Sprint 16 – 2026-04-10]

### Added
- **DI Cleanup** – `build_application()` container véglegesítve, lazy init eltávolítva
- **ML ensemble tag** (`ENABLE_ML_ENSEMBLE` flag, default: false) – LSTM opcionális ensemble member
- **Broker adapter shell** – `LiveExecutionEngine` + `AlpacaExecutionEngine` bewired (`BROKER_ADAPTER` env var)

### Metrics
- Tesztek: 1130 → **~1165**

---

## [Sprint 15 – 2026-04-09]

### Fixed
- `dh.confidence` column guard – `PRAGMA table_info` check, 5 pre-existing teszthiba javítva
- Phase7 production import migration (`audit_builder.py`, `admin_dashboard.py`)
- `fetch_recent_outcomes` graceful fallback ha nincs outcomes tábla

### Added
- `LiveExecutionEngine` broker-agnostic shell bewired
- `PortfolioRebalancer` bewired (`ENABLE_REBALANCER` flag, default: false)
- `enforce_correlation_limits` bewired (`ENABLE_CORRELATION_LIMITS` flag, default: false)
- `ENABLE_ML_ENSEMBLE`, `ENABLE_REBALANCER`, `REBALANCE_THRESHOLD`, `ENABLE_CORRELATION_LIMITS`, `MAX_CORRELATION` Settings mezők

### Metrics
- Tesztek: 1112 → **1130** (+18)

---

## [Sprint 14 – 2026-04-07]

### Fixed – Quant/Pénzügyi logika
- **GA fitness fordított drawdown büntetés** (`app/optimization/fitness.py`): `- 2.0 * max_drawdown` → `- 2.0 * abs(max_drawdown)`. A `max_drawdown` negatív szám, ezért az eredeti képlet jutalmzata a nagy drawdown-t.
- **Bollinger Band API inversion** (`app/core/analysis/analyzer.py`): `middle, upper, lower = volatility.bbands(...)` → `upper, middle, lower`. A SELL szignál most valóban a felső sávnál sül el (nem az SMA-nál), és az RL observation vector `BB_upper` feature-je most a tényleges felső sávot tartalmazza.
- **Walk-Forward cache kontamináció** (`app/optimization/genetic_optimizer.py`): A `_FITNESS_CACHE` kulcsa kiegészült adat-slice fingerprint-tel (`len|start|end`) – különböző WF foldok azonos paraméterrel most nem osztják meg a cached értéket.
- **Sharpe ratio helytelen annualizáció** (`app/backtesting/backtester.py`): A `sqrt(252)` állandó helyett trade-frekvencia-alapú annualizáló faktor (`sqrt(trades_per_year)`). Ez megakadályozza a 2–5× torzítást ritkán kereskedő stratégiáknál.

### Fixed – Architektúra
- **`core/decision/` réteg `bootstrap` → `config` import** (8 fájl): `from app.bootstrap.build_settings import build_settings` → `from app.config.build_settings import build_settings`. Megszünteti a core→bootstrap irányú függési sértést.
- **`recommender.py` `os.getenv()` → settings flag**: `os.getenv("VALIDATION_DISABLE_SAFETY")` → `getattr(cfg, "VALIDATION_DISABLE_SAFETY", False)`. A konfiguráció olvasása most a Settings objektumon keresztül történik.
- **`TrainRLModelUseCase` exception leak** (`app/application/use_cases/train_rl_model.py`): `train_rl_agent()` hibái most `error()` Result-ot adnak vissza, nem kezeletlen traceback-et.

### Added
- **CONTRIBUTING.md**: Branch naming, PR workflow, kódolási elvek, tesztelési szabályok
- **CHANGELOG.md**: Ez a fájl

---

## [Sprint 13 – 2026-04-02 → 2026-04-07]

### Fixed
- ADX valódi Wilder DM+/DM- implementáció (`app/core/indicators/trend.py`)
- PaperExecution SELL fix – `PaperPosition(**v)` konverzió a `_load_latest_state()`-ben
- Commission levonás BUY/SELL esetén – `TRANSACTION_FEE_PCT` alkalmazva
- Governance FrozenInstanceError – `os.environ["PIPELINE_AUDIT_MODE"]` via env var
- RL end date dinamikus – `end = end or str(date.today())`
- `ENABLE_RL=false` log warning – `RunMonthlyRetrainingUseCase`-ben
- RSI/ATR Wilder EMA simítás – helyes implementáció
- SQLite WAL mode bekapcsolva (`DataManager._get_conn()`)
- Hardcoded `end="2025-06-30"` → `str(date.today())` az analyzer `__main__` blokkokban

### Added
- `.env.example` – teljes environment variable sablon
- Három review agent (economist, IT architect, docs reviewer)

---

## [Sprint 12 és korábbiak]

Részletes sprint history: [docs/SPRINTS.md](docs/SPRINTS.md)

### Breaking Changes (referencia)
- **Sprint 10**: MACD visszatérési érték 2-tuple → 3-tuple (`macd, signal, histogram`)
- **Sprint 12**: RL observation vector 10 feature → 11 feature (ADX hozzáadva)
