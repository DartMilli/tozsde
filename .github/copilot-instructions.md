# Tozsde Trading System – Copilot Instructions

## Projekt áttekintés
Python alapú algoritmikus trading rendszer (daily/weekly/monthly pipeline).
- **Stack:** Python 3.11+, Flask, SQLite, PyTorch, stable-baselines3, pandas, yfinance
- **Architektúra:** Clean Architecture – bootstrap → interfaces → application/use_cases → domain
- **Tesztelés:** pytest, 1820+ teszt, 80% coverage
- **CLI belépő:** gyökér `main.py` (daily, weekly, monthly, walk-forward, train-rl, validate, governance)
- **Figyelmeztetés:** `app/main.py` egy régi, párhuzamos CLI – ne ezt használd!

## Ismert kritikus problémák (kód-alapú elemzés, 2026-04-02 – mind javítva 2026-04-07)
> Teljes lista: `docs/KNOWN_ISSUES.md`

Minden korábban listázott 🔴/🟠 probléma **javítva van** a kódbázisban:
- ✅ ADX valódi Wilder DM+/DM- implementáció – `app/core/indicators/trend.py`
- ✅ PaperExecution SELL fix – `PaperPosition(**v)` konverzió a `_load_latest_state()`-ben
- ✅ Commission levonás BUY/SELL esetén – `TRANSACTION_FEE_PCT` alkalmazva
- ✅ Governance FrozenInstanceError – `os.environ["PIPELINE_AUDIT_MODE"]` via env var
- ✅ RL end date – `end = end or str(date.today())` dinamikus
- ✅ ENABLE_RL=false log warning – `RunMonthlyRetrainingUseCase`-ben
- ✅ RSI/ATR Wilder EMA simítás – mindkét indikátor helyes implementációval

**Aktív figyelési pontok:**
- 🟡 `ADMIN_API_KEY` és `SECRET_KEY` default értékek – production deploymentnél kötelező felülírni (`.env.example` elérhető a gyökérben)
- 🟡 SQLite WAL mode – bekapcsolva (2026-04-07), párhuzamos futás biztonságosabb
- 🟡 `EXECUTION_MODE=live` – `NoopExecutionEngine`-re mutat, valódi live execution nincs implementálva

## Könyvtárstruktúra

```
app/
  analysis/         - teljesítmény- és megbízhatóság-elemzés
  application/      - use case-ek, pipeline orchestráció
  backtesting/      - historikus szimuláció
  bootstrap/        - dependency injection container
  config/           - settings, environment vars
  core/             - döntési logikát tartalmaz (NEM tiszta domain entitások – dict alapú Decision/Portfolio)
  data/             - adatbetöltő (OHLCV, yfinance)
  data_access/      - SQLite repók (DataManager)
  decision/         - döntési politika, ensemble modellek
  governance/       - quant validáció és riportolás
  indicators/       - technikai indikátorok (SMA, RSI, MACD, BB, ATR)
  infrastructure/   - logger, scheduler, error handling
  interfaces/       - CLI adapters, compat réteg
  models/           - RL model wrapperek (DQN, PPO)
  notifications/    - email/értesítés
  optimization/     - walk-forward, DEAP genetikus optimalizáció
  reporting/        - riport generálás
  services/         - domain service-ek
  ui/               - Flask admin API
  validation/       - Phase 5 & 6 validáció
```

## Kódolási konvenciók

### Általános szabályok
- **Típusjelölések kötelezők** minden public függvényen és metóduson
- **Result pattern:** `ok(value)` / `error(message)` a `app/application/use_cases/result.py`-ból – ne dobj kivételt üzleti logikában
- **Bootstrap container** keresztül injektáld a függőségeket (`build_application()`)
- **Ne módosítsd** a `app/interfaces/compat/` könyvtárat – ez a kompatibilitási réteg
- **Naplózás:** `setup_logger(__name__)` az `app/infrastructure/logger`-ből

### Tesztelési szabályok
- Minden új feature-höz írj unit tesztet `tests/` alá
- Fixture-ok: `tests/conftest.py` – `test_db`, `sample_ohlcv`, `mock_config`
- Ne használj éles yfinance hívást tesztekben – mockold
- Futtatás: `pytest` vagy `python scripts/run_all_tests.py`

### Adatbázis
- SQLite, a `DataManager` osztályon keresztül kezelendő
- Közvetlen SQL string helyett a réglévő CRUD metódusokat használd
- Migráció: meglévő sémát ne törd el visszafelé inkompatibilis módon

### RL modellek
- DQN és PPO modellek stable-baselines3-mal
- Modellfájlok: `models/` könyvtár, `.zip` + `.meta.json`
- Betanítás: `python main.py train-rl <TICKER>`

## Fontos CLI parancsok

```bash
# Napi pipeline
python main.py daily
python main.py daily --ticker VOO --dry-run

# Backtesting / validáció
python main.py run-paper-history --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
python main.py validate --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31

# Governance
python main.py governance --mode full

# Tesztek
pytest
python scripts/run_all_tests.py
python scripts/run_tests_with_report.py --skip-tests --with-validation --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31

# RL betanítás
python main.py train-rl VOO
python main.py walk-forward VOO
```

## Amit SOHA ne csinálj
- Ne írd felül a `app/interfaces/compat/` fájlokat
- Ne adj hozzá közvetlen yfinance hívást a domain réteghez (csak `app/data/` réteget keresztül)
- Ne törj el meglévő teszteket refaktoráláskor
- Ne commitolj éles API kulcsot vagy `.env` értéket
- Ne generálj alternatív architektúrát – a Clean Architecture rétegezést tartsd meg

## Környezeti változók
- `DRY_RUN=true/false` – mellékhatások ki/be
- `LOGGING_LEVEL=DEBUG/INFO/WARNING`
- `ADMIN_API_KEY` – Flask admin végpontok kulcsa
- `ENABLE_RL=true/false` – RL retraining engedélyezése (monthly pipeline, default: **false**)
- `ENABLE_RELIABILITY=true/false` – weekly reliability scoring (default: false)
- `PIPELINE_AUDIT_MODE=true/false` – governance audit mód (frozen Settings workaround)
