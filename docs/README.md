Recommended: Run governance/validation via main.py:

	python main.py governance --mode diagnostics
	python main.py governance --mode validation
	python main.py governance --mode full

> **Früssítés (2026-04-07):** Sprint 13 lezárva. A 2026-04-02-én azonosított összes kritikus hiba (ADX stub, PaperExecution SELL bug, commission hiány, Governance FrozenInstanceError, RSI/ATR Wilder EMA, RL end date) javítva. SQLite WAL mode bekapcsolva. Részletek: [KNOWN_ISSUES.md](KNOWN_ISSUES.md), [SPRINTS.md](SPRINTS.md).  
> **Früssítés (2026-04-09):** Sprint 14 **LEZÁRVA** (18 probléma implementálva). Q6 safety filter, Q2 equity allokáció, Q1 normalizált fitness, Q3 half-Kelly, Q5 MAX_MODEL_AGE_DAYS, B1/B2/B3 architektúra javítások, A1–A3 deprecation, D1–D5 docs. Részletek: [SPRINT14_PLAN.md](SPRINT14_PLAN.md).

# Tozsde Trading System

## English

### Overview
Tozsde is a Python trading system that runs a daily decision pipeline, records auditable decisions and outcomes in SQLite, and provides backtesting, historical paper runs, and validation tooling (Phase 5 and Phase 6). It supports paper execution, model ensembles, position sizing, and reliability analysis, with reporting and operational tooling for monitoring and maintenance.

### Feature Map (Detailed – kód-alapú, 2026-04-15)
- **Daily pipeline:** `load_data(180 nap)` → `prepare_df()` (10 indikátor) → `RLModelEnsembleRunner` (top 3 modell, WF-súlyozott szavazás) → `SafetyRuleEngine` (cooldown/drawdown/VIX/bear_market) → `allocate_capital()` (volatility + correlation) → `PaperExecutionEngine` (next_open ár) → DB + email
- **Champion-Challenger Shadow Pipeline (S23):** New RL models run in shadow mode first, in parallel with the champion. Shadow evaluation logs decisions and, after a configurable period, automatically promotes the challenger if its Sharpe ratio exceeds the champion's by a threshold. Shadow status and promotion recommendation are available via API and on the dashboard. The public dashboard displays a live-updating shadow summary block, powered by the `/shadow-summary` endpoint (public, no auth), showing champion/challenger status, promotion recommendation, evaluation days, and Sharpe ratios.
- **Paper execution:** portfolio state (cash + positions JSON) + outcomes DB-be mentve. Commission (`TRANSACTION_FEE_PCT`) levonva BUY és SELL esetén.
- **Historical paper runner:** deterministic backfills for a date range; fallback HOLD decisions if no RL models are present. Idempotent (skip if date already in DB).
- **Walk-forward:** DEAP GA (ngen=30, pop=50) → OOS fold értékelés → `production_score = 0.4*sharpe + 0.2*stability + 0.2*robustness + 0.2*(1-mxdd)`
- **RL training:** DQN + PPO (MlpPolicy, net_arch=[64,32]) → `ModelPromotionGate` → `.zip` + `.meta.json`. End dátum dinamikus (`str(date.today())`).
- **Validation:** Phase5 = DecisionQualityAnalyzer + ConfidenceCalibrator + WalkForwardStabilityAnalyzer + SafetyStressTester
- **Governance:** subprocess-ként fut `quant_runner.py` – pytest + diagnostics + 13 validáció + 10-tételes checklist → `reports/<timestamp>/`
- **Ops tooling:** health checks, backups, error reporting, cron scheduling, log management.

> ✅ **Sprint 13 (2026-04-07):** ADX, RSI, ATR helyesen implementálva (Wilder-féle). Minden kritikus hiba javítva. Részletek: [KNOWN_ISSUES.md](KNOWN_ISSUES.md)

### Quick Start (Windows IDE)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Note: initialize.sh is not used on Windows. Use the commands above in your IDE terminal.

### CLI Usage (Project Root)
```bash
python main.py daily
python main.py daily --ticker VOO
python main.py weekly
python main.py monthly
python main.py walk-forward VOO
python main.py train-rl VOO
python main.py run-paper-history --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
python main.py validate --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
```

### Data and Storage
- SQLite persists decisions, outcomes, portfolio state, and validation metrics.
- Market data is loaded via the data loader and stored as OHLCV.
- Decision history captures model votes, audits, and position sizing for traceability.

### Validation and Reporting
- Phase 5 aggregates decision quality, calibration, walk-forward stability, and safety stress.
- Phase 6 checks effectiveness, position sizing monotonicity, model trust, reward shaping, and promotion gates.
- Append validation results to the test report:

```bash
python scripts/run_tests_with_report.py --skip-tests --with-validation --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
```

### Governance Runner
**Ajánlott belépési pont** (DI-kompatibilis, `.env` automatikusan betöltődik):

```bash
python main.py governance --mode diagnostics
python main.py governance --mode validation
python main.py governance --mode full
```

> **Megjegyzés:** A közvetlen `python app/governance/quant_runner.py --mode full` hívás **csak debugging célra** ajánlott,  
> mert DI bypass-szal fut (nem a bootstrap container settings-ét használja).  
> Használat előtt győződj meg róla, hogy a `.env` fájl a gyöér könyvtárban van.

Riportok: `reports/<timestamp>/` – tartalmaz: summary, validation, diagnostics, tests, checklist, run log.

### Admin & Public API (Selected Endpoints)
Admin endpoints require the X-Admin-Key header (see Config.ADMIN_API_KEY).

- GET /admin/health
- GET /admin/performance/summary?days=30
- GET /admin/performance/drawdown?days=90
- GET /admin/performance/rolling?days=90&window=30
- GET /admin/errors/summary
- GET /admin/capital/status
- GET /shadow-summary *(public, no auth)* – JSON summary of current shadow evaluation (champion/challenger, Sharpe, days, promotion recommendation). Used by the live-refresh dashboard block.

### Tests
- Latest test status and coverage: docs/testing/TEST_STATUS_REPORT.md
- Run locally:

```bash
pytest
```

- One-command full suite:

```bash
python scripts/run_all_tests.py
```

### Project Layout (Current)
- app/: application modules (decision, backtesting, analysis, data access, services)
- scripts/: developer utilities and reporting helpers
- tests/: pytest suite
- docs/: documentation and test reports

### Documentation
- docs/INDEX.md for navigation
- docs/SPRINTS.md for sprint history and architecture narrative
- docs/TROUBLESHOOTING_GUIDE.md for operational issues
- docs/deployment for Raspberry Pi setup
- docs/validation_framework.md for quant validation and governance

### CI workflows (GitHub Actions)
- .github/workflows/phase6_check.yml: runs Phase 5 + Phase 6 validation in CI.
	- Usage: GitHub -> Actions -> "phase6-check" -> Run workflow.
- .github/workflows/train_models.yml: model training (minimal/full) when required.
	- Usage: GitHub -> Actions -> "train-models" -> Run workflow (mode minimal or full).

Note: deploy_rpi.sh supports optional RL training and optional RL cron via environment variables.

### Notes
- Historical paper runner uses a fallback HOLD decision if no RL model files are present.
- Validation depends on outcomes being recorded; without outcomes, effectiveness and trust metrics report no_data.
- `ENABLE_RL` defaults to `false` – monthly retraining will NOT update RL models unless explicitly set.
- `app/main.py` is a legacy CLI entry point; the active entry is the root `main.py`.
- Known issues and theoretical problems: see [docs/KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## Magyar

### Attekintes
A Tozsde egy Python alapu kereskedesi rendszer, amely napi dontesi pipeline-t futtat, auditalhato donteseket es outcome-okat ment SQLite-ba, valamint backtestinget, historikus paper futtatast es validacios toolingot ad (Phase 5 es Phase 6). Tamogatja a paper vegrehajtast, model ensemble-t, poziciomeretezest es megbizhatosag-elemzest, monitorozasi es karbantartasi eszkozokkel.

### Funkciótérkép (részletes – kód-alapú, 2026-04-15)
- **Napi pipeline:** `load_data(180 nap)` → `prepare_df()` (10 indikátor) → `RLModelEnsembleRunner` (top 3 modell, WF-súlyozott szavazás) → `SafetyRuleEngine` (cooldown/drawdown/VIX/bear_market) → `allocate_capital()` → `PaperExecutionEngine` (next_open ár) → DB + email
- **Champion-Challenger Shadow Pipeline (S23):** Az új RL modellek először shadow módban futnak, párhuzamosan a champion modellel. A shadow értékelés naplózza a döntéseket, és automatikus promóció történik, ha a challenger teljesítménye (Sharpe) meghaladja a champion-ét (küszöb + nap limit). A shadow státusz és promóciós javaslat elérhető API-n és dashboardon. A publikus dashboardon egy élőben frissülő shadow summary blokk jelenik meg, amely a /shadow-summary endpointot hívja meg JavaScripten keresztül, és mutatja a champion/challenger státuszt, promóciós javaslatot, értékelési napokat és Sharpe-okat.
- **Paper execution:** portfolio state (cash + positions JSON) + outcomes DB-be mentve. Commission (`TRANSACTION_FEE_PCT`) levonva BUY és SELL esetén.
- **Historikus paper runner:** determinisztikus visszatöltés; fallback HOLD döntés, ha nincs RL modell. Idempotens (skip, ha a dátum már DB-ben van).
- **Walk-forward:** DEAP GA (ngen=30, pop=50) → OOS fold értékelés → production_score
- **RL tanulás:** DQN + PPO (MlpPolicy) → ModelPromotionGate → .zip + .meta.json. ⚠️ ENABLE_RL=false default – havonta NEM frissül automatikusan.
- **Validáció:** Phase 5 = DecisionQualityAnalyzer + ConfidenceCalibrator + WalkForwardStabilityAnalyzer + SafetyStressTester
- **Governance:** subprocess-ként fut quant_runner.py – pytest + diagnostics + 13 validáció + 10 checklist → reports/<timestamp>/
- **Ops tooling:** health check, backup, error reporting, cron ütemezés, log menedzsment.

> ⚠️ **Ismert korlatozasok:** ADX indikator stub (konstans ~30.0), RSI/ATR nem Wilder-fele standard. Reszletek: [KNOWN_ISSUES.md](KNOWN_ISSUES.md)

### Gyors Start (Windows IDE)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Megjegyzes: initialize.sh nem szukseges Windows alatt. Hasznald a fenti parancsokat az IDE terminalban.

### CLI Hasznalat (projekttoba)
```bash
python main.py daily
python main.py daily --ticker VOO
python main.py weekly
python main.py monthly
python main.py walk-forward VOO
python main.py train-rl VOO
python main.py run-paper-history --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
python main.py validate --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
```

### Adat es tarolas
- SQLite tarolja a donteseket, outcome-okat, portfolio state-et es validacios metrikakat.
- Piaci adatok a data loaderen keresztul jonnek, OHLCV-kent mentve.
- Decision history tartalmazza a model vote-okat, auditot es poziciomeretezest.

### Validacio es riport
- Phase 5: dontesi minoseg, kalibracio, WF stabilitas, safety stress.
- Phase 6: hatekonysag, poziciomeretezes monotonitas, model trust, reward shaping, promotion gate.
- Validacio beirasa a teszt riportba:

```bash
python scripts/run_tests_with_report.py --skip-tests --with-validation --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
```

### Admin & Publikus API (kiemelt endpointok)
Az admin endpointokhoz X-Admin-Key header szükséges (Config.ADMIN_API_KEY).

- GET /admin/health
- GET /admin/performance/summary?days=30
- GET /admin/performance/drawdown?days=90
- GET /admin/performance/rolling?days=90&window=30
- GET /admin/errors/summary
- GET /admin/capital/status
- GET /shadow-summary *(publikus, nincs auth)* – JSON summary az aktuális shadow értékelésről (champion/challenger, Sharpe, napok, promóciós javaslat). Ezt használja a dashboard élő shadow summary blokkja.

### Tesztek
- Legfrissebb teszt statusz es coverage: docs/testing/TEST_STATUS_REPORT.md
- Lokalis futtatas:

```bash
pytest
```

- Teljes suite:

```bash
python scripts/run_all_tests.py
```

### Projekt szerkezet (jelenlegi)
- app/: alkalmazas modulok (decision, backtesting, analysis, data access, services)
- scripts/: fejlesztoi segedeszkozok
- tests/: pytest suite
- docs/: dokumentacio es teszt riportok

### Dokumentáció
- docs/INDEX.md navigáció
- docs/SPRINTS.md sprint történet és architektúra
- docs/ECONOMIC_EFFICIENCY_ROADMAP.md feature roadmap (S19–S23, shadow pipeline részletekkel)
- docs/TROUBLESHOOTING_GUIDE.md hibakeresés
- docs/deployment Raspberry Pi telepítés
- docs/deployment Raspberry Pi telepítés

### Megjegyzesek
- A historikus paper runner fallback HOLD dontest ad, ha nincs RL modell.
- Validacio outcome-ok nelkul no_data-t ad az effectiveness/trust metrikakra.
- `ENABLE_RL` alaperteke `false` – havilag NEM frissulnek az RL modellek, ha nincs be env var beallitva.
- Az `app/main.py` egy regi CLI belepesi pont; az aktiv belep a gyoker `main.py`.
- Ismert problemak es elmeleti hibak: lasd [docs/KNOWN_ISSUES.md](KNOWN_ISSUES.md).
