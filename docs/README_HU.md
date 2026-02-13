# ToZsDE Trading System - Dokumentáció

**Státusz:** ✅ Sprint 12 Stabilizáció + Validáció Complete | 1070 passing teszt | 98% coverage

---

## 🚀 Gyors Start

**Új vagy itt?** Olvass [INDEX.md](./INDEX.md)-et a teljes navigációért.

**Meg akarod érteni a projektet?** Kezd az [SPRINTS.md](./SPRINTS.md)-el.

**Architektúra review:** [ARCH_REVIEW.md](./ARCH_REVIEW.md)

**Teljes funkcionalitás (felhasználói + fejlesztői):** [SZOFTVER_FUNKCIONALITAS_HU.md](./SZOFTVER_FUNKCIONALITAS_HU.md)

**Segítség kell?** Nézd meg a [FAQ.md](./FAQ.md) vagy [TROUBLESHOOTING_GUIDE.md](./TROUBLESHOOTING_GUIDE.md).

**Raspberry Pi-ra telepítesz?** Kövesd az [deployment/RASPBERRY_PI_SETUP_GUIDE_HU.md](./deployment/RASPBERRY_PI_SETUP_GUIDE_HU.md).

# Tozsde Trading System - Dokumentacio (HU + EN)

## Magyar

### Attekintes
A Tozsde egy Python alapú kereskedési rendszer, amely napi döntési pipeline-t futtat, auditálható döntéseket és outcome-okat ment SQLite-ba, valamint backtestinget, historikus paper futtatást és validációs toolingot ad (Phase 5 és Phase 6). Támogatja a paper végrehajtást, model ensemble-t, pozícióméretezést és megbízhatóság-elemzést, monitorozási és karbantartási eszközökkel.

### Funkciok es modulok
- Napi pipeline: adatbetöltés, jelgenerálás, policy, allokáció, mentés, értesítés.
- Paper execution: portfolio state és outcome számítás.
- Historikus paper runner: determinisztikus visszatöltés; fallback HOLD döntés, ha nincs RL modell.
- Validáció: döntési minőség, kalibráció, walk-forward stabilitás, safety stress, Phase 6 ellenőrzések.
- Backtesting és audit: replay, audit nyomvonalak, reward shaping elemzés, riportok.
- Ops tooling: health check, backup, error reporting, cron ütemezés, log menedzsment.

### Gyors Start
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

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

### Validacio es tesztek
- A validációs snapshot a teszt riportban jelenik meg.
- Futtatás:

```bash
python scripts/run_tests_with_report.py --skip-tests --with-validation --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
```

- Tesztek futtatasa:

```bash
pytest
```

- Egyparancsos teljes tesztfutas:

```bash
python scripts/run_all_tests.py
```

### Admin API (kiemelt endpointok)
Az admin endpointokhoz X-Admin-Key header szukseges (Config.ADMIN_API_KEY).

- GET /admin/health
- GET /admin/performance/summary?days=30
- GET /admin/performance/drawdown?days=90
- GET /admin/performance/rolling?days=90&window=30
- GET /admin/errors/summary
- GET /admin/capital/status

### Dokumentacio navigacio
- docs/INDEX.md navigacio
- docs/SPRINTS.md sprint tortenet es architektura
- docs/TROUBLESHOOTING_GUIDE.md hibakereses
- docs/deployment Raspberry Pi telepites
- docs/testing/TEST_STATUS_REPORT.md teszt statusz

### CI workflow-k (GitHub Actions)
- .github/workflows/phase6_check.yml: Phase 5 + Phase 6 validacio futtatasa CI-ben.
	- Hasznalat: GitHub -> Actions -> "phase6-check" -> Run workflow.
- .github/workflows/train_models.yml: Model training (minimal / full sweep) csak ha szukseges.
	- Hasznalat: GitHub -> Actions -> "train-models" -> Run workflow (mode/minimal vagy full).

Megjegyzes: a deploy_rpi.sh nem indit RL traininget; a Raspberry Pi cron csak daily/weekly/GA monthly futasokat kezel.

## English

### Overview
Tozsde is a Python trading system with a daily decision pipeline, auditable SQLite persistence, backtesting, historical paper runs, and Phase 5/6 validation. It supports paper execution, model ensembles, position sizing, and operational tooling.

### Quick Start
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### CLI Usage (Project Root)
```bash
python main.py daily
python main.py run-paper-history --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
python main.py validate --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
```

### Docs Map
- docs/README.md for the full bilingual overview
- docs/INDEX.md for navigation
- docs/SPRINTS.md for sprint history
- docs/TROUBLESHOOTING_GUIDE.md for operational fixes
- docs/deployment for Raspberry Pi setup
## 📚 Dokumentáció Mappa Index
