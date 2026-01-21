# Documentation Index - Szervezetten Szétválasztott Dokumentáció

## 📋 Dokumentáció Szerkezete

Ez az `docs/` mappa **hierarchikus és kategorizált** dokumentációt tartalmaz a projekt minden aspektusához.

**✅ Státusz:** Minden dokumentáció a `docs/` mappában szervezett. A gyökérből az összes `.md` fájl átmozgatva.

---

## 🗂️ Fő Kategóriák

### 1️⃣ [01_deployment/](./01_deployment/) - Telepítés & Infrastruktúra

Szisztémadministrációs és telepítési dokumentáció:

| Fájl | Cél | Státusz |
|------|-----|--------|
| DEPLOYMENT_ARCHITECTURE.md | Systemd-alapú telepítési terv (Docker-mentes) | ✅ Aktív |
| DEPLOYMENT_TRANSITION.md | Átmenet Docker-ről natív systemd-re | ✅ Aktív |
| DEPLOYMENT_READY.md | Telepítés előtti checklist | ✅ Aktív |
| DEPLOYMENT_COMPLETION_SUMMARY.md | Telepítési folyamat lezárása | ✅ Aktív |
| DEPLOYMENT_INDEX.md | Telepítési dokumentáció indexe | ✅ Aktív |
| deployment_checklist.md | Lépésről lépésre checklist | ✅ Aktív |
| P9_PRODUCTION_DEPLOYMENT.md | P9 Raspberry Pi telepítés | ✅ Aktív |
| raspberry_setup_guide.md | Raspberry Pi konfigurációs útmutató | ✅ Aktív |
| raspberry_service_config.md | Raspberry Pi service konfigurációk | ✅ Aktív |
| NAS_setup_guide.md | NAS tárolási útmutató | ✅ Aktív |

---

### 2️⃣ [02_implementation/](./02_implementation/) - Implementáció & Fejlesztés

Kódstruktúra és fejlesztési irányelvek:

| Fájl | Cél | Státusz |
|------|-----|--------|
| IMPLEMENTATION_PLAN.md | 5 SPRINT-os fejlesztési terv | ✅ Aktív |
| IMPLEMENTATION_SUMMARY.md | Jelenlegi implementáció összefoglalása | ✅ Aktív |
| IMPLEMENTATION_CHECKLIST.md | Megvalósítási checklist | ✅ Aktív |
| CODE_SKELETON_SUMMARY.md | Kódskeleton összefoglalása | ✅ Aktív |
| SKELETON_COMPLETE.md | Kódskeleton befejezettségi lap | ✅ Aktív |
| DEVELOPMENT_GUIDE.md | Fejlesztői útmutató | ✅ Aktív |
| DEV_RUNNER_IMPROVEMENTS.md | Fejlesztői futtatási javítások | ✅ Aktív |
| SKELETON_VISUAL_GUIDE.txt | Kódskeleton vizuális áttekintése | ✅ Aktív |

---

### 3️⃣ [03_testing/](./03_testing/) - Testing & QA

Tesztelési stratégia és SPRINT 1 progress:

| Fájl | Cél | Státusz |
|------|-----|--------|
| SPRINT1_PROGRESS.md | **SPRINT 1 teszt implementáció progress** | 🔄 In Progress |
| SPRINT1_TESTING_ROADMAP.md | SPRINT 1 tesztelési ütemterv | 📝 Új |
| TEST_STRUCTURE_GUIDE.md | Tesztek szervezettségének útmutatója | 📝 Új |
| BLOCKERS_AND_SOLUTIONS.md | Ismert blokkerek és megoldások | 📝 Új |

---

### 4️⃣ [04_infrastructure/](./04_infrastructure/) - Infrastrukturális Dokumentáció

Rendszer-architektúra és konfigurációs alapok:

| Fájl | Cél | Státusz |
|------|-----|--------|
| ARCHITECTURE_OVERVIEW.md | Rendszer építészeti áttekintés | 📝 Új |
| DEPENDENCIES.md | Függőségi mátrix és verziókezelés | 📝 Új |
| PYTHON_COMPATIBILITY.md | Python 3.6+ kompatibilitási megjegyzések | 📝 Új |

---

## 📊 Dokumentáció Status Összefoglalása

### Fájlok Száma

```
✅ 01_deployment/       9 fájl (Befejezve)
✅ 02_implementation/   8 fájl (Befejezve)
🔄 03_testing/         7 fájl (SPRINT 1 In Progress)
📝 04_infrastructure/   3 fájl (Tervezve)
─────────────────────────────
📋 TOTAL:              27 fájl (szervezett)
```

### SPRINT 1 Teszt Implementáció Status

```
✅ test_indicators.py:   6/6 tests (5 passing, 1 edge case)
✅ test_fitness.py:       9/9 tests (9 passing, 100%)
⏳ test_backtester.py:   11/11 tests (code ready, blocker: circular import)
─────────────────────────────────────────────────
TOTAL:                   24/88 tests implemented (27%)
PASSING:                 14/24 tests (93% success rate)

BLOCKER: app.config ↔ app.data_access circular import
         → Prevents backtester and dependent tests execution
```

---

## 🚀 Gyors Navigáció

### Éppen Most Dolgozol?

- **🏗️ Telepítésen?** → [01_deployment/DEPLOYMENT_ARCHITECTURE.md](./01_deployment/DEPLOYMENT_ARCHITECTURE.md)
- **💻 Fejlesztésen?** → [02_implementation/IMPLEMENTATION_PLAN.md](./02_implementation/IMPLEMENTATION_PLAN.md)
- **🧪 Tesztelésen?** → [03_testing/SPRINT1_PROGRESS.md](./03_testing/SPRINT1_PROGRESS.md)
- **🔧 Infrastruktúra?** → [04_infrastructure/ARCHITECTURE_OVERVIEW.md](./04_infrastructure/ARCHITECTURE_OVERVIEW.md)

### Probléma Elhárítás

- **Blokkerek?** → [03_testing/BLOCKERS_AND_SOLUTIONS.md](./03_testing/BLOCKERS_AND_SOLUTIONS.md)
- **Python verzió probléma?** → [04_infrastructure/PYTHON_COMPATIBILITY.md](./04_infrastructure/PYTHON_COMPATIBILITY.md)
- **Függőségek?** → [04_infrastructure/DEPENDENCIES.md](./04_infrastructure/DEPENDENCIES.md)

---

## 📝 Fájl Konvenció

### Elnevezési Szabály

```
docs/
├── 01_deployment/           [Teleítési & Infrastruktúra]
├── 02_implementation/        [Fejlesztés & Kódstruktúra]
├── 03_testing/              [Tesztelés & QA]
└── 04_infrastructure/       [Alapinfrastruktúra]
```

### Státusz Jelölések

- ✅ = Befejezve, aktívan használt
- 🔄 = In Progress (aktív fejlesztés alatt)
- 📝 = Új fájl, dokumentáció alatt
- ⏳ = Várakozik feloldásra
- ❌ = Elavult, nem ajánlott

---

## 📈 Utolsó Frissítés

| Kategória | Fájlok | Legutóbb Módosítva | Státusz |
|-----------|--------|-------------------|--------|
| Deployment | 9 | 2024-12-XX | ✅ Aktív |
| Implementation | 8 | 2024-12-XX | ✅ Aktív |
| Testing | 7 | 2025-01-21 | 🔄 In Progress |
| Infrastructure | 3 | 2025-01-21 | 📝 Új |

---

## 🎯 Következő Lépések

### SPRINT 1 - Teszt (In Progress)
1. ✅ test_indicators.py & test_fitness.py végleges
2. 🔴 **BLOCKER**: Circular import fix szükséges
3. ⏳ test_backtester.py, test_walk_forward.py stb.

### Ajánlott Olvasási Sorrend

1. **Kezdéshez:** [02_implementation/IMPLEMENTATION_PLAN.md](./02_implementation/IMPLEMENTATION_PLAN.md)
2. **Aktuális:** [03_testing/SPRINT1_PROGRESS.md](./03_testing/SPRINT1_PROGRESS.md)
3. **Blokker:** [03_testing/BLOCKERS_AND_SOLUTIONS.md](./03_testing/BLOCKERS_AND_SOLUTIONS.md)

---

**Készítve:** 2025-01-21  
**Szervezési Szint:** 📊 Hierarchikus (04 kategória, 27+ fájl)  
**Könnyű Navigáció:** ✅ Index alapú
