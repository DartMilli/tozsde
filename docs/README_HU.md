# ToZsDE Trading System - Dokumentáció

**Státusz:** ✅ Sprint 11c Maintenance Complete | 1070 passing teszt | 98% coverage

---

## 🚀 Gyors Start

**Új vagy itt?** Olvass [INDEX.md](./INDEX.md)-et a teljes navigációért.

**Meg akarod érteni a projektet?** Kezd az [SPRINTS.md](./SPRINTS.md)-el.

**Teljes funkcionalitás (felhasználói + fejlesztői):** [SZOFTVER_FUNKCIONALITAS_HU.md](./SZOFTVER_FUNKCIONALITAS_HU.md)

**Segítség kell?** Nézd meg a [FAQ.md](./FAQ.md) vagy [TROUBLESHOOTING_GUIDE.md](./TROUBLESHOOTING_GUIDE.md).

**Raspberry Pi-ra telepítesz?** Kövesd az [deployment/RASPBERRY_PI_SETUP_GUIDE_HU.md](./deployment/RASPBERRY_PI_SETUP_GUIDE_HU.md).

**Gyors egészség‑ellenőrzés?** Futtasd a smoke tesztet:

```
python -m app.scripts.smoke_test
```

---

## 📂 Dokumentáció Szerkezete

```
docs/
├── INDEX.md                             ◄──── Navigációs hub
├── README.md, README_HU.md              ◄──── Projekt áttekintés
├── SPRINTS.md                           ◄──── Sprint történet (egy helyen)
├── FAQ.md                               ◄──── GYIK
├── TROUBLESHOOTING_GUIDE.md             ◄──── Hibaelhárítás
│
├── deployment/                          ◄──── Raspberry Pi telepítés
│   ├── RASPBERRY_PI_SETUP_GUIDE_HU.md
│   ├── RASPBERRY_PI_SETUP_GUIDE.md
│   └── DEPLOYMENT_VERIFICATION_CHECKLIST.md
│
└── testing/                             ◄──── Teszt eredmények
    └── TEST_STATUS_REPORT.md
```

---

## 📊 Projekt Státusza

- ✅ **1070 passing teszt** (legutóbbi futás)
- ✅ **98% kód lefedettség**
- ✅ **Production-ready**
- ✅ **Sprint 1-10 kész + Sprint 11c maintenance**

**Cleanup elvégezve (2026-02-02):**
- ✅ Eltávolítva: START_HERE.txt, CLEANUP_SUMMARY.md, 04_infrastructure/ (üres)
- ✅ Összevonva: Összes sprint terv → SPRINTS.md
- ✅ Eltávolítva: 02_implementation/*.md (6 fájl összevonva)
- ✅ Eredmény: **4 alapvető dokumentációs fájl** (15+-ről lecsökkentve)

**Sprint 11c Maintenance (2026-02-03):**
- ✅ PerformanceAnalytics modul (500+ sor)
- ✅ ErrorReporter komponens (580+ sor)
- ✅ AdminDashboard bővítés (12 REST API endpoint)
- ✅ 1070 passing teszt (legutóbbi futás)
- Health monitoring

---

## 📁 Projekt Szerkezete

```
tozsde_webapp/
├── app/                          [Alkalmazás Kód]
│   ├── data_access/             [SQLite adatszint]
│   ├── decision/                [Kereskedési Logika]
│   ├── backtesting/             [Walk-Forward Tesztelés]
│   ├── optimization/            [GA Optimizer]
│   ├── ui/                      [Flask API]
│   └── ... (további modulok)
├── tests/                        [1070 passing]
├── docs/                         [Dokumentáció]
├── deploy_rpi.sh                 [ONE-CLICK Telepítő Script]
├── requirements.txt              [Python Függőségek]
└── pytest.ini                    [Teszt Konfiguráció]
```

---

## 🎯 Következő Lépések

1. **Olvasd:** [deployment/RASPBERRY_PI_SETUP_GUIDE_HU.md](./deployment/RASPBERRY_PI_SETUP_GUIDE_HU.md)
2. **Készítsd el:** Raspberry Pi 4/5 + SD-kártya + USB tápegység
3. **Telepítsd:** `bash deploy_rpi.sh`
4. **Ellenőrizd:** `curl http://raspberrypi.local:5000/api/health`
5. **Futtasd:** [deployment/DEPLOYMENT_VERIFICATION_CHECKLIST.md](./deployment/DEPLOYMENT_VERIFICATION_CHECKLIST.md)
6. **Monitorozz:** Naplók 6:00 AM után (első napi futás)

System LIVE lesz és kereskedni fog! 🚀

---

## 📚 Dokumentáció Mappa Index

| Mappa | Fájlok | Leírás |
|-------|--------|--------|
| **deployment/** | 4 | Raspberry Pi telepítési útmutatók + ellenőrzőlista |
| **testing/** | 1 | Teszt státusz | 
| **archive/** | - | Történeti dokumentumok |

---

## ✨ Legutóbbi Frissítések

- **[SPRINTS.md](./SPRINTS.md)** - Sprint 1-9 teljes történet (Sprint 9 Product Hardening hozzáadva)
- **[TEST_STATUS_REPORT.md](./testing/TEST_STATUS_REPORT.md)** - legfrissebb státusz
- **[RASPBERRY_PI_SETUP_GUIDE_HU.md](./deployment/RASPBERRY_PI_SETUP_GUIDE_HU.md)** - Magyar Teljes Telepítési Útmutató
- **[DEPLOYMENT_VERIFICATION_CHECKLIST.md](./deployment/DEPLOYMENT_VERIFICATION_CHECKLIST.md)** - Telepítés utáni ellenőrzőlista

---

**Státusz:** ✅ Sprint 11c Maintenance kész | ⏳ Hardver telepítés pending
**Cél Platform:** 🍓 Raspberry Pi 4/5 (64-bit ARM)
**Teszt Lefedettség:** ✅ 98% (legutóbbi futás)

Jó kereskedést! 🎉
