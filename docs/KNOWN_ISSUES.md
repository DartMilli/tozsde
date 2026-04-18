# Ismert problémák és elméleti hibák – Kód-alapú elemzés
**Elemzés dátuma:** 2026-04-02  
**Utoljára frissítve:** 2026-04-09 (Sprint 15 implementálva)  
**Módszer:** Teljes kódátvizsgálás – kizárólag forráskód alapján, dokumentáció figyelmen kívül hagyásával

> ✅ **2026-04-07:** Minden K/H/M/A kategóriás hiba javítva. SQLite WAL mode bekapcsolva.  
> ✅ **2026-04-09 – Sprint 14 LEZÁRVA:** Minden Q/A/B tételmszám implementálva.
> 
> **Javított problémák (Sprint 14):**
> - ✅ **Q6:** `_quality_label()` → `EnsembleQualityBucket` enum; safety filter CHAOTIC-ra is aktiválódik
> - ✅ **Q2:** `allocate_capital()` aktuális portfolio equity-t használ (nem hardcoded `INITIAL_CAPITAL`)
> - ✅ **Q1:** `fitness_single()` normalizált formula (net_profit/capital, max_dd/capital)
> - ✅ **Q3:** Kelly-kritérium half-Kelly dampening (0.5× szorzó, `KELLY_FRACTION_MULTIPLIER` env var)
> - ✅ **Q5:** `MAX_MODEL_AGE_DAYS` env var – elavult modellek kizárása az ensemble-ből
> - ✅ **Q8:** `BEAR_MARKET_LOOKBACK_DAYS` 400→250 napra csökkentve
> - ✅ **B2:** `MarketDataFetcher` cache thread-safe (`threading.Lock`)
> - ✅ **B3:** `_build_ticker_provider()` refactored, duplikáció megszüntetve
> - ✅ **A1:** `SafetyRuleEngine`/`RiskParityAllocator` fallback-deprecation warning és docstring
> - ✅ **A2:** `CapitalUtilizationOptimizer._init_db()` lazy inicializáció (`_ensure_db()`)
> - ✅ **A3:** `app/decision/` passthrough fájlok DEPRECATED docstring
> - ✅ **B1:** `RunDailyPipelineUseCase` lazy pipeline init
> 
> ✅ **2026-04-09 – Sprint 15 LEZÁRVA:** Codebase audit alapján azonosított open-item-ek implementálva.  
> 
> **Javított problémák (Sprint 15):**
> - ✅ **S15C-1:** `dh.confidence` column guard – `PRAGMA table_info` check, 5 pre-existing test hiba javítva
> - ✅ **S15C-2:** PyFolio 5 TODO comment eltávolítva (implementáció már megvolt)
> - ✅ **S15A-1:** `load_recent_outcomes()` docstring javítva
> - ✅ **S15A-2:** Phase7 production import migration (`audit_builder.py`, `admin_dashboard.py`)
> - ✅ **S15B-1:** `LiveExecutionEngine` broker-agnostic shell implementálva és bewired
> - ✅ **S15B-2:** `PortfolioRebalancer` bewired (`ENABLE_REBALANCER` flag, default: false)
> - ✅ **S15B-3:** `enforce_correlation_limits` bewired (`ENABLE_CORRELATION_LIMITS` flag, default: false)
> - ✅ **S15B-4:** ML/LSTM opcionális ensemble member (`ENABLE_ML_ENSEMBLE` flag, default: false)
> - ✅ **S15B-5:** 5 új Settings mező + env var (ENABLE_REBALANCER, REBALANCE_THRESHOLD, ENABLE_CORRELATION_LIMITS, MAX_CORRELATION, ENABLE_ML_ENSEMBLE)
> - ✅ **S15B-6:** `fetch_recent_outcomes` graceful fallback ha nincs outcomes tábla
> 
> **Aktív figyelési pontok:**
> - 🟡 `ADMIN_API_KEY`/`SECRET_KEY` production értékek kötelező felülírása (`.env.example` elérhető)
> - 🟡 `EXECUTION_MODE=live` + `BROKER_ADAPTER=noop` esetén nincs valódi order-routing; productionhoz `BROKER_ADAPTER=alpaca` (vagy egyéb adapter) szükséges
> - ✅ `build_settings()` fallback eltávolítva az `app/core/decision/` rétegből – S17 lezárva; S18-ban `warnings.warn` deprecation hozzáadva mind a 20 `app/decision/` shimhez
> - 🟢 **Phase7 DataManager singleton** (`test_phase7_contract_freeze.py`) – ✅ ZÖLD (2026-04-10 ellenőrizve)
>
> ✅ **2026-04-13 – Sprint 18 LEZÁRVA:** 1503→1820 teszt, 75%→80% coverage.
> - ✅ `app/decision/*.py` (20 fájl) – `warnings.warn` deprecation hozzáadva
> - ✅ `SqliteDecisionRepository` – 100% lefedettség
> - ✅ `DataManagerRepository` – 99% lefedettség
> - ✅ `app/core/analysis/decision_effectiveness.py` NameError javítva (`get_settings` → `build_settings` alias)
> - ✅ 8 db új tesztfájl hozzáadva (`test_sprint18_*.py`)

---

## 🔴 KRITIKUS PROBLÉMÁK

### K1 – ADX implementáció STUB (konstans érték) ✅ JAVÍTVA – 2026-04-02
**Fájl:** `app/core/indicators/trend.py` – `adx()` függvény  
**Leírás:** Az ADX indikátor **nem valódi számítást végez** – ha az adatsor eléri az `n >= period` feltételt, az utolsó elembe hardcoded `30.0` értéket ír (`plus=24.0`, `minus=6.0`). Ez egy befejezetlen stub implementáció.  
**Következmény:**
- A `TradingEnv` observation vectorban az `adx/100` feature **mindig ~0.3** → teljesen uninformativ RL feature
- Az RL modellek erre az értékre tanultak → ADX-alapú döntéseik érvénytelenek
- A backtester ADX-összefüggései is torzak
- Az ATR-alapú stop-loss (`entry - 2.0 * ATR`) megbízhatósága szintén kérdéses  
**Teendő:** Valódi Wilder-féle ADX implementáció szükséges (+DI/-DI-vel).

---

### K2 – PaperExecutionEngine: `positions` dict vs PaperPosition attribútum-hiba ✅ JAVÍTVA – 2026-04-02
**Fájl:** `app/services/paper_execution.py` – `_load_latest_state()` + `execute()`  
**Leírás:** A `portfolio_state.positions_json` JSON-oszlopból visszatöltött pozíciók **plain dict**-ek, de az `execute()` metódus `pos.qty` és `pos.entry_price` attribute accesszel kezeli őket – mintha `PaperPosition` objektumok lennének.  
**Következmény:** Minden SELL döntésnél `AttributeError: 'dict' object has no attribute 'qty'` – a SELL soha nem hajtódik végre helyesen, ha a state DB-ből lett visszatöltve (azaz multi-run esetén). Csak az első futtatáson (friss `positions={}`) nincs hiba.  
**Teendő:** A `_load_latest_state()` deserializáláskor `PaperPosition(**v)` konverziót kell alkalmazni.

---

### K3 – PaperExecutionEngine: nincs commission levonás ✅ JAVÍTVA – 2026-04-02
**Fájl:** `app/services/paper_execution.py`  
**Leírás:** A paper execution engine nem vonja le a tranzakciós jutalékot (`TRANSACTION_FEE_PCT`), noha a `TradingEnv.step()` pontosan számolja: `commission = qty * price * fee_pct`.  
**Következmény:** A paper P&L **szisztematikusan optimista** – reális kehreskedésben a jutalékok erodálják a hozamot. A backtester és a paper trader más feltételrendszeren dolgozik, a teljesítményösszehasonlítás félrevezető.  
**Teendő:** `proceeds -= commission` és `cash -= allocation_amount + commission` a BUY/SELL lépésekbe.

---

### K4 – `Settings` frozen dataclass + mutáció kísérlet ✅ JAVÍTVA – 2026-04-02
**Fájl:** `app/governance/quant_runner.py`  
**Leírás:** A `Settings` `@dataclass(frozen=True)` – immutable. A `quant_runner.py` azonban megpróbálja: `settings.pipeline_audit_mode = True`.  
**Következmény:** Futásidőben `dataclasses.FrozenInstanceError` keletkezik amikor `governance --mode full` fut `PIPELINE_AUDIT_MODE` nincs beállítva. A teljes governance futás összeomlik ahelyett, hogy továbblépne.  
**Teendő:** A `pipeline_audit_mode` flagot env var-ból kell olvasni, vagy `dataclasses.replace(settings, pipeline_audit_mode=True)` kell.

---

## 🟠 MAGAS SÚLYOSSÁGÚ PROBLÉMÁK

### H1 – RSI nem standard Wilder-féle implementáció ✅ JAVÍTVA – 2026-04-02
**Fájl:** `app/core/indicators/momentum.py` – `rsi()`  
**Leírás:** Az implementáció SMA convolutiont használ az up/down mozgások simítására, míg az iparági standard a **Wilder-féle exponenciális simítás** (EWMA, `alpha=1/period`).  
**Következmény:** Az RSI értékek szisztematikusan eltérnek minden más platformtól (TradingView, TA-Lib). ~14 periódusra a különbség ~3-8 pont lehet. Modell cross-validáció és összehasonlítás torzíthat.

---

### H2 – ATR nem standard Wilder-féle implementáció ✅ JAVÍTVA – 2026-04-02
**Fájl:** `app/core/indicators/volatility.py` – `atr()`  
**Leírás:** Az ATR simple rolling mean-t használ a True Range sorozaton, szemben a **Wilder-féle RMA (EMA alpha=1/period)** simítással.  
**Következmény:** Az ATR értékek kisimítottabbak, mint a standard – alacsonyabb volatilitás-becslés. Az ATR-alapú stop-loss és a position sizer torzult inputot kap.

---

### H3 – RL betanítás hardcoded `end="2025-06-30"` ✅ JAVÍTVA – 2026-04-02
**Fájl:** `app/models/rl_trainer.py` – `train_rl_agent()`  
**Leírás:** Az RL betanítás adatletöltési végdátuma hardcoded `"2025-06-30"` string.  
**Következmény:** 2026 januártól kezdve a modellek **8+ hónapos elavult adatokon** tanulnak; a legfrissebb piaci rezsim (volatilitás, trendek) kimarad a tanító készletből. A modellek "stale" állapotban vannak.  
**Teendő:** `end=str(date.today())` – dinamikus végdátum.

---

### H4 – `ENABLE_RL=false` default → havi retraining sosem fut ✅ JAVÍTVA – 2026-04-03
**Fájl:** `app/config/build_settings.py`  
**Leírás:** Az `ENABLE_RL` env var default értéke `"false"`. Ha nincs `.env` fájl vagy nincs benne `ENABLE_RL=true`, a `RunMonthlyRetrainingUseCase` **soha nem tanítja újra az RL modelleket** – pedig ez a havi pipeline elsődleges célja.  
**Következmény:** A rendszer frissítés nélkül működik. A live deployment esetén automatikusan nem frissülnek a modellek.

---

### H5 – `ensemble_quality` float vs string inkonzisztencia ✅ JAVÍTVA – 2026-04-03
**Fájl:** `app/core/decision/ensemble_aggregator.py` + `app/decision/recommender.py`  
**Leírás:** Az `aggregate_weighted_ensemble()` az `ensemble_quality`-t float értékként (`score_diff`) adja vissza, de a `build_recommendation()` string értéket vár (`"STABLE"`, `"CHAOTIC"`). A konverzió valahol implicit kell legyen – de ha ez eltörik, silent wrongdoing: a safety filter nem blokkol `"CHAOTIC"` esetén.  
**Teendő:** Explicit típusdeklaráció és egységes konverziós logika.

---

## 🟡 KÖZEPES SÚLYOSSÁGÚ PROBLÉMÁK

### M1 – `dry_run` elvész a walk-forward use case-ben ✅ JAVÍTVA – 2026-04-02
**Fájl:** `app/application/use_cases/run_walk_forward.py` + gyökér `main.py`  
**Leírás:** A `main.py` `run_walk_forward_manual()` wrapper elfogadja a `dry_run` paramétert, de **nem adja tovább** a `container.walk_forward.run()` hívásba – a flag teljesen figyelmen kívül marad.  
**Következmény:** `python main.py walk-forward VOO --dry-run` valódi adatírással és param-mentéssel fut.

---

### M2 – `DailyPipelineUseCase.run()` nem követ Result patternt ✅ JAVÍTVA – 2026-04-03
**Fájl:** `app/application/use_cases/daily_pipeline.py`  
**Leírás:** A belső `DailyPipelineUseCase.run()` plain dict-et ad vissza `{"completed": ..., "processed": ...}`, nem `UseCaseResult`-ot. A `RunDailyPipelineUseCase` javítja ezt, de az architektúra inkonzisztens.

---

### M3 – Kritikus pénzügyi paraméterek nem konfigurálhatók env var-ból ✅ JAVÍTVA – 2026-04-03
**Fájl:** `app/config/settings.py`  
**Leírás:** `INITIAL_CAPITAL=10000`, `RISK=0.02`, `TRANSACTION_FEE_PCT=0.001`, `CONFIDENCE_NO_TRADE_THRESHOLD=0.25`, `STRONG_CONFIDENCE_THRESHOLD=0.75` stb. **nem szerepelnek az env var mappingban** – csak forráskód módosítással változtathatók.  
**Következmény:** Live deployment esetén ezek a paraméterek rögzítve vannak; A/B tesztelés, risk profil váltás csak kóddal lehetséges.

---

### M4 – `app/main.py` és gyökér `main.py` párhuzamos, inkonzisztens CLI-k ✅ JAVÍTVA – 2026-04-03
**Fájl:** `app/main.py` (régi), gyökér `main.py` (aktív)  
**Leírás:** Az `app/main.py` egy régi belépési pont, eltérő parancsnevekkel (`daily-pipeline`, `validate-model`), hiányzik belőle a `weekly`, `monthly`, `governance`, `validate` parancs.  
**Következmény:** Fejlesztők vagy deployment scriptek tévedésből a régi CLI-t futtatják, meglepő viselkedéssel.

---

### M5 – `governance` subprocess dispatch – DI bypass ✅ JAVÍTVA – 2026-04-03
**Fájl:** gyökér `main.py`, `app/governance/quant_runner.py`  
**Leírás:** A `governance` parancs `subprocess.call()`-lal indítja a `quant_runner.py` scriptet, amely saját maga hívja `build_settings()`-t, nem a DI containertől kapja.  
**Javítás:** `RunGovernanceUseCase` use case osztály (`app/application/use_cases/run_governance.py`) – in-process hívja a `quant_runner` belső funkcióit, beinjektált `settings`-szel. `ApplicationContainer.governance` mező a bootstrap-ban.

---

### M6 – Domain entitások nincsenek formalizálva ✅ JAVÍTVA – 2026-04-03
**Fájl:** az egész `app/core/` és `app/decision/`  
**Leírás:** A `Decision`, `Portfolio`, `Trade`, `Position` objektumok **plain dict**-ek sémadefiníció nélkül – sem `dataclass`, sem `TypedDict`. Az implicit séma szétszórt a kódbázisban.  
**Javítás:** `app/domain/types.py` – TypedDict definíciók: `Decision`, `PolicyPayload`, `ModelVote`, `DailyCandidate`, `TradeItem`, `PortfolioState`, `WalkForwardSummary`. Kulcsfüggvények type annotációval frissítve.

---

## 🔵 ALACSONY SÚLYOSSÁGÚ PROBLÉMÁK

### A1 – MACD histogram hiányzik az observation vectorból ✅ JAVÍTVA – 2026-04-03
**Fájl:** `app/core/indicators/trend.py` – `macd()`, `app/models/trading_env.py`  
**Leírás:** A `macd()` visszatérési értéke `(line, signal)` – histogram nincs. A `TradingEnv` observation csak `MACD` és `MACD_SIGNAL` mezőket kap, histogram-alapú momentum signal nem elérhető az RL modellnek.  
**Javítás:** `macd()` 3-tuple-re bővítve `(line, signal, histogram)`. `analyzer.py` + `data_cleaner.py` + `model_trainer.py` frissítve – az RL observation vector 11 feature-re bővült.

---

### A2 – `ohlcv_repo` / `data_manager_repo` duplikáció a DI containerben ✅ JAVÍTVA – 2026-04-03
**Fájl:** `app/bootstrap/bootstrap.py`  
**Leírás:** Az `ApplicationContainer`-ben `data_manager` és `data_manager_repo` ugyanazt a `DataManagerRepository` objektumot tartalmazza; `ohlcv_repo` is `SqliteOhlcvRepository(data_manager=dm)` – ugyanazt a dm-et wrappeli.  
**Javítás:** `dm_repo` redundáns wrapper eltávolítva; `data_manager_repo` mezőbe direktben `dm` kerül.

---

### A3 – `build_application()` modul-importkor fut le ✅ JAVÍTVA – 2026-04-03
**Fájl:** gyökér `main.py`  
**Leírás:** `_APP_CONTAINER = build_application(ensure_dirs=False)` modul-szintű – minden `import main` hatására lefut `build_settings()`, DataManager init, DB directory ellenőrzés. Tesztelési mellékhatás.  
**Javítás:** Lazy `_get_container()` függvény – csak az első CLI parancs végrehajtásakor épül fel a container. `_SETTINGS` modul-szintű változó eltávolítva.

---

### A4 – `LiveExecutionEngine` nem implementált ✅ JAVÍTVA – 2026-04-03
**Fájl:** `app/services/execution_engines.py`  
**Leírás:** A `EXECUTION_MODE="live"` elfogadott konfigurációs érték, de arra `NoopExecutionEngine` van regisztrálva – valódi kereskedés technikai infrastruktúra nélkül nem lehetséges. A kód "live" módot enged beállítani anélkül, hogy figyelmeztetne, hogy az nem tényleges live végrehajtás.

---

### A5 – Biztonsági default értékek nem blokkolják a production indítást ✅ JAVÍTVA – 2026-04-03
**Fájl:** `app/config/build_settings.py`  
**Leírás:** `ADMIN_API_KEY="admin_key_12345"` és `SECRET_KEY="dev_key_do_not_use_in_prod"` default értékekkel a rendszer simán elindul – csak log warning keletkezik. Production deploymentnél ez komoly biztonsági kockázat (authentication bypass).

---

## Összefoglaló táblázat

| ID | Súlyosság | Területe | Hatás | Státusz |
|----|-----------|----------|-------|---------|
| K1 | 🔴 Kritikus | ADX stub | RL modellek torzultak, ADX feature uninformative | ✅ Javítva 2026-04-02 |
| K2 | 🔴 Kritikus | Paper execution | SELL AttributeError multi-run esetén | ✅ Javítva 2026-04-02 |
| K3 | 🔴 Kritikus | Paper execution | P&L optimista, commission hiányzik | ✅ Javítva 2026-04-02 |
| K4 | 🔴 Kritikus | Governance | FrozenInstanceError – governance összeomlik | ✅ Javítva 2026-04-02 |
| H1 | 🟠 Magas | RSI indikátor | Nem standard, eltérő iparági értékektől | ✅ Javítva 2026-04-02 |
| H2 | 🟠 Magas | ATR indikátor | Nem standard, stop-loss torzít | ✅ Javítva 2026-04-02 |
| H3 | 🟠 Magas | RL training | Stale adatokon tanult modellek | ✅ Javítva 2026-04-02 |
| H4 | 🟠 Magas | Monthly pipeline | Automatikus retraining nem fut | ✅ Javítva 2026-04-03 |
| H5 | 🟠 Magas | Döntési logika | ensemble_quality típus inkonzisztencia | ✅ Javítva 2026-04-03 |
| M1 | 🟡 Közepes | CLI / dry_run | dry_run figyelmen kívül marad walk-forward-nál | ✅ Javítva 2026-04-02 |
| M2 | 🟡 Közepes | Use case réteg | Inkonzisztens Result pattern | ✅ Javítva 2026-04-03 |
| M3 | 🟡 Közepes | Konfiguráció | Pénzügyi paraméterek nem env var-ból | ✅ Javítva 2026-04-03 |
| M4 | 🟡 Közepes | CLI | Két párhuzamos CLI belépő | ✅ Javítva 2026-04-03 |
| M5 | 🟡 Közepes | Bootstrap/DI | Governance DI bypass | ✅ Javítva 2026-04-03 |
| M6 | 🟡 Közepes | Domain modell | Formalizálatlan entitások (dict helyett typed) | ✅ Javítva 2026-04-03 |
| A1 | 🔵 Alacsony | MACD obs | Histogram hiányzik RL observation vectorból | ✅ Javítva 2026-04-03 |
| A2 | 🔵 Alacsony | Bootstrap/DI | Duplikált repo referenciák | ✅ Javítva 2026-04-03 |
| A3 | 🔵 Alacsony | Bootstrap | Import mellékhatás | ✅ Javítva 2026-04-03 |
| A4 | 🔵 Alacsony | Execution | Live mode nincs implementálva, de konfigurálható | ✅ Javítva 2026-04-03 |
| A5 | 🔵 Alacsony | Biztonság | Insecure default értékek nem blokkolnak | ✅ Javítva 2026-04-03 |

---

## Javítási prioritás

1. ✅ **K4** – Governance FrozenInstanceError: javítva 2026-04-02 (env var + `set_settings`)
2. ✅ **K2** – PaperPosition deserializálás: javítva 2026-04-02 (`PaperPosition(**v)` konverzió)
3. ✅ **K3** – Commission levonás paper execution-ben: javítva 2026-04-02 (BUY + SELL)
4. ✅ **H3** – RL trainer hardcoded end date: javítva 2026-04-02 (`str(date.today())`)
5. ✅ **H4** – `ENABLE_RL` default: javítva 2026-04-03 (WARNING log ha ENABLE_RL=false)
6. ✅ **K1** – ADX valódi implementáció: javítva 2026-04-02 (Wilder-féle +DI/-DI)
7. ✅ **H1/H2** – RSI + ATR Wilder-féle EMA simítás: javítva 2026-04-02
8. ✅ **M1** – dry_run walk-forward chain: javítva 2026-04-02 (3 fájl)
9. ✅ **M3** – Pénzügyi paraméterek env var mappingba: javítva 2026-04-03 (INITIAL_CAPITAL, RISK, TRANSACTION_FEE_PCT, stb.)
10. ✅ **H5** – ensemble_quality típus: javítva 2026-04-03 (_quality_label() float→STABLE/NORMAL/CHAOTIC)
11. ✅ **M2** – DailyPipelineUseCase Result pattern: javítva 2026-04-03
12. ✅ **M4** – app/main.py deprecáció: javítva 2026-04-03 (DeprecationWarning)
13. ✅ **A4** – LiveExecutionEngine fallback warning: javítva 2026-04-03
14. ✅ **A5** – SECRET_KEY + ADMIN_API_KEY insecure warning: javítva 2026-04-03
15. ✅ **M5** – Governance DI bypass (RunGovernanceUseCase): javítva 2026-04-03
16. ✅ **M6** – Domain TypedDict-ek (app/domain/types.py): javítva 2026-04-03
17. ✅ **A1** – MACD histogram observation vector: javítva 2026-04-03
18. ✅ **A2** – Bootstrap dm_repo duplikáció eltávolítva: javítva 2026-04-03
19. ✅ **A3** – Lazy _get_container() init: javítva 2026-04-03
