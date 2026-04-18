# Tozsde Trading System - Atfogo architektura review

> **Update (2026-02-09):** Stabilization + Validation completed (Sprint 12). See [SPRINTS.md](SPRINTS.md) for Phase 0-5 changes and validation tooling.
> **Update (2026-04-02):** Mélyreható kód-alapú elemzés elvégezve (csak forráskód, dokumentáció nélkül). Lásd KNOWN_ISSUES.md az azonosított elméleti és implementációs problémákhoz.
> **Update (2026-04-07):** Sprint 13 lezárva – összes kritikus hiba javítva. SQLite WAL mode bekapcsolva. Observation vector 11 feature-re bővült (MACD histogram). Governance in-process (`RunGovernanceUseCase`).
> **Update (2026-04-09):** Sprint 14 review elvégezve – 18 új probléma azonosítva (Q1–Q8, A1–A5, B1–B3, D1–D5). Lásd [SPRINT14_PLAN.md](SPRINT14_PLAN.md) a részletes javítási tervért.

### Ismert architektúrális törékeny pontok (Sprint 14 scope)

| ID | Komponens | Probléma |
|----|-----------|----------|
| Q6 🔴 | `recommender.py` → `_quality_label()` | `"STABLE"` stringet ad vissza STRONG esetén, de a `SafetyRuleEngine` `"CHAOTIC"` string egyenlőséget vizsgál. Az `EnsembleQualityBucket` enum értékei: `STRONG/NORMAL/WEAK/CHAOTIC` – `"STABLE"` nem létezik az enumban. Chaotic ensemble esetén a safety override **nem aktiválódik**. |
| Q2 🔴 | `allocation.py` → `allocate_capital()` | Mindig `INITIAL_CAPITAL`-ra allokál, nem az aktuális portfólióértékre. |
| A1 🟠 | `core/decision/*` | Több core osztály `settings or build_settings()` fallbacket tartalmaz – réteghatár-sértés. |
| A4 🟠 | `recommender.py` | `_quality_label()` threshold értékei hardcode-oltak, eltérnek a `DEFAULT_THRESHOLDS`-tól. |
| Q1 🟠 | `fitness.py` → `fitness_single()` | Abszolút pénzösszegek összeadása (dimenzióhiba) – normalizálás szükséges. |

## Addendum – Kód-alapú elemzés (2026-04-02)

### Valós rendszerállapot (csak kódból feltérképezve)

**Pipeline flow összefoglalás:**

| Parancs | Belépő | Végső logika |
|---------|--------|--------------|
| `daily` | `main.py → RunDailyPipelineUseCase` | `TradingPipelineService` → döntés → allokáció → `PaperExecutionEngine` → DB + email |
| `weekly` | `main.py → RunWeeklyReliabilityUseCase` | `ModelReliabilityAnalyzer` → DB mentés |
| `monthly` | `main.py → RunMonthlyRetrainingUseCase` | WalkForward + (ha ENABLE_RL=true) RL tanítás |
| `walk-forward <ticker>` | `main.py → RunWalkForwardUseCase` | `WalkForwardOptimizer` → DEAP GA → OOS foldenként |
| `train-rl <ticker>` | `main.py → TrainRLModelUseCase` | `train_rl_agent()` → DQN+PPO → `ModelPromotionGate` → `.zip` + `.meta.json` |
| `validate` | `main.py → RunPhase5ValidationUseCase` | `DecisionQualityAnalyzer` + `ConfidenceCalibrator` + `WalkForwardStabilityAnalyzer` + `SafetyStressTester` |
| `run-paper-history` | `main.py → RunHistoricalPaperUseCase` | `HistoricalPaperRunner` → `bdate_range` iteráció → per nap döntés + végrehajtás |
| `governance` | `main.py → RunGovernanceUseCase` | pytest + diagnostics + 13 validáció + checklist + reports/ |

**Döntési logika tényleges folyamata:**
```
load_data(180 nap) → prepare_df() (indikátorok) → RLModelEnsembleRunner (top 3 modell)
  → aggregate_weighted_ensemble (confidence × wf_score × rank × recency súlyozás)
  → scale_confidence_by_volatility()
  → build_recommendation() (NO_TRADE ha confidence < 0.25)
  → SafetyRuleEngine (cooldown / drawdown_guard / chaotic_ensemble / bear_market / high_VIX)
  → apply_decision_policy() → final Decision dict
```

**Observation vector (RL modellek):** 11 float32 elem:
`[price_to_sma, bb_pos, rsi/100, adx/100, macd/close, macd_sig/close, macd_hist/close, atr/close, cash/initial, portfolio_pct, is_holding]`

**SQLite séma (15+ tábla, DataManager kezeli):**
ohlcv, trades, recommendations, decision_history, outcomes, portfolio_state,
model_reliability, model_trust_metrics, model_registry, decision_effectiveness,
confidence_calibration, walk_forward_results, wf_stability_metrics,
safety_stress_results, validation_reports, decision_quality_metrics,
market_metadata, pipeline_metrics

**Ismert architektúrális eltérések a tervtől:**
- `app/core/` döntési logikát tartalmaz, nem tiszta domain entitásokat
- `app/governance/quant_runner.py` subprocess-ként fut, DI bypass-szal
- `app/main.py` egy régi, párhuzamos CLI – az aktív belépő a gyökér `main.py`
- Several service-ek (`TradingPipelineService`, `SafetyRuleEngine`) saját `build_settings()` hívással dolgoznak
- `Settings(frozen=True)` dataclass: immutable, nem módosítható futásidőben

**Azonosított kritikus problémák (részletek: KNOWN_ISSUES.md) – mind javítva 2026-04-07:**
- ✅ K1: ADX valódi Wilder implementáció – javítva 2026-04-02
- ✅ K2: PaperExecution SELL AttributeError – javítva 2026-04-02
- ✅ K3: Commission levonás paper executionben – javítva 2026-04-02
- ✅ K4: FrozenInstanceError governance futásnál – javítva 2026-04-02
- ✅ H1-H5: RSI/ATR Wilder-féle EMA simítás, RL stale adatok, ENABLE_RL default, ensemble_quality típus – javítva 2026-04-03
- ✅ M1-M6: dry_run, Result pattern, env var hiányok, CLI dupl., DI bypass, TypedDict – javítva 2026-04-03
- ✅ A1-A5: MACD histogram, Bootstrap duplikáció, lazy init, live mode, biztonsági default – javítva 2026-04-03

## Addendum (EN) – Legacy
This review is historical. The current system includes:
- Paper execution with portfolio state tracking (PaperExecutionEngine).
- Historical paper runner with deterministic fallback when no RL models exist.
- Decision history with outcomes and position sizing persistence.
- Phase 5 and Phase 6 validation tooling with report integration.

## Addendum (HU) – Legacy
Ez a review historikus. A jelenlegi rendszerben:
- Paper execution portfolio state mentessel (PaperExecutionEngine).
- Historikus paper runner determinisztikus fallback-kel, ha nincs RL modell.
- Dontesek es outcome-ok SQLite-ban, pozicio meretezessel.
- Phase 5 es Phase 6 validacios tooling, riport integracioval.

## 0. Kontextus
- Attekintett fo belepesi pontok es modulok: main.py, app/config/config.py, app/data_access/*, app/analysis/*, app/indicators/*, app/decision/*, app/backtesting/*, app/models/*, app/optimization/*, app/reporting/*, app/notifications/*, app/infrastructure/*, app/ui/*, app/scripts/*, run_dev.py.
- Cel: a jelenlegi allapot felmerese es ertekelese (javitasi javaslatok nelkul).

---

## 1. PROJECT OVERVIEW (osszefoglalo)
**Alkalmazas celja:** Endtoend kvantitativ kereskedesi rendszer. Kozponti funkciok: piaci adatok begyujtese es tarolasa, indikator es RLalapu donteshozatal, napi ajanlasok generalasa, backtesting es walkforward optimalizacio, admin UI es ertesitesek.

**Fo komponensek:**
- **Belepesi pontok / Orchestration:** main.py (daily/weekly/monthly pipeline), run_dev.py (dev mod)
- **Konfiguracio:** app/config/config.py
- **Adateleres es tarolas (SQLite):** app/data_access/data_manager.py, data_loader.py
- **Indikatorok es elemzes:** app/analysis/analyzer.py, app/indicators/technical.py
- **Dontesi logika:** app/decision/* (RL ensemble, policy, safety, allocation)
- **Backtesting / Walkforward:** app/backtesting/*
- **Modellek (RL/ML/LSTM):** app/models/*
- **Optimalizacio (GA):** app/optimization/*
- **Reporting / Audit:** app/reporting/*
- **Ertesitesek:** app/notifications/*
- **Infra / Ops:** app/infrastructure/*
- **UI:** app/ui/*

**Interakciok roviden:**
- Daily pipeline: adatbetoltes -> RL ensemble dontes -> safety/policy -> allokacio -> DB mentes -> email.
- Weekly/Monthly: megbizhatosagi elemzes + walkforward + RL trening.
- Admin UI: monitoring es teljesitmeny riportok.

---

## 2. FUNKCIONALIS TELJESSEG (osszefoglalo)
**Megvalositott fo funkciok:**
- Adatbetoltes es DB cache (data_loader.py, data_manager.py)
- Indikatorok, jelgeneralas (analyzer.py, technical.py)
- RL ensemble inference (rl_inference.py) es dontesi logika (recommender.py)
- Safety / policy retegek (safety_rules.py, decision_policy.py)
- Backtesting + walkforward (backtester.py, walk_forward.py)
- RL trening (model_trainer.py)
- Audit es reporting (audit_builder.py, reporting/*)
- Ertesitesek (mailer.py, alerter.py)
- Admin UI (ui/app.py, ui/admin_dashboard.py)

**Reszben vagy gyengen integralt elemek:**
- Decision history mentes es audit szerkezet elteres (history_store.py  data_manager.py).
- Outcome es reliability pipeline jelenleg kettos (DB vs fajlrendszer).
- Tobb modul implementacioja "TODO" jelolesekkel (ui/app.py, run_dev.py, pyfolio_report.py, alerter.py).
- Ops/cron modul tobb hianyzo vagy eltero APIt hiv.

---

## 3. ARCHITEKTURA ES DESIGN (osszefoglalo)
- **Stilus:** modularis, reteges struktura (data_access, decision, backtesting, models, reporting).
- **Erosseg:** jol elkulonitett felelossegek a backtesting / optimization / decision teruleteken.
- **Gyengeseg:** globalis Configfuggoseg, tobb helyen kozvetlen DBhivasok az uzleti logikaban, reszleges szolgaltatasreteg hianya.

---

## 4. DATA FLOW & STATE MANAGEMENT (osszefoglalo)
- **Adatfolyam:** yfinance -> SQLite -> prepare_df -> RL ensemble -> decision/safety -> DB -> email.
- **State:** SQLite DB + model fajlok (models/) + log fajlok (logs/).
- **Kockazat:** tobb helyen implicit I/O es side effect (data_loader yfinance, mailer SMTP, decision history blob). DB es fajlalapu allapot keveredik.

---

## 5. TESZTELHETOSEG ES MEGBIZHATOSAG (osszefoglalo)
- **Eros tesztek:** szeles tesztlefedettseg a tests/ konyvtar alapjan.
- **Kihivasok:** eros Config es I/O fugges, kozvetlen DB/SMTP hasznalat mockolas nelkul.

---

## 6. TELJESITMENY ES SKALAZHATOSAG (osszefoglalo)
- **Bottleneckek:** perticker download, RL modellbetoltes, Python loop alapu backtester.

---

## 7. TECHNIKAI ADOSSAG ES KOCKAZAT (osszefoglalo)
1) Decision history mentesi sema es hivas mismatch (magas).
2) Reliability pipeline DBfajl kozti kettosseg (magas).
3) Safety drawdown guard mukodeskeptelen (kozepes).
4) Dev/cron hivasok hianyzo APIkra (kozepes).

---

## 8. READINESS (osszefoglalo)
- **Backtesting:** kozel kesz, de nehany inkonzisztencia es placeholder csokkenti a megbizhatosagot.
- **Paper trading:** nincs vegrehajtasi/portfolio motor.
- **Live trading:** nincs broker integracio es ordermenedzsment.

---

# Reszletes alrendszerreview

## A) Belepesi pontok es konfiguracio
**Fajlok:** main.py, run_dev.py, app/config/config.py
- **Main pipeline:** daily/weekly/monthly flows atlathatoan szervezettek. A daily pipeline adatbetoltestol a dontes mentesig es emailig vegigfut.
- **Config:** kozponti parameterek es utvonalak, lazy ticker betoltes. Eros global statehatas.
- **Dev runner:** tobb TODO, es tobb helyen hibas/eltero fuggvenyhivasok (pl. main.run_walk_forward nem letezik; main.train_rl_agent sem exportalt). Reszben inkonzisztens a main.py APIhoz.

## B) Data Access & Storage
**Fajlok:** app/data_access/data_manager.py, data_loader.py, data_cleaner.py, market_context.py
- **DataManager:** SQLite schema es DAO reteg jol szervezett; tobb tablat tamogat (ohlcv, recommendations, decision_history, pipeline_metrics, market_metadata).
- **DataLoader:** yfinance alapu letoltes es DB cache. Single source of truth a DBbol.
- **Market context:** VIX/IRX frissites DBbe.
- **Kockazat:** DataManager.save_history_record signature es decision history schema nem egyezik a HistoryStore hasznalataval (explanation field). DataLoader implicit network side effects.

## C) Analysis & Indicators
**Fajlok:** app/analysis/analyzer.py, app/indicators/technical.py
- **Analyzer:** indikatorok es signallogika, parameterfallback logika.
- **Indicators:** szeles indikator keszlet, caching, pandasalapu EMA; jol dokumentalt.
- **Kockazat:** compute_signals jelenleg korlatozott perbar jel logika (csak EMA/SMA), a tobbi indikator perbar jel nem integralt.

## D) Decisioning & Policy
**Fajlok:** app/decision/* (recommender.py, decision_engine.py, safety_rules.py, decision_policy.py, recommendation_builder.py, ensemble_aggregator.py, decision_reliability.py)

> **Sprint 14 frissítés:** Az `app/decision/` könyvtár egy **kompatibilitási shim-réteg** – minden fájl az `app/core/decision/` valódi implementációkra hajt végre. Az összes fájl `DEPRECATED` modul-szintű docstringet kapott. Új kódban csak `app.core.decision.*` importot használj. Részletes deprecation terv: [PHASE7_DEPRECATION_PATH.md](PHASE7_DEPRECATION_PATH.md).

- **Ensemble:** RL model votes + confidence + WF score aggregacio. Elavult modellek kizárhatók `MAX_MODEL_AGE_DAYS` env var-ral.
- **Policy:** reliabilityalapu trade engedelyezes es reward hint.
- **Safety:** cooldown, drawdown guard, VIX es bear market guard. Sprint 14 után helyesen tüzeli a CHAOTIC ensemble minőséget (Q6 fix).
- **Kockazat:** drawdown guard outcomes adatra tamaszkodik, de nincs outcomes tabla (fetch_recent_outcomes stub). Safety engine DataManagert hiv (DBfugges).

## E) Allocation & Portfolio Optimization
**Fajlok:** app/decision/allocation.py, risk_parity.py, capital_optimizer.py, confidence_allocator.py, rebalancer.py, etf_allocator.py, portfolio_correlation_manager.py
- **Allocation:** inversevol + correlation penalty, de enforce_correlation_limits nincs integralva a daily pipelineba.
- **Risk parity / correlation manager:** fejlettebb allokacios logika, viszont fo pipeline nem hasznalja.
- **Capital optimizer / confidence allocator / rebalancer / ETF allocator:** modularis, de jelenleg izolalt (nincs latvanyos integracio).

## F) Backtesting, Audit, Replay
**Fajlok:** app/backtesting/backtester.py, walk_forward.py, history_store.py, replay_runner.py, backtest_audit.py, outcome_evaluator.py, reward_engine.py, dataset_builder.py, training_dataset.py, transaction_costs.py
- **Backtester:** egyetlen ticker per futas, indikatorokra tamaszkodo trade logika.
- **Walkforward:** GAval egyuttmukodve, idoablakkezelessel.
- **Replay/audit:** dontesek determinisztikus visszajatszasa es reward szamitasa.
- **Kockazat:** decision history mentesben schemamismatch; outcome_evaluator audit_blobot modosit, de a pipeline masik iranyban ir donteseket; integracio toredezett.

## G) Modellek (RL/ML/LSTM)
**Fajlok:** app/models/rl_inference.py, model_trainer.py, ml_predictor.py, lstm_predictor.py, model_output.py
- **RL:** PPO/DQN inference es trening. TradingEnv obsstruktura rugalmas.
- **ML/LSTM:** kulon prediktorok scikitlearn / PyTorch alapokon, de nincs a dontesi pipelineba bekotve.
- **Kockazat:** ML/LSTM modulok potencialisan arnyekban, nincs integracio a dontesi lancba.

## H) Optimization (GA)
**Fajlok:** app/optimization/genetic_optimizer.py, fitness.py, ga_wf_normalizer.py
- **GA:** DEAPalapu optimalizacio parameterekre, jo logolassal.
- **Fitness:** single es WF score szamitas.
- **Kockazat:** hosszu futasok, compute intenziv; pipelineba kapcsolas csak monthly walkforwardban.

## I) Reporting & Analytics
**Fajlok:** app/reporting/audit_builder.py, metrics.py, performance_analytics.py, pyfolio_report.py, plotter.py
- **Audit builder:** decision meta es audit summary.
- **Performance analytics:** reszletes metrikak es drawdown analitika.
- **PyFolio:** opcionalis, tobb TODO.
- **Plotter:** lazy import, rugalmas grafikon generalas.

## J) Notifications
**Fajlok:** app/notifications/mailer.py, alerter.py, email_formatter.py
- **Mailer:** SMTP alapu kuldes.
- **Alerter:** kritikus/warning esemenyek, tobb TODO (pl. remediation mapping teljes).
- **Kockazat:** tobb helyen hianyzo / reszlegesen implementalt dontes az alertkuldesre.

## K) Infrastructure / Ops
**Fajlok:** app/infrastructure/logger.py, metrics.py, error_reporter.py, health_check.py, backup_manager.py, log_manager.py, cron_tasks.py, decision_logger.py
- **Logger/metrics:** egyseges logging, pipeline metrics DBbe.
- **Backup/log manager:** Pi deployment orientalt, jol strukturalt.
- **Health check:** atfogo, API + DB + rendszer eroforrasok.
- **Cron tasks:** tobb regi/eltero APIra hivatkozik (DataManager.update_market_data, DecisionEngine.generate_recommendations), ami nem talalhato a jelenlegi kodban.

## L) UI
**Fajlok:** app/ui/app.py, app/ui/admin_dashboard.py
- **Admin dashboard:** tobb teljesitmeny es monitoring endpoint; auth kulccsal vedett.
- **Dev endpoints:** tobb TODO (pl. dev/status, dev/config, dev/metrics), reszben implementalt.

## M) Scripts
**Fajlok:** app/scripts/apply_schema.py, app/scripts/smoke_test.py
- **apply_schema:** DB inicializalas.
- **smoke_test:** gyors ellenorzes (tickers + DB tablak).

---

# Zaro megjegyzes
Ez a dokumentum kizarolag a jelenlegi allapot ertekelese. A kovetkezo lepesben (ha kered) tudok javasolni konkret javitasi iranyokat es refaktorokat alrendszerszinten.
