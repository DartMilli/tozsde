# Tozsde Trading System – Átfogó architektúra review

> **Update (2026-02-09):** Stabilization + Validation completed (Sprint 12). See [SPRINTS.md](SPRINTS.md) for Phase 0–5 changes and validation tooling.

## Addendum (EN)
This review is historical. The current system includes:
- Paper execution with portfolio state tracking (PaperExecutionEngine).
- Historical paper runner with deterministic fallback when no RL models exist.
- Decision history with outcomes and position sizing persistence.
- Phase 5 and Phase 6 validation tooling with report integration.

## Addendum (HU)
Ez a review historikus. A jelenlegi rendszerben:
- Paper execution portfolio state mentessel (PaperExecutionEngine).
- Historikus paper runner determinisztikus fallback-kel, ha nincs RL modell.
- Döntesek es outcome-ok SQLite-ban, pozicio meretezessel.
- Phase 5 es Phase 6 validacios tooling, riport integracioval.

## 0. Kontextus
- Áttekintett fő belépési pontok és modulok: main.py, app/config/config.py, app/data_access/*, app/analysis/*, app/indicators/*, app/decision/*, app/backtesting/*, app/models/*, app/optimization/*, app/reporting/*, app/notifications/*, app/infrastructure/*, app/ui/*, app/scripts/*, run_dev.py.
- Cél: a jelenlegi állapot felmérése és értékelése (javítási javaslatok nélkül).

---

## 1. PROJECT OVERVIEW (összefoglaló)
**Alkalmazás célja:** End‑to‑end kvantitatív kereskedési rendszer. Központi funkciók: piaci adatok begyűjtése és tárolása, indikátor‑ és RL‑alapú döntéshozatal, napi ajánlások generálása, backtesting és walk‑forward optimalizáció, admin UI és értesítések.

**Fő komponensek:**
- **Belépési pontok / Orchestration:** main.py (daily/weekly/monthly pipeline), run_dev.py (dev mód)
- **Konfiguráció:** app/config/config.py
- **Adatelérés és tárolás (SQLite):** app/data_access/data_manager.py, data_loader.py
- **Indikátorok és elemzés:** app/analysis/analyzer.py, app/indicators/technical.py
- **Döntési logika:** app/decision/* (RL ensemble, policy, safety, allocation)
- **Backtesting / Walk‑forward:** app/backtesting/*
- **Modellek (RL/ML/LSTM):** app/models/*
- **Optimalizáció (GA):** app/optimization/*
- **Reporting / Audit:** app/reporting/*
- **Értesítések:** app/notifications/*
- **Infra / Ops:** app/infrastructure/*
- **UI:** app/ui/*

**Interakciók röviden:**
- Daily pipeline: adatbetöltés → RL ensemble döntés → safety/policy → allokáció → DB mentés → email.
- Weekly/Monthly: megbízhatósági elemzés + walk‑forward + RL tréning.
- Admin UI: monitoring és teljesítmény riportok.

---

## 2. FUNKCIONÁLIS TELJESSÉG (összefoglaló)
**Megvalósított fő funkciók:**
- Adatbetöltés és DB cache (data_loader.py, data_manager.py)
- Indikátorok, jelgenerálás (analyzer.py, technical.py)
- RL ensemble inference (rl_inference.py) és döntési logika (recommender.py)
- Safety / policy rétegek (safety_rules.py, decision_policy.py)
- Backtesting + walk‑forward (backtester.py, walk_forward.py)
- RL tréning (model_trainer.py)
- Audit és reporting (audit_builder.py, reporting/*)
- Értesítések (mailer.py, alerter.py)
- Admin UI (ui/app.py, ui/admin_dashboard.py)

**Részben vagy gyengén integrált elemek:**
- Decision history mentés és audit szerkezet eltérés (history_store.py ↔ data_manager.py).
- Outcome és reliability pipeline jelenleg kettős (DB vs fájlrendszer).
- Több modul implementációja "TODO" jelölésekkel (ui/app.py, run_dev.py, pyfolio_report.py, alerter.py).
- Ops/cron modul több hiányzó vagy eltérő API‑t hív.

---

## 3. ARCHITEKTÚRA ÉS DESIGN (összefoglaló)
- **Stílus:** moduláris, réteges struktúra (data_access, decision, backtesting, models, reporting).
- **Erősség:** jól elkülönített felelősségek a backtesting / optimization / decision területeken.
- **Gyengeség:** globális Config‑függőség, több helyen közvetlen DB‑hívások az üzleti logikában, részleges szolgáltatásréteg hiánya.

---

## 4. DATA FLOW & STATE MANAGEMENT (összefoglaló)
- **Adatfolyam:** yfinance → SQLite → prepare_df → RL ensemble → decision/safety → DB → email.
- **State:** SQLite DB + model fájlok (models/) + log fájlok (logs/).
- **Kockázat:** több helyen implicit I/O és side effect (data_loader yfinance, mailer SMTP, decision history blob). DB és fájl‑alapú állapot keveredik.

---

## 5. TESZTELHETŐSÉG ÉS MEGBÍZHATÓSÁG (összefoglaló)
- **Erős tesztek:** széles tesztlefedettség a tests/ könyvtár alapján.
- **Kihívások:** erős Config és I/O függés, közvetlen DB/SMTP használat mockolás nélkül.

---

## 6. TELJESÍTMÉNY ÉS SKÁLÁZHATÓSÁG (összefoglaló)
- **Bottleneckek:** per‑ticker download, RL modell‑betöltés, Python loop alapú backtester.

---

## 7. TECHNIKAI ADÓSSÁG ÉS KOCKÁZAT (összefoglaló)
1) Decision history mentési séma és hívás mismatch (magas).
2) Reliability pipeline DB‑fájl közti kettősség (magas).
3) Safety drawdown guard működésképtelen (közepes).
4) Dev/cron hívások hiányzó API‑kra (közepes).

---

## 8. READINESS (összefoglaló)
- **Backtesting:** közel kész, de néhány inkonzisztencia és placeholder csökkenti a megbízhatóságot.
- **Paper trading:** nincs végrehajtási/portfólió motor.
- **Live trading:** nincs broker integráció és order‑menedzsment.

---

# Részletes alrendszer‑review

## A) Belépési pontok és konfiguráció
**Fájlok:** main.py, run_dev.py, app/config/config.py
- **Main pipeline:** daily/weekly/monthly flows átláthatóan szervezettek. A daily pipeline adatbetöltéstől a döntés mentésig és emailig végigfut.
- **Config:** központi paraméterek és útvonalak, lazy ticker betöltés. Erős global state‑hatás.
- **Dev runner:** több TODO, és több helyen hibás/eltérő függvényhívások (pl. main.run_walk_forward nem létezik; main.train_rl_agent sem exportált). Részben inkonzisztens a main.py API‑hoz.

## B) Data Access & Storage
**Fájlok:** app/data_access/data_manager.py, data_loader.py, data_cleaner.py, market_context.py
- **DataManager:** SQLite schema és DAO réteg jól szervezett; több táblát támogat (ohlcv, recommendations, decision_history, pipeline_metrics, market_metadata).
- **DataLoader:** yfinance alapú letöltés és DB cache. Single source of truth a DB‑ből.
- **Market context:** VIX/IRX frissítés DB‑be.
- **Kockázat:** DataManager.save_history_record signature és decision history schema nem egyezik a HistoryStore használatával (explanation field). DataLoader implicit network side effects.

## C) Analysis & Indicators
**Fájlok:** app/analysis/analyzer.py, app/indicators/technical.py
- **Analyzer:** indikátorok és signal‑logika, paraméter‑fallback logika.
- **Indicators:** széles indikátor készlet, caching, pandas‑alapú EMA; jól dokumentált.
- **Kockázat:** compute_signals jelenleg korlátozott per‑bar jel logika (csak EMA/SMA), a többi indikátor per‑bar jel nem integrált.

## D) Decisioning & Policy
**Fájlok:** app/decision/* (recommender.py, decision_engine.py, safety_rules.py, decision_policy.py, recommendation_builder.py, ensemble_aggregator.py, decision_reliability.py)
- **Ensemble:** RL model votes + confidence + WF score aggregáció.
- **Policy:** reliability‑alapú trade engedélyezés és reward hint.
- **Safety:** cooldown, drawdown guard, VIX és bear market guard.
- **Kockázat:** drawdown guard outcomes adatra támaszkodik, de nincs outcomes tábla (fetch_recent_outcomes stub). Safety engine DataManager‑t hív (DB‑függés).

## E) Allocation & Portfolio Optimization
**Fájlok:** app/decision/allocation.py, risk_parity.py, capital_optimizer.py, confidence_allocator.py, rebalancer.py, etf_allocator.py, portfolio_correlation_manager.py
- **Allocation:** inverse‑vol + correlation penalty, de enforce_correlation_limits nincs integrálva a daily pipeline‑ba.
- **Risk parity / correlation manager:** fejlettebb allokációs logika, viszont fő pipeline nem használja.
- **Capital optimizer / confidence allocator / rebalancer / ETF allocator:** moduláris, de jelenleg izolált (nincs látványos integráció).

## F) Backtesting, Audit, Replay
**Fájlok:** app/backtesting/backtester.py, walk_forward.py, history_store.py, replay_runner.py, backtest_audit.py, outcome_evaluator.py, reward_engine.py, dataset_builder.py, training_dataset.py, transaction_costs.py
- **Backtester:** egyetlen ticker per futás, indikátorokra támaszkodó trade logika.
- **Walk‑forward:** GA‑val együttműködve, időablak‑kezeléssel.
- **Replay/audit:** döntések determinisztikus visszajátszása és reward számítása.
- **Kockázat:** decision history mentésben schema‑mismatch; outcome_evaluator audit_blob‑ot módosít, de a pipeline másik irányban ír döntéseket; integráció töredezett.

## G) Modellek (RL/ML/LSTM)
**Fájlok:** app/models/rl_inference.py, model_trainer.py, ml_predictor.py, lstm_predictor.py, model_output.py
- **RL:** PPO/DQN inference és tréning. TradingEnv obs‑struktúra rugalmas.
- **ML/LSTM:** külön prediktorok scikit‑learn / PyTorch alapokon, de nincs a döntési pipeline‑ba bekötve.
- **Kockázat:** ML/LSTM modulok potenciálisan árnyékban, nincs integráció a döntési láncba.

## H) Optimization (GA)
**Fájlok:** app/optimization/genetic_optimizer.py, fitness.py, ga_wf_normalizer.py
- **GA:** DEAP‑alapú optimalizáció paraméterekre, jó logolással.
- **Fitness:** single és WF score számítás.
- **Kockázat:** hosszú futások, compute intenzív; pipeline‑ba kapcsolás csak monthly walk‑forwardban.

## I) Reporting & Analytics
**Fájlok:** app/reporting/audit_builder.py, metrics.py, performance_analytics.py, pyfolio_report.py, plotter.py
- **Audit builder:** decision meta és audit summary.
- **Performance analytics:** részletes metrikák és drawdown analitika.
- **PyFolio:** opcionális, több TODO.
- **Plotter:** lazy import, rugalmas grafikon generálás.

## J) Notifications
**Fájlok:** app/notifications/mailer.py, alerter.py, email_formatter.py
- **Mailer:** SMTP alapú küldés.
- **Alerter:** kritikus/warning események, több TODO (pl. remediation mapping teljes).
- **Kockázat:** több helyen hiányzó / részlegesen implementált döntés az alert‑küldésre.

## K) Infrastructure / Ops
**Fájlok:** app/infrastructure/logger.py, metrics.py, error_reporter.py, health_check.py, backup_manager.py, log_manager.py, cron_tasks.py, decision_logger.py
- **Logger/metrics:** egységes logging, pipeline metrics DB‑be.
- **Backup/log manager:** Pi deployment orientált, jól strukturált.
- **Health check:** átfogó, API + DB + rendszer erőforrások.
- **Cron tasks:** több régi/eltérő API‑ra hivatkozik (DataManager.update_market_data, DecisionEngine.generate_recommendations), ami nem található a jelenlegi kódban.

## L) UI
**Fájlok:** app/ui/app.py, app/ui/admin_dashboard.py
- **Admin dashboard:** több teljesítmény és monitoring endpoint; auth kulccsal védett.
- **Dev endpoints:** több TODO (pl. dev/status, dev/config, dev/metrics), részben implementált.

## M) Scripts
**Fájlok:** app/scripts/apply_schema.py, app/scripts/smoke_test.py
- **apply_schema:** DB inicializálás.
- **smoke_test:** gyors ellenőrzés (tickers + DB táblák).

---

# Záró megjegyzés
Ez a dokumentum kizárólag a jelenlegi állapot értékelése. A következő lépésben (ha kéred) tudok javasolni konkrét javítási irányokat és refaktorokat alrendszer‑szinten.
