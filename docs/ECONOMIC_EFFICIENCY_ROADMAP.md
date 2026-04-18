# Gazdasági Hatékonyság Roadmap – Feature Implementációs Terv

**Projekt:** ToZsDE Trading System  
**Készült:** 2026-04-14  
**Bázis:** Sprint 18 kész, 1820 teszt, 80% coverage  
**Cél:** A rendszer risk-adjusted hozamának növelése a meglévő infrastruktúra visszacsatolási hurkainak zárásával

---

## Összefoglaló

A kódbázis átvizsgálása 8 közgazdaságilag releváns feature-lehetőséget azonosított. Ezek nem új modulok: a kód nagy része **már létezik** (risk parity, rebalancer, regime detector, reliability scoring, adaptive strategy selector, confidence calibrator), de nincs bekötve a napi döntési láncba, vagy feature flag mögött alszik. Az alábbi roadmap ezeket a dormáns képességeket aktiválja és integrálja.

### Hatás × Ráfordítás Mátrix

| # | Feature | Várható Sharpe-javulás | Drawdown-csökkenés | Ráfordítás | Prioritás |
|---|---------|----------------------|-------------------|------------|-----------|
| F1 | Reliability-weighted ensemble | ★★★★★ | ★★★★ | Közepes | 🔴 P0 |
| F2 | Regime-aware döntés & sizing | ★★★★ | ★★★★★ | Közepes | 🔴 P0 |
| F3 | Expectancy-alapú trade gating | ★★★★★ | ★★★ | Közepes | 🔴 P0 |
| F4 | Risk parity allokáció bekötése | ★★★ | ★★★★ | Alacsony | 🟠 P1 |
| F5 | Net-alpha rebalancer (cost-aware) | ★★★ | ★★★ | Alacsony | 🟠 P1 |
| F6 | Execution cost-aware trade gating | ★★★ | ★★ | Alacsony | 🟠 P1 |
| F7 | Adaptív stratégiarotáció bekötése | ★★★ | ★★★ | Közepes | 🟡 P2 |
| F8 | Champion-challenger shadow pipeline | ★★★★ | ★★★ | Magas | 🟡 P2 |

---

## Sprint Terv

### SPRINT 19: Reliability-Weighted Ensemble + Expectancy Gate (F1 + F3)
**Becsült scope:** 5 fájl módosítás, 2 új fájl, ~40 teszt  
**Cél:** A napi ajánlás flow ne csak confidence/WF/rank/recency alapján súlyozzon, hanem a modell tényleges historikus teljesítménye (reliability) és a trade várható nettó hozama (expectancy) is döntő tényező legyen.

#### F1 – Reliability-Weighted Ensemble

**Probléma:** A `generate_daily_recommendation_payload()` a `aggregate_weighted_ensemble()` hívásban confidence × wf_score × rank × recency alapon súlyoz. A heti reliability scoring (`RunWeeklyReliabilityUseCase`) kiszámol és ment modellenkénti hit_rate, avg_return, confidence_alignment és reliability_score értékeket, de ezek **nem befolyásolják** az ensemble döntést. A `weighted_ensemble_decision()` és `compute_decision_weight()` függvények (`decision_builder.py`, `weighting.py`) már tudják kezelni a reliability-t, de a fő recommender nem hívja őket.

**Megoldás:**

1. **`app/core/decision/ensemble_aggregator.py` – reliability paraméter hozzáadása**
   ```
   aggregate_weighted_ensemble(
       votes, confidences, wf_scores, model_votes,
       reliability_scores: dict[str, float] = None,  # ← ÚJ
       settings=None,
   )
   ```
   - Minden model_vote-nál: `weight *= reliability_scores.get(model_path, 0.5)`
   - Ha `reliability_scores` üres/None → régi viselkedés (fallback 0.5)

2. **`app/core/decision/recommender.py` – reliability betöltés a flow-ba**
   - A `generate_daily_recommendation_payload()` elején:
     ```python
     from app.models.model_reliability import load_latest_reliability_scores
     reliability_scores = load_latest_reliability_scores(ticker, as_of_date=today.isoformat())
     ```
   - Átadás az `aggregate_weighted_ensemble()` hívásba
   - Átadás a payload-ba (`payload["reliability_scores"] = reliability_scores`) audit célra

3. **`ENABLE_RELIABILITY` default → `true`**
   - `app/config/build_settings.py`: `ENABLE_RELIABILITY` default `"false"` → `"true"`
   - Ezzel a weekly pipeline automatikusan frissíti a reliability score-okat

4. **Automatikus modelldemóció**
   - `ensemble_aggregator.py`: ha egy modell `reliability_score < 0.3` → `weight *= 0.1` (gyakorlatilag kizárás)
   - Log warning: `"Model {model_path} demoted: reliability={score:.3f} < 0.3"`
   - Új Settings mező: `MODEL_DEMOTION_THRESHOLD` (default: 0.3, env var)

**Érintett fájlok:**
| Fájl | Változás |
|------|----------|
| `app/core/decision/ensemble_aggregator.py` | `reliability_scores` param + súlyozás |
| `app/core/decision/recommender.py` | reliability betöltés + átadás |
| `app/config/settings.py` | `MODEL_DEMOTION_THRESHOLD: float` |
| `app/config/build_settings.py` | `ENABLE_RELIABILITY` default, `MODEL_DEMOTION_THRESHOLD` env var |
| `app/models/model_reliability.py` | (változatlan – már van `load_latest_reliability_scores`) |

**Tesztek (~15):**
- `test_ensemble_aggregator_with_reliability_scores`
- `test_ensemble_aggregator_reliability_none_fallback`
- `test_ensemble_aggregator_model_demotion_below_threshold`
- `test_recommender_loads_reliability_scores`
- `test_recommender_no_reliability_data_graceful`
- `test_enable_reliability_default_true`
- Integration: `test_daily_pipeline_uses_reliability_weighted_ensemble`

---

#### F3 – Expectancy-Alapú Trade Gating

**Probléma:** A jelenlegi trade gating a `CONFIDENCE_NO_TRADE_THRESHOLD` (0.25) és az ensemble quality alapján dönt. Ez nem veszi figyelembe, hogy egy adott ticker/rezsim kombináció historikusan pozitív edge-et produkál-e költségek után. Például: egy 0.6 confidence-ű BUY döntés VOO-n bearish rezsimben historikusan −0.3% átlaghozmot ad → de a rendszer ezt most megengedi.

**Megoldás:**

1. **Új modul: `app/core/decision/expectancy_gate.py`**
   ```python
   @dataclass
   class ExpectancyResult:
       expected_pnl: float       # fee-mentes bruttó
       expected_net_pnl: float   # fee, slippage, spread levonva
       sample_count: int
       gate_pass: bool
       reason: str
   
   class ExpectancyGate:
       def __init__(self, settings, data_manager):
           ...
       
       def evaluate(self, ticker: str, action_code: int,
                    confidence_bucket: str, regime: str,
                    as_of_date: date) -> ExpectancyResult:
           """
           Lekérdezi az adott ticker/action/confidence_bucket/regime kombinációra
           a historikus outcome-ok átlagos pnl_pct-jét, levonja a tranzakciós
           költségeket, és gate_pass=True-t ad, ha az expected_net_pnl > 0.
           Min. EXPECTANCY_MIN_SAMPLES szükséges; alatta gate_pass=True (no data).
           """
   ```

2. **Bekötés a napi flow-ba**
   - `app/core/decision/recommender.py`: a `build_recommendation()` hívás **után**, de a `decision_engine.run()` **előtt**:
     ```python
     if getattr(cfg, "ENABLE_EXPECTANCY_GATE", False):
         gate = ExpectancyGate(settings=cfg, data_manager=...)
         result = gate.evaluate(ticker, action_code, confidence_bucket, regime, today)
         if not result.gate_pass:
             decision["action_code"] = 0
             decision["no_trade"] = True
             decision["no_trade_reason"] = f"EXPECTANCY_NEGATIVE: {result.reason}"
     ```

3. **Új Settings mezők:**
   - `ENABLE_EXPECTANCY_GATE: bool` (default: `false` → fokozatos bevezetés)
   - `EXPECTANCY_MIN_SAMPLES: int` (default: `10`)
   - `EXPECTANCY_LOOKBACK_DAYS: int` (default: `180`)

**Érintett fájlok:**
| Fájl | Változás |
|------|----------|
| `app/core/decision/expectancy_gate.py` | ÚJ modul |
| `app/core/decision/recommender.py` | gate bekötés a flow-ba |
| `app/core/decision/__init__.py` | export |
| `app/config/settings.py` | 3 új mező |
| `app/config/build_settings.py` | 3 env var mapping |

**Tesztek (~15):**
- `test_expectancy_gate_positive_edge_passes`
- `test_expectancy_gate_negative_edge_blocks`
- `test_expectancy_gate_insufficient_samples_passes`
- `test_expectancy_gate_with_cost_deduction`
- `test_expectancy_gate_regime_filtering`
- `test_recommender_expectancy_gate_integration`
- `test_expectancy_gate_disabled_by_default`

**Validáció:** Paper history run összehasonlítás gate ON vs OFF, VOO 2022-2023.

---

### SPRINT 20: Regime-Aware Döntés & Allokáció (F2)
**Becsült scope:** 4 fájl módosítás, 1 új fájl, ~25 teszt  
**Cél:** A piaci rezsim (BULL/BEAR/RANGING/VOLATILE) ne csak audit metadata legyen, hanem aktívan módosítsa a confidence threshold-okat, a maximum pozícióméretet és a safety szabályokat.

**Probléma:** A `MarketRegimeDetector` (`app/core/decision/market_regime_detector.py`) teljes értékű rezsimdetektálást végez (volatility, trend_strength, trend_consistency, R² stb.), de a napi flow csak az ADX-ből levezetett TREND/RANGE/TRANSITION címkét használja (recommender.py L283). A teljes rezsim-információ elvész.

**Megoldás:**

1. **Új modul: `app/core/decision/regime_policy.py`**
   ```python
   @dataclass(frozen=True)
   class RegimePolicy:
       confidence_floor: float      # min. confidence a trade-hez
       max_position_pct: float      # max egyedi pozíció size
       safety_strictness: str       # "RELAXED" | "NORMAL" | "STRICT"
       allow_new_buys: bool         # VOLATILE rezsimben korlátozható
       ensemble_quality_floor: str  # min. ensemble quality (STRONG/NORMAL/WEAK)
   
   REGIME_POLICIES = {
       "BULL":     RegimePolicy(0.20, 0.20, "RELAXED", True,  "WEAK"),
       "RANGING":  RegimePolicy(0.30, 0.15, "NORMAL",  True,  "NORMAL"),
       "BEAR":     RegimePolicy(0.50, 0.10, "STRICT",  False, "STRONG"),
       "VOLATILE": RegimePolicy(0.45, 0.08, "STRICT",  False, "STRONG"),
   }
   
   def get_regime_policy(regime: str, settings=None) -> RegimePolicy:
       """Visszaadja az aktuális rezsimhez tartozó policy-t.
       Settings-ből override-olható env var-okkal."""
   ```

2. **`app/core/decision/recommender.py` – teljes rezsim használata**
   - A jelenlegi ADX-alapú `regime = "TREND"` logikát kiegészítjük:
     ```python
     if getattr(cfg, "ENABLE_REGIME_POLICY", False):
         from app.core.decision.market_regime_detector import MarketRegimeDetector
         detector = MarketRegimeDetector(settings=cfg)
         regime_info = detector.detect_regime("SPY")
         regime = regime_info.regime_type  # BULL/BEAR/RANGING/VOLATILE
         regime_confidence = regime_info.confidence
     ```
   - A `build_recommendation()` hívás előtt: `CONFIDENCE_NO_TRADE_THRESHOLD` override regime alapján
   - A payload-ba: `regime_info` teljes objektum (audit + downstream)

3. **`app/core/decision/position_sizer.py` – regime-aware cap**
   - `apply_position_sizing()`: a `max_position_pct` felülírása regime_policy alapján
   - Ha `regime_policy.allow_new_buys == False` → BUY action → HOLD override

4. **`app/core/decision/safety_rules.py` – strictness integráció**
   - `STRICT` rezsimben: cooldown_days × 1.5, VIX threshold × 0.8
   - `RELAXED` rezsimben: cooldown_days × 0.75

5. **Új Settings mezők:**
   - `ENABLE_REGIME_POLICY: bool` (default: `false`)
   - `REGIME_POLICY_OVERRIDES: dict` (opcionális JSON env var a threshold fine-tuning-hoz)

**Érintett fájlok:**
| Fájl | Változás |
|------|----------|
| `app/core/decision/regime_policy.py` | ÚJ modul |
| `app/core/decision/recommender.py` | teljes rezsim detektálás + policy alkalmazás |
| `app/core/decision/position_sizer.py` | regime-aware max_position_pct |
| `app/core/decision/safety_rules.py` | strictness moduláció |
| `app/config/settings.py` | 2 új mező |
| `app/config/build_settings.py` | env var mapping |

**Tesztek (~25):**
- `test_regime_policy_bull_relaxed_thresholds`
- `test_regime_policy_bear_blocks_weak_buys`
- `test_regime_policy_volatile_tight_position_cap`
- `test_recommender_uses_full_regime_detection`
- `test_position_sizer_regime_aware_cap`
- `test_safety_rules_strict_cooldown_multiplier`
- `test_safety_rules_relaxed_cooldown_multiplier`
- `test_regime_policy_disabled_by_default`
- Integration: `test_daily_pipeline_regime_aware_end_to_end`

**Validáció:** Paper history run VOO 2022-2023 – összehasonlítás a regime policy ON/OFF drawdown és Sharpe metrikákra. A 2022 bear market szakasznak szignifikánsan kevesebb BUY-t kell produkálnia.

---

### SPRINT 21: Risk Parity Bekötés + Cost-Aware Rebalancer (F4 + F5)
**Becsült scope:** 3 fájl módosítás, ~20 teszt  
**Cél:** A napi allokáció opcionálisan risk parity módba kapcsolhasson, és a rebalancer csak akkor generáljon trade-et, ha a tracking-error csökkenés meghaladja a költségeket.

#### F4 – Risk Parity Allokáció Bekötése

**Probléma:** A `RiskParityAllocator` (`app/core/decision/risk_parity.py`) teljesen kész, de a `TradingPipelineService.allocate_capital()` az egyszerű inverse-vol + correlation penalty logikát használja (`app/core/decision/allocation.py`). A risk parity **soha nem hívódik**.

**Megoldás:**

1. **`app/services/trading_pipeline.py` – allokátor választó**
   ```python
   def allocate_capital(self, candidates):
       cfg = self._get_settings()
       mode = getattr(cfg, "ALLOCATION_MODE", "default")  # "default" | "risk_parity"
       
       if mode == "risk_parity":
           return self._allocate_risk_parity(candidates)
       else:
           return self._allocate_default(candidates)
   ```

2. **`_allocate_risk_parity()` implementáció:**
   - Price history betöltés a `data_fetcher.load_data()` segítségével (60 nap close)
   - `RiskParityAllocator(settings=cfg).allocate(candidates, price_history)`
   - Utána: `enforce_correlation_limits()` (ha `ENABLE_CORRELATION_LIMITS`)

3. **Risk parity `current_equity` support:**
   - Áthozzuk az equity-lekérdezést a `_allocate_risk_parity()` metódusba is

4. **Új Settings mező:**
   - `ALLOCATION_MODE: str` (default: `"default"`, env var)

**Érintett fájlok:**
| Fájl | Változás |
|------|----------|
| `app/services/trading_pipeline.py` | allokátor választó + risk_parity ág |
| `app/core/decision/risk_parity.py` | `current_equity` paraméter hozzáadása |
| `app/config/settings.py` | `ALLOCATION_MODE: str` |
| `app/config/build_settings.py` | env var mapping |

**Tesztek (~10):**
- `test_pipeline_allocation_mode_default`
- `test_pipeline_allocation_mode_risk_parity`
- `test_risk_parity_with_current_equity`
- `test_risk_parity_single_ticker_fallback`
- `test_allocation_mode_env_var_override`

---

#### F5 – Net-Alpha Cost-Aware Rebalancer

**Probléma:** A rebalancer check a napi use case-ben csak logol és javasol, de:
- Nem hasonlítja össze a drift-csökkenés értékét a tranzakciós költségekkel
- Nem hajtja végre a rebalance trade-eket
- A `minimize_rebalancing_costs()` és `apply_tax_efficiency()` metódusok léteznek, de senki nem hívja őket

**Megoldás:**

1. **`app/application/use_cases/daily_pipeline_use_case.py` – rebalancer execution**
   - A `_run_rebalancer_check()` metódusban:
     ```python
     trades = rebalancer.generate_rebalancing_trades(...)
     optimized = rebalancer.minimize_rebalancing_costs(trades)
     cost = rebalancer.compute_rebalancing_cost(optimized)
     benefit = self._estimate_drift_reduction_benefit(result, equity)
     
     if benefit > cost * REBALANCE_COST_MULTIPLIER:  # default: 2.0
         self.pipeline.execute_trades(optimized_as_decisions, as_of=today)
         logger.info("Rebalancer executed %d trades, cost=$%.2f, benefit=$%.2f",
                     len(optimized), cost, benefit)
     else:
         logger.info("Rebalancer skipped: benefit=$%.2f < cost*%.1f=$%.2f",
                     benefit, REBALANCE_COST_MULTIPLIER, cost * REBALANCE_COST_MULTIPLIER)
     ```

2. **Benefit becslés:**
   - `benefit = equity * drift_avg * DRIFT_ANNUAL_IMPACT_FACTOR` (default: 0.5)
   - Ez konzervatívan becsüli, hogy a drift milyen éves tracking-error degradációt okoz

3. **Új Settings mezők:**
   - `REBALANCE_COST_MULTIPLIER: float` (default: `2.0`) – min. benefit/cost arány
   - `REBALANCE_EXECUTE: bool` (default: `false`) – tényleges végrehajtás engedélyezése

**Érintett fájlok:**
| Fájl | Változás |
|------|----------|
| `app/application/use_cases/daily_pipeline_use_case.py` | rebalancer execution logika |
| `app/config/settings.py` | 2 új mező |
| `app/config/build_settings.py` | env var mapping |

**Tesztek (~10):**
- `test_rebalancer_executes_when_benefit_exceeds_cost`
- `test_rebalancer_skips_when_cost_too_high`
- `test_rebalancer_disabled_by_default`
- `test_rebalancer_cost_multiplier_env_var`
- `test_rebalancer_minimize_costs_integration`

---

### SPRINT 22: Execution Cost-Aware Gating + Adaptív Stratégiarotáció (F6 + F7)
**Becsült scope:** 4 fájl módosítás, 1 új fájl, ~30 teszt  
**Cél:** Trade-szintű implementation shortfall becslés és az adaptív stratégiaválasztó tényleges bekötése a döntési láncba.

#### F6 – Execution Cost-Aware Trade Gating

**Probléma:** A paper execution kezeli a commission-t, de a döntési pillanatban nem becsüli az össz-végrehajtási költséget (fee + slippage + spread). Kis edge-ű trade-eknél ez a teljes profit erodálhatja.

**Megoldás:**

1. **Új modul: `app/core/decision/implementation_shortfall.py`**
   ```python
   @dataclass
   class ShortfallEstimate:
       commission_pct: float
       slippage_pct: float
       spread_pct: float
       total_cost_pct: float
       min_edge_required: float  # total_cost_pct * COST_BUFFER_MULTIPLIER
   
   class ImplementationShortfallEstimator:
       def __init__(self, settings):
           self.fee = settings.TRANSACTION_FEE_PCT
           self.slippage = settings.MIN_SLIPPAGE_PCT
           self.spread = settings.SPREAD_PCT
       
       def estimate(self, notional: float, atr_pct: float = None) -> ShortfallEstimate:
           """Ha ATR elérhető, a slippage-et ATR-arányosan skálázzuk."""
   ```

2. **Bekötés: `app/application/use_cases/execution_coordinator.py`**
   - A `split_and_finalize()` metódusban a `finalized_decisions` szűrése:
     ```python
     if getattr(cfg, "ENABLE_COST_GATE", False):
         estimator = ImplementationShortfallEstimator(cfg)
         finalized = [d for d in finalized if self._passes_cost_gate(d, estimator)]
     ```
   - `_passes_cost_gate()`: ha `allocation_amount * expected_edge < total_cost` → skip + log

3. **Új Settings mezők:**
   - `ENABLE_COST_GATE: bool` (default: `false`)
   - `COST_BUFFER_MULTIPLIER: float` (default: `1.5`) – biztonsági szorzó

**Érintett fájlok:**
| Fájl | Változás |
|------|----------|
| `app/core/decision/implementation_shortfall.py` | ÚJ modul |
| `app/application/use_cases/execution_coordinator.py` | cost gate szűrés |
| `app/config/settings.py` | 2 új mező |
| `app/config/build_settings.py` | env var mapping |

**Tesztek (~12):**
- `test_shortfall_estimator_basic`
- `test_shortfall_estimator_atr_scaling`
- `test_cost_gate_blocks_low_edge_trade`
- `test_cost_gate_passes_high_edge_trade`
- `test_cost_gate_disabled_by_default`
- Integration: `test_execution_coordinator_with_cost_gate`

---

#### F7 – Adaptív Stratégiarotáció Bekötése

**Probléma:** Az `AdaptiveStrategySelector` (Thompson Sampling) a Sprint 6-ban készült el, de:
- A napi döntési flow **nem hívja**
- A strategy outcome feedback loop **nincs lezárva** (az `update_strategy()` nem hívódik az outcome mentésnél)
- Az admin dashboard hardcoded `["momentum", "mean_reversion", "breakout"]` stratégiákat listáz

**Megoldás:**

1. **Stratégia-weight integráció a recommender-be:**
   - `app/core/decision/recommender.py`:
     ```python
     if getattr(cfg, "ENABLE_ADAPTIVE_STRATEGY", False):
         from app.core.decision.adaptive_strategy_selector import AdaptiveStrategySelector
         selector = AdaptiveStrategySelector(settings=cfg)
         regime_info = ...  # re-use from F2 ha engedélyezve
         selection = selector.explore_or_exploit(market_regime=regime)
         # selection.selected_strategies → strategy weights
         # apply as additional multiplier to ensemble votes
     ```

2. **Outcome feedback loop zárása:**
   - `app/services/paper_execution.py` – a SELL végrehajtásnál:
     ```python
     if getattr(cfg, "ENABLE_ADAPTIVE_STRATEGY", False):
         selector = AdaptiveStrategySelector(settings=cfg)
         strategy_name = pos.strategy_name or "RL_ENSEMBLE"
         selector.update_strategy(strategy_name, success=outcome["pnl_pct"] > 0)
     ```
   - A `PaperPosition` dataclass bővítése: `strategy_name: str = "RL_ENSEMBLE"`

3. **Admin dashboard fix:**
   - `app/ui/admin_dashboard.py` `get_strategy_performance()`: 
     - Hardcoded lista helyett `AdaptiveStrategySelector.get_strategy_stats()` használata

4. **Új Settings mező:**
   - `ENABLE_ADAPTIVE_STRATEGY: bool` (default: `false`)

**Érintett fájlok:**
| Fájl | Változás |
|------|----------|
| `app/core/decision/recommender.py` | strategy weight integráció |
| `app/services/paper_execution.py` | outcome loop + PaperPosition.strategy_name |
| `app/ui/admin_dashboard.py` | dinamikus stratégia lista |
| `app/config/settings.py` | 1 új mező |
| `app/config/build_settings.py` | env var mapping |

**Tesztek (~18):**
- `test_adaptive_selector_explore_exploit`
- `test_adaptive_selector_regime_context`
- `test_outcome_updates_strategy_bandit`
- `test_paper_execution_strategy_name_tracking`
- `test_admin_dashboard_dynamic_strategies`
- `test_adaptive_strategy_disabled_by_default`
- Integration: `test_daily_pipeline_with_adaptive_strategy`

---

### SPRINT 23: Champion-Challenger Shadow Pipeline (F8)
**Becsült scope:** 3 új fájl, 4 fájl módosítás, ~30 teszt  
**Cél:** Új RL modellek ne rögtön élesben fussanak, hanem először shadow módban gyűjtsenek evidence-t, majd automatikus promóció történjen, ha az eredmények jobbak a champion modellnél.

**Probléma:** A havi retraining (`RunMonthlyRetrainingUseCase`) jelenleg: train → ModelPromotionGate → ha pass → save → azonnal aktív. Nincs „próbaidő", nincs párhuzamos futás. Ha az új modell rosszabb a valóságban, mint az OOS teszten, az csak az outcome-ok utólagos elemzéséből derül ki.

**Megoldás:**

1. **Új modul: `app/models/shadow_evaluator.py`**
   ```python
   class ShadowEvaluator:
       """Párhuzamosan futtatja a champion és challenger modelleket,
       napi szinten összegyűjti mindkettő döntéseit, és
       SHADOW_EVAL_DAYS nap után összehasonlítja a teljesítményt."""
       
       def register_challenger(self, ticker: str, model_path: str, meta: dict) -> None:
           """Új challenger modell regisztrálása shadow értékelésre."""
       
       def record_shadow_decision(self, ticker: str, date: date, 
                                   champion_action: int, challenger_action: int,
                                   market_outcome: float) -> None:
           """Napi shadow record mentése."""
       
       def evaluate_promotion(self, ticker: str) -> dict:
           """Ha elég nap telt el, összehasonlítás és promóciós javaslat.
           Returns: {promote: bool, champion_sharpe, challenger_sharpe, days_evaluated}"""
   ```

2. **Új tábla: `shadow_evaluations`**
   ```sql
   CREATE TABLE shadow_evaluations (
       id INTEGER PRIMARY KEY,
       ticker TEXT NOT NULL,
       date TEXT NOT NULL,
       champion_model TEXT,
       challenger_model TEXT,
       champion_action INTEGER,
       challenger_action INTEGER,
       market_return REAL,
       created_at TEXT DEFAULT (datetime('now')),
       UNIQUE(ticker, date, challenger_model)
   );
   ```

3. **Monthly retraining módosítás:**
   - `app/application/use_cases/run_monthly_retraining.py`:
     ```python
     if getattr(cfg, "ENABLE_SHADOW_EVAL", False):
         # Nem rögtön promote, hanem register as challenger
         shadow_eval.register_challenger(ticker, new_model_path, meta)
     else:
         # Régi viselkedés: azonnali mentés
         self._train_rl_fn(ticker, wf_score, wf_summary)
     ```

4. **Daily pipeline bővítés:**
   - `app/application/use_cases/daily_pipeline_use_case.py`:
     ```python
     # A napi döntés után, ha van aktív challenger:
     if getattr(cfg, "ENABLE_SHADOW_EVAL", False):
         self._run_shadow_evaluation(daily_candidates)
     ```
   - Ez a challenger modellt is lefuttatja (inference only), és menti a shadow döntést

5. **Automatikus promóció:**
   - A daily pipeline végén: `shadow_eval.evaluate_promotion(ticker)`
   - Ha `challenger_sharpe > champion_sharpe * 1.1` és `days >= SHADOW_EVAL_DAYS`:
     - Champion → archive
     - Challenger → champion
     - Log + email notification

6. **Új Settings mezők:**
   - `ENABLE_SHADOW_EVAL: bool` (default: `false`)
   - `SHADOW_EVAL_DAYS: int` (default: `30`)
   - `SHADOW_PROMOTION_THRESHOLD: float` (default: `1.1`) – challenger/champion Sharpe arány

**Érintett fájlok:**
| Fájl | Változás |
|------|----------|
| `app/models/shadow_evaluator.py` | ÚJ modul |
| `app/data_access/data_manager.py` | `shadow_evaluations` tábla + CRUD |
| `app/application/use_cases/run_monthly_retraining.py` | challenger registration |
| `app/application/use_cases/daily_pipeline_use_case.py` | shadow evaluation hívás |
| `app/config/settings.py` | 3 új mező |
| `app/config/build_settings.py` | env var mapping |

**Tesztek (~30):**
- `test_shadow_evaluator_register_challenger`
- `test_shadow_evaluator_record_decisions`
- `test_shadow_evaluator_promotion_passes`
- `test_shadow_evaluator_promotion_insufficient_days`
- `test_shadow_evaluator_promotion_threshold_not_met`
- `test_monthly_retraining_shadow_mode`
- `test_daily_pipeline_shadow_evaluation`
- `test_shadow_table_creation`
- `test_shadow_evaluator_multi_ticker`
- Integration: `test_champion_challenger_full_cycle`

---

## Sprint Áttekintő Táblázat

| Sprint | Feature-ok | Becsült tesztek | Új fájlok | Módosított fájlok | Előfeltétel |
|--------|-----------|-----------------|-----------|-------------------|-------------|
| **S19** | F1 + F3 (Reliability ensemble + Expectancy gate) | ~40 | 1 | 5 | – |
| **S20** | F2 (Regime-aware döntés & sizing) | ~25 | 1 | 5 | S19 (regime info a payload-ban) |
| **S21** | F4 + F5 (Risk parity + Cost-aware rebalancer) | ~20 | 0 | 5 | – |
| **S22** | F6 + F7 (Cost gate + Adaptív stratégia) | ~30 | 1 | 5 | S20 (regime info), S19 (outcome loop) |
| **S23** | F8 (Champion-challenger shadow) | ~30 | 1 | 5 | S19 (reliability), S20 (regime) |
| **ÖSSZESEN** | 8 feature | **~145 teszt** | **4 új** | **~12 módosított** | |

---

## Bevezetési Stratégia

### Fázis 1: Shadow/Analytics mód (S19-S20)
Minden feature `ENABLE_*=false` default-tal indul. Először csak a **méréseket** aktiváljuk:
- `ENABLE_RELIABILITY=true` (reliability scoring futtatása)
- Paper history összehasonlítási riportok generálása: ON vs OFF minden feature-re
- A governance runner bővítése az új metrikák ellenőrzésével

### Fázis 2: Fokozatos aktiválás (S21-S22)
A paper history eredmények alapján:
1. Ha a reliability-weighted ensemble javít → `ENABLE_RELIABILITY=true` + ensemble integráció aktív
2. Ha a regime policy csökkenti a drawdown-t → `ENABLE_REGIME_POLICY=true`
3. Ha az expectancy gate szűrése nettó pozitív → `ENABLE_EXPECTANCY_GATE=true`

### Fázis 3: Teljes integráció (S23)
- Champion-challenger pipeline bekapcsolás
- `ALLOCATION_MODE=risk_parity` kipróbálás
- `ENABLE_ADAPTIVE_STRATEGY=true` aktiválás

### Rollback terv
Minden feature env var mögött van. Bármely feature egyetlen `ENABLE_*=false` váltással kikapcsolható, a rendszer visszaáll a jelenlegi viselkedésre. Nincs destruktív sémaváltozás.

---

## Governance Bővítés

Minden sprint végén a governance runner ellenőrzi:

| Check | Leírás |
|-------|--------|
| `reliability_coverage` | Hány tickerre van friss (< 7 napos) reliability score |
| `regime_consistency` | A detektált rezsim konzisztens-e az SPY és VIX jelzéseivel |
| `expectancy_sample_depth` | Hány ticker/bucket kombinációra van ≥ `EXPECTANCY_MIN_SAMPLES` |
| `shadow_eval_active` | Hány ticker-en fut aktív challenger modell |
| `rebalancer_cost_efficiency` | A rebalancer trade-ek átlagos benefit/cost aránya |
| `strategy_bandit_health` | Nincs-e stuck bandit (> 100 trial, < 0.4 expected_value) |

---

## Sikerkritérium

A roadmap **akkor sikeres**, ha a VOO 2022-2023 paper history teszten:

| Metrika | Jelenlegi (baseline) | Cél (összes feature ON) |
|---------|---------------------|------------------------|
| Sharpe Ratio | mérni kell | ≥ +0.15 javulás |
| Max Drawdown | mérni kell | ≥ 20% csökkenés |
| Win Rate | mérni kell | ≥ +3pp javulás |
| Trade Count | mérni kell | csökkenhet (jobb szűrés) |
| Avg PnL per Trade | mérni kell | ≥ +0.1pp javulás |

**Megjegyzés:** A baseline mérés az első lépés; az S19 implementáció előtt `python main.py run-paper-history --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31` futtatása szükséges a kiindulási értékek rögzítéséhez.
