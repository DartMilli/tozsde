# QUANT VALIDATION & HARDENING PLAN
## Cél: Production-grade pénzügyi megbízhatóság elérése

---

# 0. Dokumentum célja

Ez a dokumentum a rendszer:

- Torzításmentességének
- Tudományos validitásának
- Pénzügyi stabilitásának
- Live-ready állapotának

szisztematikus audit és javítási terve.

A cél nem az, hogy “jól működjön”, hanem hogy:

> Objektíven eldönthető legyen: rá merhető-e bízni valódi tőke.

---

# 1. KRITIKUS TORZÍTÁSOK VIZSGÁLATA

---

# 1.1 Lookahead Bias Audit

## Probléma

A jelenlegi rendszer:

- Close-to-close logikát használ
- Same-bar execution lehetséges
- Daily pipeline záróár alapú döntést hozhat

Ez implicit jövőbelátás lehet.

---

## 1.1.1 Kötelező javítás – As-Of Data Policy

### Követelmény

Minden döntés:

```
decision_timestamp = t
execution_price = open[t+1]
```

Soha nem használható:
- t napi close ár execution-re
- t napi close ár, ha a rendszer napközben fut

---

## 1.1.2 Implementációs feladat

Copilot prompt:

> Refactor the backtester and live pipeline to enforce strict next-bar execution policy.  
> Replace any same-bar close execution with next-bar open execution.  
> Add a configurable execution_mode flag:  
> - "close_to_close" (for research only)  
> - "next_open" (production safe)

---

## 1.1.3 Validáció

Futtasd újra:

- GA optimalizáció
- Walk-forward
- RL backtest

### Értékelés

| Eredmény | Jelentés |
|----------|----------|
| <15% romlás | elfogadható |
| 15–35% | torzítás volt |
| >35% | modell hamis előnyre épült |

---

# 1.2 Data Leakage Audit

---

## Probléma

- Reliability számítás outcome alapján
- RL train/test explicit szeparáció nélkül
- Confidence kalibráció teljes adaton

---

## 1.2.1 Kötelező javítás – Időalapú szeparáció

Minden tanulási komponensre:

```
train_end_date
validation_start_date
test_start_date
```

Soha nem számolhat jövőbeli adatból.

---

## 1.2.2 Implementációs feladat

Copilot prompt:

> Refactor RL training to enforce explicit train/validation/test split by date.  
> Add rolling window cross validation support.  
> Ensure reliability and confidence modules only use past data relative to decision date.

---

## 1.2.3 Leakage Teszt

1. Futtasd reliability számítást csak múlt adatra
2. Hasonlítsd össze az eredménnyel

Ha jelentősen romlik → leakage volt.

---

# 2. RL STABILITÁSI VIZSGÁLAT

---

# 2.1 Overfitting Stress Test

---

## 2.1.1 Period Reversal Test

Train: 2018–2021  
Test: 2022–2023  

Majd fordítva.

Ha egyik jó, másik rossz → overfit.

---

## 2.1.2 Noise Injection Test

Indikátorokhoz adj:

```
feature *= (1 + random.normal(0, 0.05))
```

Ha performance összeomlik → túl érzékeny modell.

---

## 2.1.3 Seed Stability

Futtasd RL traininget 5 külön seed-del.

Számold:

- mean Sharpe
- std Sharpe

Ha std túl nagy → instabil.

---

# 3. WALK-FORWARD ROBUSZTUSSÁG

---

## 3.1 Ablak stabilitás

Minden WF ablakra ments:

- OOS return
- OOS Sharpe
- OOS maxDD

Számold:

```
stability_score = std(OOS Sharpe)
```

Ha túl magas → paraméter instabil.

---

## 3.2 GA Seed Variance Test

Futtasd GA-t 5 külön seed-del.

Vizsgáld:

- Paraméter variancia
- Fitness variancia

Ha paraméterek teljesen eltérnek → nincs stabil optimum.

---

# 4. BACKTEST vs LIVE KONZISZTENCIA

---

## 4.1 Shadow Mode

30 napig párhuzamos futtatás:

- Live pipeline
- Backtester replay ugyanarra a napra

Összehasonlítás:

| Metric | Elvárt eltérés |
|--------|----------------|
| Entry signal | 0 |
| Position size | 0 |
| Equity | <2% |

Ha eltér → külön logika van.

---

## 4.2 Kötelező javítás

Unifikálni kell:

- Indicator számítás
- Position sizing
- Execution policy

Külön kódút nem maradhat.

---

# 5. RISK MANAGEMENT VALIDÁCIÓ

---

## 5.1 Position Sizing Stress

Teszt:

- Equity 0.2x
- Equity 10x
- Volatility spike

Ellenőrizd:

- Allocation ≤ 100%
- Cash ≥ 0
- Position ≤ max_limit

---

## 5.2 Bear Market Replay

Futtasd:

- 2008
- 2020
- 2022

Ha:

- MaxDD > 40%
- Safety nem aktivál

→ risk system nem production-ready.

---

# 6. PRODUCTION HARDENING

---

## 6.1 Egységesített Időkezelés

Minden modul kapjon:

```
as_of_date
```

Ne használjon implicit “latest available” adatot.

---

## 6.2 Single Source of Truth

Reliability:

- Vagy DB
- Vagy file

Nem lehet kettő.

---

## 6.3 Logging & Audit Trail

Minden döntéshez:

- as_of_date
- used_features_hash
- model_version
- reliability_score
- execution_price
- safety_flags

---

# 7. DÖNTÉSI MÁTRIX – LIVE ENGEDÉLYEZÉS

A rendszer csak akkor mehet live:

| Kritérium | Minimum |
|-----------|----------|
| OOS Sharpe | >0.8 |
| Profit factor | >1.3 |
| MaxDD | <25% |
| WF stability | alacsony szórás |
| RL seed std | alacsony |
| GA seed variance | stabil |
| Shadow diff | <2% |
| Lookahead romlás | <15% |

Ha bármelyik nem teljesül → csak paper trading.

---

# 8. JAVÍTÁSI PRIORITÁSI SORREND

1. Lookahead fix (kötelező)
2. RL train/test split
3. Shadow consistency
4. WF stabilitás
5. Risk stress test
6. Reliability leakage fix
7. GA robustness tuning

---

# 9. VÉGÁLLOMÁS

A rendszer akkor tekinthető production-ready-nek, ha:

- Torzításmentes
- Időalapú szeparáció garantált
- OOS stabil
- Seed-robosztus
- Backtest = live logika
- Risk kontroll validált
- Audit trail teljes

---

# 10. Következő lépés

Ha ezt a dokumentumot elfogadod, a következő fázis lehet:

- Automatikus validation runner script
- Quant stability score generálás
- Confidence-weighted capital allocation rendszer

---

Ez a dokumentum egy mérnöki validációs roadmap.
Ha ezt végigviszed, a rendszer egy hobbi-projektből valódi quant rendszer szintre lép.



---

# 11. AUTOMATIKUS VALIDATION RUNNER – TERV

## 11.1 Cél

Egyetlen parancsból futtatható teljes rendszer-validáció:

```
python validation_runner.py --mode full
```

A runner:

- Lefuttat minden kritikus torzítás tesztet
- Kiszámolja a stabilitási metrikákat
- Összehasonlítja backtest vs shadow eredményeket
- Generál egy összesített Quant Stability Score-t
- Készít egy gépileg és emberileg olvasható riportot

---

## 11.2 Javasolt mappastruktúra

```
app/
  validation/
    validation_runner.py
    bias_tests.py
    rl_stress_tests.py
    wf_analysis.py
    ga_robustness.py
    shadow_compare.py
    risk_stress.py
    scoring.py
    report_builder.py
```

---

## 11.3 validation_runner.py – Fő Orkesztrátor

Feladata:

1. Konfiguráció betöltése
2. Tesztek sorrendben futtatása
3. Eredmények aggregálása
4. Scoring meghívása
5. Report generálása

Példa struktúra:

```
class ValidationRunner:
    def __init__(self, config):
        self.config = config

    def run_bias_tests(self):
        pass

    def run_rl_stress(self):
        pass

    def run_walk_forward_analysis(self):
        pass

    def run_ga_robustness(self):
        pass

    def run_shadow_comparison(self):
        pass

    def run_risk_stress(self):
        pass

    def compute_final_score(self):
        pass
```

CLI módok:

- `--mode quick` → csak kritikus torzítások
- `--mode full` → minden teszt
- `--mode shadow` → csak backtest vs live összehasonlítás

---

## 11.4 Kimenet

A runner generál:

```
validation_reports/
  validation_YYYYMMDD.json
  validation_YYYYMMDD.md
```

JSON → gépi feldolgozás
MD → emberi audit

---

# 12. MÉRŐSZÁM MODUL STRUKTÚRA

## 12.1 Cél

Minden validáció ugyanazokat a standardizált metrikákat használja.

Ne legyen szétszórt Sharpe / drawdown számítás külön modulokban.

---

## 12.2 metrics_core.py

Központi metrika számítás:

```
compute_returns()
compute_sharpe()
compute_sortino()
compute_max_drawdown()
compute_profit_factor()
compute_calmar()
compute_volatility()
compute_trade_statistics()
```

Minden visszatér:

```
Dict[str, float]
```

---

## 12.3 stability_metrics.py

Walk-forward és seed stabilitás:

```
compute_oos_stability(wf_results)
compute_seed_variance(seed_results)
compute_parameter_variance(ga_runs)
```

Fontos mutatók:

- std(OOS Sharpe)
- std(OOS Return)
- Sharpe dispersion ratio
- Parameter coefficient of variation

---

## 12.4 bias_metrics.py

Lookahead és leakage mérés:

```
compare_execution_modes(close_results, next_open_results)
compute_leakage_delta(original_confidence, strict_confidence)
```

Kimenet:

- relative_performance_drop
- leakage_suspected (bool)

---

## 12.5 risk_metrics.py

Stress teszt mutatók:

```
compute_crash_drawdown()
compute_volatility_spike_response()
compute_allocation_violations()
```

---

# 13. QUANT STABILITY SCORE

## 13.1 Cél

Egyetlen 0–100 közötti pontszám.

---

## 13.2 Javasolt súlyozás

| Komponens | Súly |
|------------|------|
| OOS Sharpe | 20 |
| Max Drawdown | 15 |
| WF Stability | 15 |
| RL Seed Stability | 10 |
| GA Robustness | 10 |
| Bias Safety | 15 |
| Shadow Consistency | 10 |
| Risk Stress | 5 |

---

## 13.3 scoring.py

```
def compute_quant_score(results_dict):
    score = 0
    # normalize metrics
    # apply weights
    return score
```

Kimenet:

```
{
  "quant_score": 78.4,
  "status": "PAPER_READY"  # or RESEARCH_ONLY / LIVE_READY
}
```

---

# 14. AUTOMATIKUS REPORT GENERÁLÁS

report_builder.py:

- Táblázatos metrika összegzés
- GO / NO-GO jelzés
- Kritikus figyelmeztetések
- Ajánlott következő lépések

MD report szekciók:

1. Executive Summary
2. Bias Audit
3. RL Stability
4. Walk-Forward
5. GA Robustness
6. Risk Stress
7. Shadow Consistency
8. Final Score
9. Recommendation

---

# 15. IMPLEMENTÁCIÓS SORREND

1. metrics_core.py létrehozása
2. bias_tests.py implementálása
3. walk_forward_analysis.py
4. rl_stress_tests.py
5. scoring.py
6. validation_runner.py
7. report_builder.py

---

# 16. VÉGSŐ CÉL

A rendszer akkor tekinthető validáltnak, ha:

- A validation_runner egyetlen parancsból lefut
- Determinisztikus eredményt ad seed mellett
- Automatikusan GO/NO-GO státuszt állít be
- Riport archiválva van

---

Ezzel a struktúrával a projekt egy kutatási prototípusból auditálható quant rendszerré válik.

