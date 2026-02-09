# ToZsDE Trading System – Teljes funkcionalitás (HU)

## English Summary (Current)
Tozsde is a Python trading system that runs a daily decision pipeline, persists auditable decisions and outcomes in SQLite, and provides backtesting, historical paper runs, and Phase 5/6 validation. Historical paper runs use a deterministic fallback HOLD decision when no RL models are available. CLI entry point is main.py.

Key CLI examples:
```bash
python main.py daily
python main.py run-paper-history --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
python main.py validate --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
```

**Cél:** Ez a dokumentum a rendszer teljes működését írja le **felhasználói** és **fejlesztői** szemszögből, különös tekintettel a funkcionalitásra, a teljes pipeline-ra és az UI-ra.

---

## 1) Rövid áttekintés

A ToZsDE Trading System egy **adatvezérelt kereskedési döntéstámogató** platform, amely az adatoktól az ajánlásig mindent lefed:
- Piaci adatok gyűjtése és tárolása (OHLCV, benchmarkok, volatilitási indexek).
- Technikai indikátorok számítása és jelgenerálás.
- Döntésépítés (BUY/SELL/HOLD) magyarázattal és audit mezőkkel.
- Kockázatkezelés, korrelációs korlát és tőkekiosztás.
- Naplózás, riportálás, megbízhatóság-elemzés és monitorozás.

**Fontos:** a rendszer **nem** hajt végre valós tőzsdei tranzakciókat. A kimenet ajánlás és döntéstámogatás, nem automatikus végrehajtás.

---

## 2) Felhasználói nézőpont

### 2.1 Napi működés (felhasználóként)
Egy napi futás után a felhasználó az alábbi információkat kapja eszközönként:
- **Ajánlás (BUY/SELL/HOLD)**: a rendszer döntése.
- **Döntés indoklása**: indikátorok, trend, volatilitás, modell-megbízhatóság, döntési erősség.
- **Tőkekiosztás**: javasolt pozícióméret az adott eszközre.
- **Kockázati jelölések**: no-trade ok, korrelációs korrekció, piaci rezsim figyelmeztetés.

### 2.2 Tipikus felhasználói kérdések és válaszok
- **Miért HOLD?** → mert a bizalom alacsony, a piaci rezsim kedvezőtlen, vagy túl magas a korreláció.
- **Miért BUY?** → mert több indikátor megerősíti a trendet, és a megbízhatóság megfelelő.
- **Mennyi tőkét javasol?** → a risk parity és korrelációs korlátok alapján.
- **Mennyire megbízható a jel?** → a drift és megbízhatósági metrikák alapján.

---

## 3) Fejlesztői nézőpont – Fő komponensek (részletes)

### 3.1 Adat és tárolás
- **DataLoader**: adatok letöltése külső forrásból, normalizálás, mentés.
- **DataManager**: SQLite réteg, OHLCV, ajánlások, döntésnapló, audit, metrikák.
- **Cache és adatfrissítés**: a régi adatok újratöltése és hiányzó napok pótlása.

### 3.2 Döntés és stratégia
- **RecommendationBuilder**: standard döntési objektum (confidence, strength, no_trade okok).
- **DecisionPolicy**: szabályok és küszöbök alkalmazása.
- **AdaptiveStrategySelector**: stratégiák súlyozása Thompson sampling alapján.
- **ConfidenceAllocator**: tőkeelosztás bizalmi sávok szerint.

### 3.3 Kockázat és optimalizáció
- **Risk Parity**: inverz volatilitás-alapú súlyozás.
- **Correlation Limits**: túl magas korreláció esetén súlycsökkentés.
- **Capital Utilization Optimizer**: Kelly-alapú méretezés és drawdown-becslés.

### 3.4 Backtesting és audit
- **Backtester**: múltbeli szimulációk.
- **Walk-forward**: időben gördülő tesztelés.
- **BacktestAuditor**: döntési szintek és túlzott önbizalom felismerése.

### 3.5 Infrastruktúra
- **CronTaskScheduler**: napi/heti/havi feladatok.
- **HealthCheck**: API, DB, CPU, memória és lemez ellenőrzés.
- **BackupManager / LogManager**: mentések és logrotáció.
- **Notifications**: email és riasztás (error/critical eseményeknél).

---

## 4) Teljes pipeline (részletes végigvezetés)

A pipeline célja, hogy **friss piaci adatokból stabil, auditált ajánlásokat készítsen**, miközben biztosítja a minőséget és a rendszer egészségét.

### 4.1 Napi pipeline – részletes lépések
1. **Adatfrissítés**
   - OHLCV adatok frissítése.
   - Hiányzó napok pótlása.
   - VIX/benchmark adatok betöltése.

2. **Adattisztítás és előkészítés**
   - NaN/hiányzó értékek kezelése.
   - Konzisztens időskála.
   - Átlagos volatilitás számítása.

3. **Indikátor-számítás**
   - SMA, EMA, RSI, MACD, Bollinger Bands, ATR, ADX, Stochastic.
   - Jelzési feltételek validálása.

4. **Jelgenerálás**
   - Indikátor-keresztezések, túladott/túlvett szintek.
   - Előzetes BUY/SELL/HOLD javaslat.

5. **Döntésépítés és policy alkalmazása**
   - Bizalom és erősség besorolás (STRONG/NORMAL/WEAK).
   - No-trade logika (pl. alacsony bizalom, kaotikus ensemble).
   - Piaci rezsim figyelembevétele (bear market szűrő).

6. **Kockázatkezelés és allokáció**
   - Risk parity súlyozás.
   - Korrelációs limit.
   - Pozícióméret véglegesítése.

7. **Mentés, audit és értesítés**
   - Döntés és ajánlás mentése DB-be.
   - Audit metadata és összegzés rögzítése.
   - Email vagy riasztás küldése.

8. **Karbantartás**
   - Logrotáció.
   - Backup.
   - Régi mentések törlése.

### 4.4 Validációs pipeline (Phase 5)
- **Döntési minőség**: confidence bucket alapú hit rate és hozam metrikák.
- **Kalibráció**: confidence → valószínűség (isotonic) + ECE/Brier.
- **WF stabilitás**: paraméter variancia és konzisztencia.
- **Safety stress**: extrém szcenáriók (volatilítás, gap, drawdown).
- **Riport**: egyesített validációs jelentés a teszt riportban.

Futtatás:
```
python scripts/run_tests_with_report.py --with-validation --skip-tests --ticker VOO --start-date 2020-01-01 --end-date 2024-01-01
```

### 4.5 Phase 6 ellenorzesek (P6)
- **Effectiveness**: outcome alapu hatasossag.
- **Position sizing**: monotonitas es cap ellenorzes.
- **Model trust**: model vote-ok es outcome-ok alapu sulyok.
- **Reward shaping**: reward komponensek logolasa.
- **Promotion gate**: baseline vs candidate ellenorzes.

### 4.2 Heti pipeline
- **Model reliability**: döntések és outcome-ok alapján megbízhatósági score.
- **Drift detection**: romló teljesítmény észlelése.

### 4.3 Havi pipeline
- **Walk-forward optimalizáció**: paraméterek és fitness újrakalibrálása.
- **RL modell tréning** (opcionális): új modellek betanítása.

---

## 5) UI és API funkcionalitás

A rendszer Flask-alapú UI-t és admin API-t biztosít, amely **monitorozásra, vizualizációra és állapotellenőrzésre** szolgál.

### 5.1 Felhasználói UI
- **/ (Főoldal)**: áttekintés, alap státusz.
- **/chart**: árfolyam és indikátor grafikonok.
- **/history**: historikus döntések és ajánlások.
- **/indicators**: indikátor leírások és magyarázatok.
- **/report**: teljesítmény-riport és összesített metrikák.
- **/params**: aktuális paraméterek megtekintése.

### 5.2 Admin Dashboard (auth kulccsal)
- **/admin/health**: egészségügyi állapot.
- **/admin/metrics**: pipeline metrikák, futási statisztikák.
- **/admin/performance/**: teljesítmény, drawdown, rolling metrikák.
- **/admin/errors/**: hibastatisztikák és trendek.
- **/admin/capital/**: tőkehasználat, allokációs átláthatóság.
- **/admin/decisions/**: no-trade okok és trendek.

### 5.3 Dev mód (fejlesztői endpointok)
- Fejlesztői státusz, konfiguráció és DB inicializálás csak dev környezetben.

---

## 6) Algoritmusok és módszerek (kibővítve)

### 6.1 Technikai indikátorok
- **SMA (Simple Moving Average):**
  $$SMA_t = \frac{1}{n} \sum_{i=0}^{n-1} P_{t-i}$$
- **EMA (Exponential Moving Average):**
  $$EMA_t = \alpha P_t + (1-\alpha)EMA_{t-1}$$
- **RSI (Relative Strength Index):**
  $$RSI = 100 - \frac{100}{1 + RS}$$
- **MACD:**
  $$MACD = EMA_{fast} - EMA_{slow}$$
- **Bollinger Bands:**
  $$BB_{upper} = SMA + k\sigma, \quad BB_{lower} = SMA - k\sigma$$
- **ATR / ADX / Stochastic:** volatilitás és trend-erősség mérésére.

### 6.2 Piaci rezsim detektálás
- Volatilitás, trend és konszisztencia alapján:
  - **VOLATILE**: magas volatilitás.
  - **BULL**: pozitív trend, stabil erő.
  - **BEAR**: negatív trend, erős lejtmenet.
  - **RANGING**: oldalazó piac.

### 6.3 Risk Parity (inverz volatilitás)
- **Súly:** $w_i \propto \frac{1}{\sigma_i}$
- **Normalizálás:** $\sum w_i = 1$

### 6.4 Korrelációs limit
- Magas korreláció esetén a gyengébb eszköz súlya csökken.

### 6.5 Kelly-kritérium (tőkeoptimalizálás)
- **Kelly:**
  $$f = \frac{p\cdot a - (1-p)\cdot b}{a}$$
- A rendszer korlátozza a túl agresszív kockázatvállalást.

### 6.6 Thompson sampling (adaptív stratégia)
- A stratégiák sikerességét **Beta eloszlás** modellezi.
- A döntésnél véletlen mintavétel történik, így a felfedezés megmarad.

---

## 7) Példák (kibővítve)

### 7.1 Indikátoros jel példa
- EMA keresztezi az SMA-t felfelé → BUY jelzés.
- RSI 30 alulról átlépi → BUY megerősítés.
- ADX > 25 → trend megerősítés.

**Eredmény:** STRONG BUY ajánlás magasabb bizalommal.

### 7.2 Risk parity példa
- Két eszköz: A vol = 10%, B vol = 20%.
- Inverz súly: A = 10, B = 5 → normalizálva: A = 66.7%, B = 33.3%.

### 7.3 Korrelációs limit példa
- Korreláció 0.9 két BUY eszköz között → gyengébb eszköz súlya feleződik.

### 7.4 No-trade példa
- Bizalom 0.2 < küszöb → no-trade döntés, indoklással.

---

## 8) Kimenetek és riportok

- **Ajánlások**: ticker, signal, confidence, allocation.
- **Döntés-magyarázatok**: indikátor-alapú okok.
- **Audit**: megbízhatóság, drift, konszenzus, minőség.
- **Riportok**: backtest, drawdown, win-rate, Sharpe, trendek.

---

## 9) Limitációk és biztonsági megjegyzések

- **Nem hajt végre valós kereskedést.**
- Ajánlás jellegű, emberi döntést igényel.
- Minden döntés auditálható és visszakereshető.
- Tesztkörnyezetben és Raspberry Pi célplatformon validált.

---

## 10) Gyors navigáció

- Fő dokumentáció: [README_HU.md](README_HU.md)
- GYIK: [FAQ.md](FAQ.md)
- Hibaelhárítás: [TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)
- RPi telepítés: [deployment/RASPBERRY_PI_SETUP_GUIDE_HU.md](deployment/RASPBERRY_PI_SETUP_GUIDE_HU.md)
