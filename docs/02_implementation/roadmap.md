
---

# 📦 2) **roadmap.md — Snapshot-friendly verzió (tökéletesített, hierarchikus)**

Ez a fájl kapott egy **egyértelmű főcímet**, stabil szerkezetet, egyenletes H2/H3 szintű bontást, és minimális kötőszöveget, hogy biztosan dokumentumként legyen felismerve.

---

```md
# MASTER ROADMAP  
**Python-alapú Tőzsdei Ajánló & Backtesting Rendszer — P0 → P9**

Ez a dokumentum tartalmazza a teljes rendszer hosszú távú fejlesztési útitervét.  
A cél egy stabil, determinisztikus, Synology NAS-on futó, alacsony karbantartási igényű tőzsdei ajánlórendszer, amely technikai indikátorokra, walk-forward optimalizációra és ML-re épül.

---

## P0 — Kritikus Stabilitás és Hibajavítás (MUST HAVE)

### Problémák
- Cron alatti relatív útvonalhibák  
- Duplikált ajánlások  
- Flask input validáció hiánya  
- Silent cron failure  
- Nem biztonságos SECRET_KEY

### Megoldások
- pathlib alapú abszolút útvonalak  
- Központosított state könyvtár: `app/data`  
- SQLite `(ticker, date)` alapú UPSERT  
- Validált API bemenetek  
- Cron-level logging + exception handling  
- SECRET_KEY environment policy

### Hatás
- 0 silent failure  
- determinisztikus működés  
- konzisztens adatállapot  

---

## P1 — Adatmodell és Adatkezelés (FOUNDATION)

### Problémák
- Normalizálatlan OHLCV  
- Lassú lekérdezések  
- Ad-hoc DataFrame → DB folyamatok

### Megoldások
- Egységes DB séma  
- `ohlcv(ticker, date)` PRIMARY KEY  
- Indexek ticker + date  
- DAO/DataManager réteg  
- Standardizált load/save API

### Hatás
- Gyorsabb backtest  
- Egységes adatforrás  
- Könnyebb migrálhatóság  

---

## P2 — Teljesítmény és Realisztikus Szimuláció (EDGE)

### Problémák
- Lassú indikátorszámítás  
- Túl optimista backtest  
- GA overfitting kockázat

### Megoldások
- Indikátor- és fitnesz-cache  
- Tranzakciós költségmodell (commission, slippage, spread)  
- Realisztikus backtester  
- Teljesítménymutatók: Total Return, Sharpe, Max Drawdown  
- Fitness memoization  
- Overfitting elleni büntetések

### Hatás
- 3–10× gyorsabb optimalizáció  
- Stabilabb paraméterek  
- Realisztikus profitvárakozások  

---

## P3 — UX, Riport és Visszacsatolás (OBSERVABILITY)

### Problémák
- Nem látszik, miért lett egy ajánlás  
- Nem összehasonlítható backtestek

### Megoldások
- Flask riport oldal  
- Equity curve  
- Teljesítménymutatók  
- Paramétervizualizáció  
- Egységes riport pipeline

### Hatás
- Erősebb döntéstámogatás  
- Magasabb felhasználói bizalom  

---

## P4 — Automatizálás, Audit és Ellenőrzés (OPS)

### Cél
A rendszer üzembiztos hosszú távon.

### Megoldások
- `apply_schema()` — DB séma verziózás  
- Smoke test  
- Daily pipeline  
- HistoryStore (append-only modell)  
- Audit metadata  
- Quality score  
- Consistency flags  
- Email/riport audit adatokkal

### Hatás
- Biztonságos deploy  
- Gyors hibaészlelés  
- Teljes döntési audit trail  

---

## P5 — Walk-Forward & Időbeli Validáció (PROFIT GUARD)

### Cél
A rendszer ne csak „múltban működjön”.

### Megoldások
- Rolling train/test ablakok  
- Walk-forward score  
- Időbeli aggregálás  
- Csak múltbeli paraméterek használata

### Hatás
- Hamis profit kiszűrése  
- Stabilabb ajánlások  
- Reálisabb viselkedés  

---

## P6 — Confidence & Allokáció Logika (CAPITAL EFFICIENCY)

### Cél
Optimális döntések kevés tőkéből is.

### Megoldások
- Confidence score  
- Confidence bucket  
- NO-TRADE logika  
- Strength kategóriák (STRONG/NORMAL/WEAK)  
- Alapszintű allokációs modell

### Hatás
- Drawdown csökkenés  
- Tőkefelhasználás javulása  

---

## P7 — Portfólió-szintű Optimalizáció (SCALING)

### Cél
Ne legyen minden ticker izolált.

### Megoldások
- Korreláció-elemzés  
- Kockázatparitás  
- ETF + részvény mix  
- Portfólió-szintű allokáció

### Hatás
- Csökkenő volatilitás  
- Stabilabb teljesítmény  

---

## P8 — Tanuló Rendszer & Visszacsatolás (LONG-TERM MOAT)

### Cél
A rendszer idővel okosodjon.

### Megoldások
- Döntési history elemzése  
- Reliability score  
- Performance drift figyelés  
- RL agent stratégia-választásra (nem trade-re)  
- PyFolio integráció (rolling performance, drawdown profile)

### Hatás
- Adaptív rendszer  
- Piaci rezsimek felismerése  
- Hosszú távú edge  

---

## P9 — Engineering & Product Hardening (FINAL POLISH)

### Cél
Szoftverérett minőség.

### Megoldások
- Unit tesztek (indikátorok, WF, fitness)  
- Integration tesztek  
- UI/UX finomítás  
- Admin dashboard  
- Error visibility & monitoring

### Hatás
- Könnyebb karbantartás  
- Biztonságos refaktorálás  

---

## Összegzés

- **P0–P4:** stabil alap, éles rendszerszint  
- **P5–P6:** profit + kockázat kontroll  
- **P7–P8:** portfólió és tanulás  
- **P9:** termékszerű minőség  