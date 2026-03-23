# Train-RL Checklist (HU)

Celjai:
- Minimalis, teszteleshez elegendo modellek letrehozasa (dontos, szavazas, explainability, email, outcome tesztekhez).
- Teljes sweep terv (osszes ticker, model tipus, reward strategia).
- CI/deploy iranyelvek: mikor kell ujratanitas, mikor nem; greenfield telepites menete.

## Forrasok (kod)
- CLI es pipeline: main.py (daily/weekly/monthly/walk-forward/train-rl)
- RL training: app/models/model_trainer.py (train_rl_agent, run_training, run_backtest)
- Walk-forward + GA: app/backtesting/walk_forward.py -> app/optimization/genetic_optimizer.py
- Ticker lista: app/data_access/data_loader.py + app/config/config.py
- Ensemble szavazas: app/decision/recommender.py + app/decision/ensemble_aggregator.py

## Alapertelmezett tickerek
A rendszer a kovetkezo tickereket hasznalja (kizartak nelkul):
- VOO, VTI, QQQ, SPY, SCHD, RSP, SMH, GLD
Kizart (nem futnak): OTP.BD, MOL.BD, RICHTER.BD

## Milyen modelleket kell tanitani?
- RL modellek: DQN es PPO tamogatott.
- Reward strategiak: TradingEnv.get_reward_strategies() listabol jon (tobb opcio).
- Ensemble szavazashoz tobb modell kell (legalabb 2).

## Minimalis teszteleshez szukseges training
Cel: legyenek valodi modellek, legyen szavazas, dontes, email, es legyen esely trade-re.

### Minimalis kombinacio (ajanlott)
- 1 ticker: VOO vagy SPY
- 2 model tipus: DQN + PPO
- 1 reward strategia: portfolio_value
- Top_N az ensemble-ben: 3 (ha van, de 2 mar eleg a szavazas teszthez)

### Lepesek (minimalis)
1) Elokeszites
   - Internet elerheto (yfinance letoltes)
   - requirements.txt telepitve
   - models/ mappa ures vagy rendezett

2) Adat letoltes/cache
   - A GA/RL futas elott automatikus cache-ellenorzes van (Config.START_DATE/END_DATE).
   - Ha a letoltes hianyos, a futas leall es a cache-t kulon kell javitani.

3) Walk-forward (GA) - kotelezo RL elott
   - python main.py walk-forward VOO
   Megjegyzes: a walk-forward belul GA-t futtat, es a default intervallumot hasznalja (Config.START_DATE/END_DATE).

4) Train-RL (manualis)
   - python main.py train-rl VOO
   Ez alapbol a model_trainer train_rl_agent-et hasznalja (DQN default).

5) Masik model tipus (PPO)
   - Hasznald a kulon CLI-t a model tipussal es reward strategy-val:
     - python scripts/train_rl.py --ticker VOO --model-type PPO --reward-strategy portfolio_value
     - python scripts/train_rl.py --ticker VOO --model-type DQN --reward-strategy portfolio_value

6) Ellenorzes
   - models/ alatt legyen .zip + .meta.json
   - futtass daily/paper run-t, hogy legyen modell vote es dontes

### Minimalis tesztelesi celok
- Dontes generikodas model vote-okkal
- Ensemble aggregation mukodese
- Explainability lint warning nelkul
- Email summary/detail letrejon
- Outcome linkage teszt csak akkor, ha tenyleges trade van

## Teljes sweep checklist (minden model, minden ticker)
Cel: teljes lefedes, top modellek kivalasztasa, promotion gate ellenorzes.

1) Elokeszites
   - megfelelo CPU/GPU
   - elegendo tarhely (sok modell fajl)
   - adatcache a teljes ticker listara

2) Walk-forward minden tickerre (GA)
   - python scripts/full_sweep.py
   (a full sweep a walk-forwardot elobb futtatja, majd RL training)

3) Teljes RL training sweep
   - Futtasd a scripts/full_sweep.py-t (run_training + backtest)
   - Ez vegigmegy: tickerek x reward strategiak x (PPO + DQN)

4) Backtest es top model valasztas
   - run_backtest() -> top model(ek) masolasa
   - A top model(ek) bekerulnek top1/top2... neven a models/ mappaba

5) Regisztracio / promotion gate
   - model metadata es DB register ellenorzes

## Idobecsles (durva)
- 1 ticker, 1 model (100k step): 15-60 perc (CPU)
- 1 ticker, DQN+PPO, 1 reward: 30-120 perc
- Walk-forward (GA) 1 ticker: 1-4 ora
- Teljes sweep (minden ticker, minden reward): 1-5 nap (gep fuggo)

## CI / Deploy felkeszites (training csak ha szukseges)
Celjai:
- Ne traineljen minden commit utan.
- Traineljen, ha:
  a) greenfield telepites (nincs modell)
  b) model relevans kod valtozott (feature/parameterek)

### Dontesi logika (javaslat)
Train szukseges, ha barmelyik igaz:
- models/ ures vagy nincs model .zip
- Training fingerprint valtozott

### Training fingerprint (javaslat)
Hozz letre egy JSON-t, pl. training_fingerprint.json, ami tartalmazza:
- Hash a kovetkezo fajlokrol/mappakrol:
  - app/models/
  - app/decision/
  - app/analysis/
  - app/data_access/
  - app/optimization/
  - app/backtesting/walk_forward.py
  - app/config/config.py
  - requirements.txt
- Ticker lista
- RL_TIMESTEPS, OPTIMIZER_POPULATION, OPTIMIZER_GENERATIONS
- GA-before-RL policy + training intervallum (START_DATE/END_DATE)

Ha az uj fingerprint elter a mentettol -> train szukseges.

Hasznalat (uj script):
- python scripts/training_fingerprint.py --check
- python scripts/training_fingerprint.py --write

Summary file:
- models/training_fingerprint_summary.json (tickerek + reward strategiak listaja)

### CI workflow (javaslat)
- Manualis workflow_dispatch: train-rl, walk-forward, full-sweep
- CI job csak akkor fut, ha:
  - input.force_train=true, vagy
  - models/ ures, vagy
  - fingerprint valtozott

Javasolt workflow: .github/workflows/train_models.yml

### Greenfield telepites menete
1) requirements.txt telepites
2) adatcache (teljes ticker lista)
3) walk-forward minden tickerre
4) RL training (legalabb minimalis set)
5) backtest + promote best models
6) daily/paper run smoke teszt

### Non-greenfield (mar vannak modellek)
- Csak akkor trainelni, ha:
  - uj feature erkezett a decision/analysis/models pipeline-ban
  - reward strategia vagy feature set valtozott
  - config parameterek valtoztak (RL_TIMESTEPS, GA parameterek)

## Minimalis vs Full sweep - osszefoglalo
Minimalis:
- 1 ticker, 2 model (DQN+PPO), 1 reward strategia
- Cel: dontes + szavazas + explainability + email + smoke

Full sweep:
- osszes ticker, osszes reward strategia, DQN+PPO
- backtest + top model promote
- hosszabb futas es nagy eroforras igeny
