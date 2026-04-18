---
mode: agent
description: RL modell (DQN vagy PPO) betanítása vagy újratanítása egy adott tickerre, eredmény validálással.
---

Tanítsd be (vagy tanítsd újra) az RL modellt a Tozsde rendszerben.

**Ticker:** ${input:ticker:pl. VOO}
**Modell típus:** ${input:model_type:DQN vagy PPO – hagyd üresen mindkettőhöz}

## Végrehajtandó lépések

### 1. RL betanítás
```bash
# Ha mindkét modellt kell tanítani:
python main.py train-rl ${input:ticker:VOO}

# Ha csak specifikus modellt:
# Módosítsd a train-rl parancsot a main.py-ban szükség esetén
```

### 2. Walk-forward validáció
```bash
python main.py walk-forward ${input:ticker:VOO}
```

### 3. Modell metaadatok ellenőrzése
Ellenőrizd a `models/` könyvtárban:
- `dqn_model_${input:ticker:VOO}_*.zip` – modelfájl
- `dqn_model_${input:ticker:VOO}_*.meta.json` – metaadatok (performance, training_date)
- `ppo_model_${input:ticker:VOO}_*.zip`
- `ppo_model_${input:ticker:VOO}_*.meta.json`

### 4. Post-training validáció
```bash
python main.py validate --ticker ${input:ticker:VOO} --start-date 2022-01-01 --end-date 2023-12-31
```

## Értékeld az eredményt
- Walk-forward stabilitás (drift < 20%)
- Model trust score (cél: > 0.6)
- Phase 6 promotion gate státusz
- TensorBoard log: `tensorboard --logdir tensorboard/`
