---
mode: agent
description: Teljes backtesting és validációs futtatás adott tickerre és időszakra, eredmények összefoglalásával.
---

Futtasd le a teljes validációs és backtesting pipeline-t az alábbi paraméterekkel:

**Ticker:** ${input:ticker:pl. VOO}
**Start date:** ${input:start_date:pl. 2022-01-01}
**End date:** ${input:end_date:pl. 2023-12-31}

## Végrehajtandó lépések

### 1. Paper history backtest
```bash
python main.py run-paper-history --ticker ${input:ticker:VOO} --start-date ${input:start_date:2022-01-01} --end-date ${input:end_date:2023-12-31}
```

### 2. Validation (Phase 5+6)
```bash
python main.py validate --ticker ${input:ticker:VOO} --start-date ${input:start_date:2022-01-01} --end-date ${input:end_date:2023-12-31}
```

### 3. Governance riport
```bash
python main.py governance --mode full
```

### 4. Tesztek + validáció együtt
```bash
python scripts/run_tests_with_report.py --skip-tests --with-validation --ticker ${input:ticker:VOO} --start-date ${input:start_date:2022-01-01} --end-date ${input:end_date:2023-12-31}
```

## Utólag ellenőrizd
- `diagnostics/` – legújabb data_integrity és pipeline_audit JSON-ok
- `reports/` – legújabb governance riport mappa
- Összefoglald: Phase 5 és Phase 6 státusz, promotion gate eredménye, kritikus hibák
