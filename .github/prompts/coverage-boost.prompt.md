---
mode: ask
description: Meglévő modul tesztjeinek átvilágítása és hiányzó lefedettség pótlása.
---

Vizsgáld át a következő modul tesztlefedettségét és egészítsd ki a hiányzó tesztekkel:

**Modul:** ${input:module_path:pl. app/decision/decision_policy.py}

## Feladat

### 1. Lefedettség mérés
```bash
pytest tests/ --cov=${input:module_path} --cov-report=term-missing -q
```

### 2. Vizsgáld meg
- Mely ágak (branches) nincsenek lefedve?
- Mely edge case-ek hiányoznak?
- Vannak-e untested error path-ok (Result.error visszatérési ágak)?

### 3. Írj teszteket a hiányokra
Fixture-ok elérhetők a `tests/conftest.py`-ban:
- `test_db` – izolált SQLite adatbázis
- `sample_ohlcv` – minta OHLCV DataFrame
- `mock_config` – mock Settings objektum

### Mintastruktúra
```python
class Test${input:module_path:Module}:
    def test_<scenario>_returns_ok(self, test_db, sample_ohlcv):
        result = module_function(valid_input)
        assert result.is_ok()
        assert result.value == expected

    def test_<scenario>_returns_error_on_<condition>(self):
        result = module_function(invalid_input)
        assert result.is_error()
        assert "expected message" in result.error
```

### 4. Futtasd és ellenőrizd
```bash
pytest tests/test_<modul>.py -v
pytest --cov=app --cov-report=term-missing -q | tail -20
```

## Cél: 91%+ coverage fenntartása, ideálisan növelése
