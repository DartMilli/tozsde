---
mode: ask
description: Új use case implementálása a Tozsde rendszerbe (Clean Architecture, Result pattern, teszttel együtt).
---

Implementálj egy új use case-t a Tozsde algoritmikus trading rendszerbe az alábbi spec szerint:

**Use Case neve:** ${input:use_case_name:pl. CalculatePortfolioRisk}
**Leírás:** ${input:description:Mit csinál ez a use case?}
**Input paraméterek:** ${input:inputs:pl. ticker: str, start_date: str}
**Várható output:** ${input:output:pl. risk_score: float, max_drawdown: float}

## Amit létre kell hoznod

### 1. Use Case fájl
Helye: `app/application/use_cases/${input:use_case_name:use_case}.py`

```python
from app.application.use_cases.result import ok, error, Result
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)

class ${input:use_case_name}:
    def __init__(self, ...):  # függőségek bootstrap-ből
        ...

    def execute(self, ...) -> Result:
        try:
            # implementáció
            return ok(result)
        except Exception as e:
            logger.error(f"${input:use_case_name} failed: {e}")
            return error(str(e))
```

### 2. Bootstrap regisztráció
Helye: `app/bootstrap/bootstrap.py` – adj hozzá a containerhez

### 3. Unit teszt
Helye: `tests/test_${input:use_case_name:use_case_name}.py`

## Szabályok
- Kötelező Result pattern (ok/error)
- Kötelező típusjelölések
- Ne hívj yfinance-t közvetlenül – csak `app/data/` réteget
- Ne módosítsd az `app/interfaces/compat/` könyvtárat
