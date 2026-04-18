---
mode: ask
description: Technikai indikátor hozzáadása az app/indicators/ modulhoz, teszttel és integrációval.
---

Adj hozzá egy új technikai indikátort a Tozsde rendszer `app/indicators/` moduljához.

**Indikátor neve:** ${input:indicator_name:pl. StochasticOscillator}
**Rövid neve/azonosítója:** ${input:short_name:pl. STOCH}
**Számítás leírása:** ${input:description:pl. %K és %D vonal, lookback periódussal}
**Paraméterek:** ${input:params:pl. period: int = 14, smooth_k: int = 3}

## Implementálandó

### 1. Indikátor függvény
Helye: `app/indicators/${input:short_name:indicator}.py`

```python
import numpy as np
import pandas as pd
from typing import Optional

def calculate_${input:short_name:indicator}(
    data: pd.DataFrame,
    ${input:params:period: int = 14}
) -> pd.Series:
    """
    Calculates ${input:indicator_name}.
    Expected columns: open, high, low, close, volume (lowercase)
    Returns: pd.Series with NaN for insufficient data
    """
    ...
```

### 2. Export az `app/indicators/__init__.py`-ban
Adj hozzá export sort.

### 3. Unit teszt
Helye: `tests/test_indicator_${input:short_name:indicator}.py`

Tesztelendő esetek:
- Happy path (normál adatokkal)
- Insufficient data (kevés sor)
- NaN handling (adatban NaN értékek)
- Edge case (mind egyforma ár)

## Szabályok
- NumPy/pandas számítás – no external libs
- NaN-t adj vissza ha nincs elég adat (ne dobj exceptiönt)
- Lowercase column names: `open`, `high`, `low`, `close`, `volume`
- Típusjelölések kötelezők
