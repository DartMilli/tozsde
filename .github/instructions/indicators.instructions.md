---
applyTo: "app/indicators/**/*.py"
---

## Indikátor modul – szabályok

### Adatformátum
- Input: `pd.DataFrame` lowercase oszlopnevekkel: `open`, `high`, `low`, `close`, `volume`
- Output: `pd.Series` (single indikátor) vagy `pd.DataFrame` (több vonal, pl. MACD)
- Ha nincs elég adat: adj vissza `NaN`-okat – **ne dobj exceptiönt**

### Kötelező minták
```python
import numpy as np
import pandas as pd

def calculate_sma(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """Returns SMA with NaN for insufficient data rows."""
    if len(data) < period:
        return pd.Series(np.nan, index=data.index)
    return data["close"].rolling(window=period).mean()
```

### Tilos
- Teljes pandas/numpy könyvtáron kívüli ML lib indikátor számításhoz
- `data["Close"]` nagybetűs – mindig `data["close"]`
- Üres/None DataFrame elfogadása ellenőrzés nélkül
