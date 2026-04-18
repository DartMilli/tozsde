---
applyTo: "tests/**/*.py"
---

## Tesztelési szabályok – Tozsde projekt

### Általános
- Minden teszt izolált – ne függjön más teszt futási sorrendjétől
- yfinance és külső API hívásokat **mindig** mockold: `@patch('app.data.loader.yfinance.download')`
- Közvetlen SQLite fájl helyett mindig `test_db` fixture-t használj
- Assertion üzeneteket írj: `assert result.is_ok(), f"Expected ok, got: {result.error}"`

### Fixture-ok (`tests/conftest.py`)
- `test_db` – izolált in-memory SQLite, automatikus teardown
- `sample_ohlcv` – 60 soros VOO OHLCV DataFrame (2023-01-01-től)
- `mock_config` – mock Settings objektum éles értékek nélkül

### Elnevezési konvenció
```
test_<entitás>_<forgatókönyv>_<elvárás>
# pl:
test_decision_policy_buy_signal_returns_ok
test_data_manager_invalid_ticker_returns_error
test_allocator_zero_capital_returns_error
```

### Result pattern assertion-ök
```python
# ✅ Helyes
result = use_case.execute(ticker="VOO")
assert result.is_ok()
assert result.value["action"] == "BUY"

# ✅ Hiba esetén
result = use_case.execute(ticker="INVALID")
assert result.is_error()
assert "No data" in result.error

# ❌ Kerülendő
assert result  # nem egyértelmű
```

### Coverage cél: 91%+ fenntartása
```bash
pytest --cov=app --cov-report=term-missing -q
```
