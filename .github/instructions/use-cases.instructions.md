---
applyTo: "app/application/use_cases/**/*.py"
---

## Use Case réteg – kötelező szabályok

- Minden use case osztály `execute()` metódusa **`Result`** típust ad vissza
- Import: `from app.application.use_cases.result import ok, error, Result`
- **Soha ne dobj** `raise Exception(...)` üzleti logikában – mindig `return error(...)`
- Konstruktor paramétereit a `build_application()` bootstrap container injektálja
- Típusjelölések minden publikus metóduson kötelezők
- Naplózás: `logger = setup_logger(__name__)` az `app/infrastructure/logger`-ből

```python
# Helyes minta
class MyUseCase:
    def __init__(self, repo: MyRepo, settings: Settings) -> None:
        self._repo = repo
        self._settings = settings

    def execute(self, ticker: str) -> Result:
        try:
            data = self._repo.get(ticker)
            if data is None:
                return error(f"No data for {ticker}")
            return ok(data)
        except Exception as e:
            logger.error(f"MyUseCase failed: {e}")
            return error(str(e))
```
