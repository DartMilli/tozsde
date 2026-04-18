# Contributing to Tozsde

## Ág-naming konvenció
- `feature/<rövid-leírás>` – új funkció
- `fix/<issue-vagy-leírás>` – hibajavítás
- `refactor/<terület>` – refaktorálás
- `docs/<mit>` – dokumentáció

## PR workflow
1. Fork vagy branch a `main`-ről
2. Implementáció a [Clean Architecture](docs/architecture_codeing_guidelines.md) elvek szerint
3. Tesztek: `pytest tests/ -x -q` – minden teszt zöld kell legyen
4. Coverage nem csökkenhet a jelenlegi szint alá: `pytest tests/ --cov=app --cov-report=term-missing -q`
5. PR létrehozása – title: `[fix|feat|refactor|docs] rövid leírás`
6. Code review: legalább 1 jóváhagyás szükséges

## Kódolási elvek
Architecture quick summary (extracted from `docs/architecture_codeing_guidelines.md`)
- Layers: `interfaces` → `application` → `core` → `infrastructure` (and `config/bootstrap`). Keep the layering strict.
- Dependency rules: only allow the documented one-way edges (e.g. `interfaces` may call `application`, `application` may call `core` and `infrastructure`, `bootstrap` may call `application` + `infrastructure`). Do NOT import downwards.
- `core` restrictions (must be pure domain logic):
	- No Flask or HTTP frameworks
	- No direct SQLite / DB drivers
	- No direct Config/env reads or global settings access
	- No logging side-effects or infrastructure concerns
	- Only Protocols/Interfaces (type hints) may appear here; implementations belong to `infrastructure`.
- Configuration: environment reads only in `app/config/build_settings.py`; pass a frozen `Settings` dataclass via DI to services/constructors.
- Repositories: define repository Protocols in `core`, implement them in `infrastructure`, inject implementations via the bootstrap container.
- Service construction: all application services must use explicit constructor DI (no implicit/global singletons, no service-locators).
- Forbidden/anti-patterns to avoid: Service Locator, God Objects, circular-dependency workarounds (lazy imports hide architecture issues), side-effectful imports.
- File & style limits (guideline): keep modules small and focused (file length and import-count limits in the full doc).

See `docs/architecture_codeing_guidelines.md` for the complete, authoritative rules and examples. This short summary is intended only as a quick reminder for contributors; follow the full doc for design and CI checks.

## Commit üzenetek
```
<típus>(<modul>): <rövid leírás>

[opcionális részletes leírás]
```
Példák:
- `fix(optimizer): abs(max_drawdown) a fitness függvényben`
- `feat(backtester): trade-frekvencia alapú Sharpe annualizáció`
