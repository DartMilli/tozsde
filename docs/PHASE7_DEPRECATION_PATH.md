# Phase 7 Deprecation Path

Date: 2026-03-21
Updated: 2026-04-09 (Sprint 14)
Status: Planned (post-freeze, controlled rollout)

## Goal

Retire compatibility shims safely after contracts are no longer required by tests and runtime consumers.

## Current shim boundaries

- UI facade contract: app/ui/app.py (implementation adapterized in app/interfaces/compat/ui_contract.py)
- Main wrapper contract: main.py (implementation adapterized in app/interfaces/compat/main_contract.py)
- Legacy persistence boundary: app/infrastructure/repositories/data_manager_repository.py
- **`app/decision/` pass-through layer** ← Sprint 14 (A3): minden fájl DEPRECATED docstringet kapott; valódi implementáció: `app/core/decision/`

## `app/decision/` deprecation status (Sprint 14)

25 passthrough fájl az `app/decision/` könyvtárban – mindegyiken `"""DEPRECATED: compatibility shim – use app.core.decision directly."""` modul-docstring.

Kivonás feltételei:
1. Minden teszt és runtime import `app.core.decision.*`-t hivatkozzon (ne `app.decision.*`)
2. `app/interfaces/compat/` ne importáljon `app/decision/`-ből
3. `app/services/trading_pipeline.py` és `app/bootstrap/bootstrap.py` direktben importáljon `core`-ból
4. Teljes teszt suite zöld marad

Ajánlott kivonási sorrend: `ensemble_aggregator.py`, `ensemble_quality.py`, `weighting.py` → `safety_rules.py`, `risk_parity.py` → `allocation.py`, `capital_optimizer.py` → komplex wrapperek (`recommender.py`, `decision_builder.py`)

## `production_score` komponensek (walk_forward.py – Q7 dokumentáció)

A `production_score` a walk-forward értékelés összesített minőségi mutatója, négy egyenlő súlyú komponensből áll:

```
production_score = 0.4 × sharpe_ratio_normalized
                 + 0.2 × stability_score
                 + 0.2 × robustness_score
                 + 0.2 × (1 - max_drawdown_normalized)
```

- **sharpe_ratio_normalized** (40%): kockázatkorrigált hozam (annualizált Sharpe)
- **stability_score** (20%): hozam konzisztenciája időablakok között
- **robustness_score** (20%): stratégia stabilitása különböző paraméterbeállítások mellett
- **1 - max_drawdown_normalized** (20%): tőke megőrzése (alacsonyabb drawdown = magasabb pont)



## Preconditions for retirement

1. Replace monkeypatch targets in tests with explicit adapter/module targets.
2. Keep all route/integration and coverage tests green.
3. Keep closure gate tests green.
4. Update compatibility freeze document and migration plan before each removal.

## Recommended retirement order

1. Dynamic imports in app/ui/admin_dashboard.py and app/reporting/audit_builder.py
2. Dynamic DecisionEngine import in app/infrastructure/cron_tasks.py
3. app/ui/app.py symbol-level shims (after tests migrate to adapter module)
4. main.py wrappers (after callers migrate to adapter functions or CLI command invocation)
5. Final removal of legacy DataManager fallback from data_manager_repository adapter

## Rollback strategy

- Re-introduce previous facade exports at same module path and symbol names.
- Re-run full test suite immediately.
- Re-open contract freeze item with reason and date.
