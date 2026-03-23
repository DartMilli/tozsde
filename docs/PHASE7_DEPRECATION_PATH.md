# Phase 7 Deprecation Path

Date: 2026-03-21
Status: Planned (post-freeze, controlled rollout)

## Goal

Retire compatibility shims safely after contracts are no longer required by tests and runtime consumers.

## Current shim boundaries

- UI facade contract: app/ui/app.py (implementation adapterized in app/interfaces/compat/ui_contract.py)
- Main wrapper contract: main.py (implementation adapterized in app/interfaces/compat/main_contract.py)
- Legacy persistence boundary: app/infrastructure/repositories/data_manager_repository.py

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
