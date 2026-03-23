# Compatibility Contract Freeze (Phase 7.1)

Date: 2026-03-21
Status: Active
Scope: Freeze currently required compatibility surfaces before deeper shim retirement.

## Purpose

This document defines the import-stable and monkeypatch-stable compatibility contracts that must remain unchanged unless tests and migration plan are updated together.

## Frozen Contract Surfaces

### 1) UI compatibility facade

Module: app/ui/app.py

Required symbols (test monkeypatch targets):
- DataManager
- get_supported_tickers
- render_template
- load_data
- sanitize_dataframe
- compute_signals
- get_params
- Backtester
- get_equity_curve_buffer
- get_drawdown_curve_buffer

Notes:
- Runtime wiring is interfaces-first, but this facade remains patchable for route-test compatibility.
- Symbol names and module path are part of the contract.
- Implementation ownership: helper logic is adapterized in `app/interfaces/compat/ui_contract.py`; facade path and symbol names remain the contract.

### 2) Root CLI wrapper surface

Module: main.py

Required symbols:
- parse_arguments
- run_daily
- run_weekly
- run_monthly
- run_walk_forward_manual
- run_train_rl_manual
- run_validation
- run_paper_history
- _SETTINGS

Notes:
- Runtime execution delegates to bootstrap-wired use-cases.
- Wrapper names remain stable for compatibility tests.

### 3) Legacy DataManager boundary

Module: app/infrastructure/repositories/data_manager_repository.py

Required behavior:
- This is the only allowed direct legacy DataManager import boundary in app runtime code.
- It remains a compatibility adapter until shim retirement.

### 4) Known dynamic compatibility imports (intentionally retained)

- app/ui/admin_dashboard.py:
  - app.decision.capital_optimizer.CapitalUtilizationOptimizer
  - app.decision.decision_history_analyzer.DecisionHistoryAnalyzer
  - app.decision.confidence_allocator.ConfidenceBucketAllocator
- app/reporting/audit_builder.py:
  - app.decision.drift_detector.PerformanceDriftDetector
- app/infrastructure/cron_tasks.py:
  - app.decision.decision_engine.DecisionEngine

Notes:
- These remain for monkeypatch compatibility and will be handled under Phase 7 adapterization.

## Change Protocol

Any change touching the frozen surfaces requires all of:
1. Update this document.
2. Update docs/Architecture_migration_plan.md progress notes.
3. Update/extend tests that rely on monkeypatch contracts.
4. Keep full regression green.

## Validation Gate

- tests/test_phase7_contract_freeze.py enforces this freeze contract at code level.
