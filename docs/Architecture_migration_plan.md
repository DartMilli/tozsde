ARCHITECTURE MIGRATION PLAN
Target: Modular Monolith optimized for Raspberry 5
0. OBJECTIVE

This document defines the complete architectural migration from the current research-style monolithic structure to a clean modular monolith architecture with:

Clear layer separation

Dependency injection

Repository abstraction

No global Config imports

No God DataManager

Production-ready single-node design

Target runtime: Raspberry 5
Single process model
SQLite remains

1. TARGET ARCHITECTURE OVERVIEW
1.1 High-Level Layer Structure
app/
    core/
    application/
    infrastructure/
    interfaces/
    config/
    bootstrap/

Layer Rules

Allowed dependencies:

interfaces -> application
application -> core
application -> infrastructure
interfaces -> application -> infrastructure


Forbidden:

core -> infrastructure
core -> config
core -> flask
application -> flask

2. MIGRATION STRATEGY

Migration is planned in 6 implementation phases + 1 closure hardening phase.

Each phase must:

Keep tests passing

Be committed separately

Avoid large uncontrolled refactors

PHASE 1 - PREPARATION
1.1 Create Target Folder Structure

Create empty folders:

app/core
app/application
app/infrastructure
app/interfaces
app/bootstrap


Add __init__.py to all.

Do NOT move anything yet.

PHASE 2 - CONFIG DECOUPLING
Objective

Eliminate global Config import usage inside business logic.

2.1 Create Settings Dataclass

Create:

app/config/settings.py


Content:

Define @dataclass(frozen=True)

Move all runtime configuration fields from Config into this dataclass

Do NOT include side effects

Example structure (no logic inside):

class Settings:
    db_path: Path
    email_user: str
    rl_timesteps: int
    ...

2.2 Create Config Builder

Create:

app/config/build_settings.py


Responsibilities:

Load env vars

Validate required fields

Create and return Settings instance

Ensure directories

No other module should read env vars directly after this.

2.3 Replace Global Config Imports

For each module that imports:

from app.config.config import Config


Replace:

Remove import

Add settings: Settings parameter to constructor or function

Pass settings from caller

DO NOT refactor all at once.

Refactor in this order:

TradingPipelineService

WalkForward

ModelTrainer

DataManager (temporary)

Remaining modules

Each step must pass tests.

PHASE 3 - INTRODUCE REPOSITORY LAYER
Objective

Break up DataManager into isolated repositories.

3.1 Create Repository Interfaces

Create:

app/core/ports/


Define Protocol interfaces:

class IOhlcvRepository(Protocol)
class IDecisionRepository(Protocol)
class IModelRepository(Protocol)
class IMetricsRepository(Protocol)


Each interface must only define necessary methods.

No SQLite references allowed here.

3.2 Create Infrastructure Implementations

Create:

app/infrastructure/repositories/


Implement:

sqlite_ohlcv_repository.py
sqlite_decision_repository.py
sqlite_model_repository.py
sqlite_metrics_repository.py


Initially:

Wrap existing DataManager methods internally

Delegate calls

DO NOT rewrite SQL yet.

3.3 Refactor Application Layer to Use Interfaces

For:

TradingPipelineService

WalkForwardService

RLTrainingService

Replace:

DataManager usage


With:

IOhlcvRepository
IDecisionRepository
...


Constructor injection required.

3.4 Gradually Shrink DataManager

After repositories work:

Move SQL logic from DataManager into respective repository

Remove duplicated methods

Remove analytics from DataManager

Eventually DataManager should disappear completely.

PHASE 4 - CORE LAYER EXTRACTION
Objective

Move pure domain logic into core layer.

4.1 Move Indicators

Move:

app/indicators/technical.py


Into:

app/core/indicators/


Split by domain:

trend.py

momentum.py

volatility.py

Remove legacy _old functions.

No Config usage allowed.

4.2 Move Decision Logic

Move:

app/decision/*


Into:

app/core/decision/


Ensure:

No repository imports

No SQLite

No Flask

No Config

4.3 Move Allocation & Safety

Into:

app/core/risk/


Ensure pure logic only.

PHASE 5 - APPLICATION LAYER RESTRUCTURE
Objective

Turn orchestration into use-case driven services.

5.1 Create Use Case Services

Create:

app/application/use_cases/


Add:

run_daily_pipeline.py
run_walk_forward.py
train_rl_model.py
validate_model.py


Each:

Accept repositories + settings via constructor

Orchestrate core logic

Return structured result objects

No direct printing.
No CLI logic.

5.2 Refactor TradingPipelineService

Split into:

DailyPipelineUseCase

ExecutionCoordinator

NotificationCoordinator

Keep files under 300 lines.

PHASE 6 - BOOTSTRAP (COMPOSITION ROOT)
Objective

Centralize dependency wiring.

6.1 Create bootstrap.py

Location:

app/bootstrap/bootstrap.py


Responsibilities:

Build settings

Instantiate repositories

Instantiate use cases

Return configured services

Example structure:

def build_application():
    settings = build_settings()
    ohlcv_repo = SqliteOhlcvRepository(settings)
    decision_repo = SqliteDecisionRepository(settings)
    ...
    daily_pipeline = DailyPipelineUseCase(...)
    return ApplicationContainer(...)

6.2 Refactor main.py

main.py should:

Import bootstrap

Call use case

Handle CLI parsing only

main.py must NOT import:

repositories

core modules directly

6.3 Refactor Flask app

Move Flask entrypoint to:

app/interfaces/web/app.py


Flask should:

Call use cases

Not call repositories directly

PHASE 7 - POST-MIGRATION HARDENING & SHIM RETIREMENT
Objective

Close migration by converting compatibility boundaries into explicit, test-owned interfaces.

7.1 Freeze compatibility contracts

Define and document the remaining compatibility surface that is still intentionally patchable for tests:

- `app/ui/app.py` symbols used by monkeypatch-based route tests
- root `main.py` legacy wrapper entrypoints
- migration adapter boundary in `app/infrastructure/repositories/data_manager_repository.py`

7.2 Reduce compatibility surface behind explicit adapters

Move remaining compatibility helpers behind dedicated adapter modules and keep `app/ui/app.py` / `main.py` as import-stable facades only.

7.3 Add migration completion gates

Add and enforce closure checks:

- No new direct repository/domain wiring in entrypoints
- Full regression remains green
- Architecture migration plan reflects final ownership map

7.4 Define final deprecation path

Prepare a controlled plan for optional future removal of compatibility shims after tests no longer require monkeypatch contracts.

---

CONFIG MIGRATION STATUS (2026-02-21)

- All direct runtime usages of global Config have been replaced with get_conf(settings) or get_conf(None) in all business logic, analysis, governance, services, and backtesting modules.
- Legacy Config alias is preserved in each module for test compatibility (monkeypatching).
- Tests that patch module-level Config continue to work (see walk_forward test compatibility fix).
- Migration was performed incrementally, package by package, with tests passing after each sweep.
- All tests pass, including those that monkeypatch Config.
- No module imports global Config for runtime logic.
- Migration checklist:
    - app/validation: complete
    - app/governance: complete
    - app/services: complete
    - app/analysis: complete
    - app/backtesting: complete
- Next steps: continue repository abstraction, core layer extraction, and bootstrap wiring as per plan.

---

PROGRESS UPDATE (2026-03-19)

- Phase 5.2 completed:
    - Daily orchestration split into:
        - `app/application/use_cases/daily_pipeline_use_case.py`
        - `app/application/use_cases/execution_coordinator.py`
        - `app/application/use_cases/notification_coordinator.py`
    - `app/services/trading_pipeline.py` `run_daily()` now delegates to use-case orchestration layer.
    - Daily pipeline service file reduced below 300 lines.

- Phase 6.2 staged progress:
    - Root `main.py` preserves legacy wrapper API for test compatibility (`run_daily/run_weekly/run_monthly/parse_arguments`).
    - Root `run_daily()` non-legacy path now delegates to bootstrap use-case (`_APP_CONTAINER.daily_pipeline.run(...)`) instead of manual service wiring.
    - This preserves existing monkeypatch-based test contracts while moving runtime execution toward composition-root orchestration.
    - Legacy branch in root `main.py::run_daily()` has been removed; runtime now always delegates to container use-case.
    - Tests that previously monkeypatched legacy `run_daily` internals were migrated to assert container delegation behavior.
    - Root `main.py` wrapper functions (`run_weekly`, `run_monthly`, `run_walk_forward_manual`, `run_train_rl_manual`, `run_validation`, `run_paper_history`) now delegate to bootstrap-wired application use-cases instead of directly orchestrating repository/domain modules.
    - New application-layer use-cases introduced for root CLI decoupling: `RunWeeklyReliabilityUseCase`, `RunMonthlyRetrainingUseCase`, `RunPhase5ValidationUseCase`, `RunHistoricalPaperUseCase`.
    - `app/bootstrap/bootstrap.py` container wiring expanded with `weekly_reliability`, `monthly_retraining`, `phase5_validation`, and `historical_paper`, making root CLI a thin dispatch+compatibility layer.

- Phase 6.3 incremental progress:
    - `app/ui/app.py` now defaults to bootstrap container wiring (`build_application`) for settings and data-repository access, reducing direct legacy-style local bootstrap behavior.
    - Test monkeypatch surface (`DataManager` symbol) is intentionally retained for compatibility while runtime defaults are composition-root driven.
    - `app/interfaces/web/app.py` routes were refactored into a reusable API blueprint factory (`create_api_blueprint`).
    - `app/ui/app.py` now registers the same interfaces/web API blueprint with the already-built container, reducing route-layer drift between UI and interfaces entrypoints.
    - Admin route ownership is now centralized in `app/ui/admin_dashboard.py` blueprint (`/admin/dashboard`, `/admin/metrics`, `/admin/health`, `/admin/force-rebalance`); duplicate route implementations were removed from `app/ui/app.py`.
    - A minimal compatibility shim remains in `app/ui/app.py` (`DataManager` alias + `_create_data_repository()` indirection) so legacy monkeypatch-based tests continue to pass while runtime uses container-backed repositories by default.
    - Development-only endpoints were extracted from `app/ui/app.py` into dedicated blueprint factory `app/ui/dev_tools.py::create_dev_blueprint(...)`, and are now registered via injected providers (`settings_provider`, `ticker_provider`, `repository_factory`) to keep route behavior stable while reducing app-module size and coupling.
    - Production UI endpoints (`/`, `/chart`, `/history`, `/indicators`, `/report`, `/params`) were extracted into `app/ui/market_routes.py::create_market_blueprint(...)`; `app/ui/app.py` now acts primarily as composition-root wiring + blueprint registration + compatibility surface.
    - Production route handlers in `market_routes` intentionally resolve dependencies through `app.ui.app` at request time to preserve existing monkeypatch-based tests while reducing route implementation density in `app/ui/app.py`.
    - `app/ui/app.py` compatibility symbols used by tests (`load_data`, `sanitize_dataframe`, `compute_signals`, `get_params`, indicator/plotter helpers) now resolve via lazy shim wrappers, reducing eager top-level coupling while preserving monkeypatch contracts.
    - `app/ui/app.py` `Backtester` export was also converted to a lazy compatibility shim, so market-route runtime behavior and monkeypatch-based tests remain stable with less eager import coupling.
    - Flask app wiring (CORS, blueprint registration, request/response logging, error handlers) was moved to interfaces layer via `app/interfaces/web/ui_app.py::create_ui_app(...)`; `app/ui/app.py` now serves as a thin compatibility/provider surface plus app instantiation call.
    - Development runner (`run_dev.py`) now starts Flask through interfaces-layer default builder (`app/interfaces/web/ui_app.py::build_default_ui_app(...)`) instead of importing the legacy UI module as primary entrypoint.

- Pre-Phase-7 closure hardening (code-level):
    - `app/application/use_cases/run_phase5_validation.py` no longer imports legacy `DataManager`; table init is now handled through injected repository dependency only.
    - Key runtime paths were moved from legacy `app.decision.*` imports to `app.core.decision.*` in:
        - `app/services/trading_pipeline.py`
        - `app/application/use_cases/execution_coordinator.py`
        - `app/backtesting/historical_paper_runner.py`
        - `app/validation/risk_stress.py`
    - Compatibility-bound legacy import points intentionally retained (test monkeypatch contracts / compatibility behavior):
        - `app/infrastructure/repositories/data_manager_repository.py` (legacy adapter boundary)
        - `app/ui/admin_dashboard.py` selected dynamic imports
        - `app/reporting/audit_builder.py` drift detector dynamic import
        - `app/infrastructure/cron_tasks.py` decision engine dynamic import

- Phase 3.4 incremental progress:
    - `app/infrastructure/repositories/sqlite_model_repository.py` now owns direct SQLite `model_registry` SQL (no `DataManagerRepository` dependency).
    - `app/infrastructure/repositories/__init__.py` legacy `DataManager` export alias was removed; only explicit repository exports remain.
    - `app/infrastructure/cron_tasks.py` now uses `DataManagerRepository` only; legacy `DataManager` fallback import path removed.
    - `tests/test_cron_tasks.py` and `tests/test_coverage_boost_infra_more.py` updated to monkeypatch repository adapter targets instead of legacy module import path.
    - `app/infrastructure/repositories/data_manager_repository.py` now lazily instantiates legacy backend and delegates OHLCV operations through `SqliteOhlcvRepository` (with compatibility fallback for legacy/mock method signatures).
    - `app/infrastructure/repositories/data_manager_repository.py` now also delegates decision/history methods to `SqliteDecisionRepository` (`save_decision`, `fetch_decision`, `fetch_decisions_for_ticker`, `save_history_record`, `fetch_history_records_by_ticker`, `has_decision_for_date`, `fetch_history_range`, `fetch_recent_outcomes`) with legacy fallback retained.
    - `app/infrastructure/repositories/data_manager_repository.py` now delegates model operations (`save_model`, `fetch_model`, `fetch_models_for_ticker`) to `SqliteModelRepository` with legacy fallback retained.
    - `app/infrastructure/repositories/sqlite_model_repository.py` now also owns `model_registry` and `model_trust_metrics` SQL operations (`register_model`, `update_model_status`, `fetch_active_models`, `save_model_trust_metrics`, `fetch_latest_model_trust_weights`, `get_top_models`).
    - `app/infrastructure/repositories/data_manager_repository.py` now delegates these model-registry/trust operations explicitly to `SqliteModelRepository` (legacy fallback retained).
    - `app/infrastructure/repositories/sqlite_decision_repository.py` expanded to own additional SQL domains previously served from `DataManager` (`save_outcome`, `get_unevaluated_buy_decisions`, portfolio state persistence/readback, decision effectiveness persistence, decision quality persistence, confidence calibration persistence/readback, wf stability persistence, safety stress persistence, validation report persistence/readback, audit update).
    - `app/infrastructure/repositories/data_manager_repository.py` now explicitly delegates these runtime decision/outcome/portfolio/analytics operations to `SqliteDecisionRepository`, reducing implicit `__getattr__` passthrough reliance.
    - Remaining runtime helper operations were also moved to explicit repository paths (`save_market_data`, `get_market_data`, `get_ticker_historical_recommendations`, `get_strategy_accuracy`, `save_model_reliability`) and are now delegated explicitly from `DataManagerRepository`.
    - `app/infrastructure/repositories/data_manager_repository.py` now includes explicit delegates for `get_today_recommendations`, `save_walk_forward_result`, `fetch_latest_decision_quality_metrics`, `update_market_data`; generic wildcard passthrough (`__getattr__`) was removed.
    - 3.4 runtime scope is effectively completed in the current codebase: app-level runtime call paths are repository-backed, while the legacy `DataManager` import remains only inside the migration adapter as a compatibility fallback boundary.
    - Remaining direct legacy import point is intentionally limited to:
        - `app/infrastructure/repositories/data_manager_repository.py` (migration adapter)

- Validation status:
    - Targeted cron/infra tests green: `40 passed, 0 failed`.
    - Targeted adapter-sensitive tests green (`test_market_regime_detector`, `test_data_loader_edge_cases`, `test_cron_tasks`, `test_coverage_boost_infra_more`).
    - Targeted decision/history pipeline tests green (`test_daily_pipeline`, `test_pipeline_diagnostics`, `test_coverage_boost_core`, `test_coverage_boost_gap98`, `test_coverage_boost_more`).
    - Targeted model/pipeline compatibility tests green (`test_lstm_predictor`, `test_daily_pipeline`, `test_pipeline_diagnostics`, `test_cron_tasks`, `test_coverage_boost_more`).
    - Targeted decision/outcome/analytics compatibility tests green (`test_data_manager_edge_cases`, `test_coverage_boost_more`, `test_coverage_boost_core`, `test_daily_pipeline`, `test_pipeline_diagnostics`, `test_cron_tasks`, `test_market_regime_detector`, `test_data_loader_edge_cases`).
    - Targeted UI/data-loader compatibility tests green (`test_app_routes_edge_cases`, `test_admin_dashboard`, `test_data_loader_edge_cases`, `test_daily_pipeline`, `test_coverage_boost_more`, `test_coverage_boost_core`).
    - Targeted admin blueprint route consolidation tests green: `51 passed, 0 failed` (`test_admin_dashboard`, `test_admin_dashboard_integration`).
    - Follow-up UI/infra compatibility regression set green: `132 passed, 0 failed` (`test_app_routes_edge_cases`, `test_coverage_boost_more`, `test_coverage_boost_infra_more`).
    - Dev-route blueprint extraction validation green: `18 passed, 0 failed` (`test_app_routes_edge_cases`), plus full suite green after extraction.
    - Production-route blueprint extraction validation green (`test_app_routes_edge_cases`, `test_coverage_additional`, `test_admin_dashboard`, `test_admin_dashboard_integration`, `test_coverage_boost_more`, `test_coverage_boost_infra_more`, `test_coverage_boost_core`).
    - Root CLI 6.2 decoupling validation green (`test_roadmap_integrity`, `test_coverage_boost_more`, `test_coverage_boost_core`, `test_coverage_additional`) and follow-up route/pipeline smoke set green (`test_app_routes_edge_cases`, `test_admin_dashboard`, `test_admin_dashboard_integration`, `test_daily_pipeline`).
    - Full suite green after changes.

PHASE CLOSURE UPDATE (2026-03-21)

- Phase 6 closed:
    - 6.2 complete: root CLI wiring is bootstrap/use-case driven, with wrapper compatibility preserved.
    - 6.3 complete: Flask wiring ownership moved to interfaces layer (`app/interfaces/web/ui_app.py`), while `app/ui/app.py` remains a thin compatibility facade.
- Validation snapshot at closure:
    - Full suite green.
- Phase 7 kickoff scope is now defined in this document as post-migration hardening and controlled shim retirement planning.

- Phase 7.1 contract-freeze delivered:
    - Frozen compatibility contract documented in `docs/COMPATIBILITY_CONTRACT_FREEZE.md`.
    - Code-level guardrails added in `tests/test_phase7_contract_freeze.py`:
        - validates frozen symbols on `app/ui/app.py` facade
        - validates frozen wrappers on `main.py`
        - validates single legacy `DataManager` direct import boundary under `app/infrastructure/repositories/data_manager_repository.py`

- Phase 7.2 adapterization progress:
    - UI compatibility helper logic was extracted from `app/ui/app.py` into dedicated adapter module `app/interfaces/compat/ui_contract.py`.
    - `app/ui/app.py` now acts as an import-stable facade that re-exports frozen symbols from the adapter while retaining contract names for monkeypatch-based tests.
    - Targeted compatibility tests and full regression remained green after extraction.
    - Main wrapper dispatch logic was extracted into `app/interfaces/compat/main_contract.py`; `main.py` keeps the same public wrapper symbols and delegates through the adapter.

- Phase 7.3 closure gates delivered:
    - Added code-level closure gate tests in `tests/test_phase7_closure_gates.py` to enforce:
        - `main.py` remains free of direct repository/domain wiring imports
        - `app/ui/app.py` remains adapterized and interfaces-builder based
        - `run_dev.py` remains interfaces-first (no primary import from `app.ui.app`)

- Phase 7.4 deprecation path delivered:
    - Added `docs/PHASE7_DEPRECATION_PATH.md` with preconditions, ordered retirement steps, and rollback protocol for compatibility shim removal.

---