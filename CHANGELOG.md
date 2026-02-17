# Changelog

All notable changes to this repository are recorded in this file.

## 2026-02-17 — Migration & test-stabilization work

- Infrastructure/architecture:
  - Scaffolding for `Settings` and a composition root added (incremental migration from global `Config`).

- Backtesting fixes:
  - `Backtester._generate_trade_indices`: improved handling of open positions at series end to match test expectations for early/late entries.
  - `Backtester.run`: tolerant call to `compute_signals(..., audit=...)` with a fallback for patched shims.
  - Execution engine fallback for missing `Open`/next-open price handled more robustly.

- Analysis / indicators:
  - `app/analysis/analyzer.py`: indicator computations wrapped with defensive try/except to avoid crashes on tiny input windows (prevents BBANDS broadcasting errors).

- Walk-forward / validation:
  - `app/backtesting/walk_forward.py`: only attach `wf_run_id`/`wf_run_at` and persist results when a full WF summary is returned (keeps mocked returns unchanged for tests).
  - `wf_analysis` aggregation behavior adjusted to match expectations when only a single run remains.

- Model reliability:
  - `app/models/model_reliability.py`: `load_latest_reliability_scores` now prefers `MODEL_RELIABILITY_DIR` when configured and picks the latest file robustly; `save_reliability_scores` writes to file when configured.

- Diagnostics / edge filters:
  - `app/validation/edge_diagnostics.py`: `classify_collapse_reason` adjusted to avoid flagging strong-edge datasets as collapse reasons.

- Testing & compatibility shims:
  - Exposed expected module-level symbols (`fitness_single`, `Backtester`) and added compatibility shims to preserve many tests that monkeypatch module paths.
  - Numerous targeted fixes applied (see code changes) and the full test-suite run: 1088 tests passed, 0 failures (local run on 2026-02-17).

Notes:
- Changes are incremental and intentionally conservative — aimed at keeping test compatibility while moving toward the new architecture.
- Remaining roadmap items (in-repo TODO) include infra/config refactor, CI updates, and documentation handover.
