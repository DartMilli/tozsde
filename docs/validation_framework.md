Recommended: Run governance/validation via main.py:

	python main.py governance --mode diagnostics
	python main.py governance --mode validation
	python main.py governance --mode full

# Quant Validation and Governance Framework

## Purpose
This framework provides a unified, production-grade validation runner that centralizes diagnostics, validation metrics, governance checks, and reporting outputs under a single run directory.

## Runner
Entry point: `app/governance/quant_runner.py`

### Modes
- `research`: Runs walk-forward training and diagnostics, then lightweight validation.
- `validation`: Runs core validation metrics and scoring.
- `diagnostics`: Runs pipeline audit, data integrity checks, and sanity strategy.
- `predeploy`: Runs diagnostics plus full validation (shadow, risk, RL stability).
- `full`: Runs tests, diagnostics, and full validation in one pass.
- `tests`: Runs pytest and captures test summary and coverage.

### Exit codes
- `0`: Pass
- `1`: Warning (non-blocking issues)
- `2`: Blocked (deployment not allowed)

## Outputs
Each run writes to `reports/<timestamp>/`:
- `summary.json`: High-level status and key metrics.
- `validation.json`: Full validation results and scoring.
- `diagnostics.json`: Pipeline audit, data integrity, and sanity strategy results.
- `tests.json`: Test summary and coverage percent.
- `checklist.json`: Go-live checklist evaluation.
- `run.log`: Master log for the run (includes run_id and git commit).

## Summary Metrics
`summary.json` includes the following key fields:
- `mean_oos_sharpe`: Walk-forward out-of-sample Sharpe mean.
- `production_score`: Final production readiness score from validation scoring.
- `stress_pass_rate`: GA robustness stress pass rate.
- `fold_discard_ratio`: Walk-forward fold discard ratio (if available).
- `tests_passed` / `tests_failed`: Pytest totals from the run.
- `collapse_stage`: Where trade generation collapses when trade_count is zero.
- `collapse_reason`: Edge diagnostics classification for edge-related collapse.

## Collapse Stage
`collapse_stage` is computed when trade_count is zero and pipeline audit is available:
- `signal_generation`: No raw signals were produced.
- `feature_dropout`: Signals dropped after feature NaN filtering.
- `edge_filter`: Signals dropped after edge thresholding.
- `position_sizing`: Signals exist, but no position sizing attempts.
- `order_creation`: Positions attempted but no orders created.
- `execution_engine`: Orders created but no executions.
- `unknown`: No clear stage identified.

## Collapse Reason
When `collapse_stage` is `edge_filter`, `collapse_reason` is derived from edge diagnostics:
- `no_edge_present`: Signals never approach the edge threshold.
- `weak_edge_distribution`: Some edge exists but is weak or clustered below threshold.
- `threshold_too_strict`: Edge exists but threshold rejects most signals.

## Checklist Enforcement
`checklist.json` is produced from:
- Validation metrics (Sharpe, stability, stress pass rate, trade count)
- Diagnostics (data integrity warnings, collapse stage)
- Test results
- Documentation scan for manual steps and TODO items

Deployment is allowed only if all required checklist items pass and no blocking issues are present.

## CLI Usage
```bash
python app/governance/quant_runner.py --mode diagnostics
python app/governance/quant_runner.py --mode validation
python app/governance/quant_runner.py --mode full
```

## Notes
- All run artifacts are stored in the timestamped report directory.
- Diagnostics do not write files directly; the runner centralizes reporting.
- Manual review items are surfaced from `docs/testing/go_live_checklist.md` and `docs/testing/quant_validation_plan.md`.
