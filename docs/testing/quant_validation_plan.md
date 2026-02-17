Recommended: Run governance/validation via main.py:

	python main.py governance --mode diagnostics
	python main.py governance --mode validation
	python main.py governance --mode predeploy
	python main.py governance --mode full

# Quant Validation and Hardening Plan (Condensed)

## Purpose
This plan defines the minimum, production-grade validation and governance checks required to decide if the system is safe to run in paper mode and eligible for deployment.

## Mandatory Validation Tracks

### 1) Lookahead Bias Guard
- **Requirement:** All execution uses next-bar open (no same-bar close execution).
- **Evidence:** Validation outputs show execution_policy is next_open.

### 2) Data Leakage Guard
- **Requirement:** Time-based splits for RL training and evaluation.
- **Evidence:** Train/validation/test windows stored in metadata.

### 3) RL Stability
- **Requirement:** Stress tests on seeds and noise injection.
- **Evidence:** RL stress test summary with mean and std Sharpe.

### 4) Walk-Forward Robustness
- **Requirement:** Stable OOS performance across folds.
- **Evidence:** walk_forward summary with mean_oos_sharpe and discard ratio.

### 5) GA Seed Variance
- **Requirement:** GA robustness stress pass rate above threshold.
- **Evidence:** ga_robustness stress_pass_rate.

### 6) Execution Sensitivity
- **Requirement:** Execution policy sensitivity does not flip conclusions.
- **Evidence:** execution_sensitivity results in validation.json.

### 7) Risk Stress
- **Requirement:** Risk stress pass rate above threshold.
- **Evidence:** risk stress results in validation.json.

### 8) Diagnostics Integrity
- **Requirement:** Pipeline audit, data integrity checks, sanity strategy all pass.
- **Evidence:** diagnostics.json with warnings empty or explained.

### 9) Edge Diagnostics
- **Requirement:** If trade_count is zero, collapse_stage and collapse_reason are determined.
- **Evidence:** summary.json collapse_stage/collapse_reason fields.

## Required Outputs
All runs write to:
- `reports/<timestamp>/summary.json`
- `reports/<timestamp>/validation.json`
- `reports/<timestamp>/diagnostics.json`
- `reports/<timestamp>/tests.json`
- `reports/<timestamp>/checklist.json`

## Manual Review Items
Some steps remain manual and are captured in checklist output:
- Offline daily run validation
- Manual go-live review

Manual items are surfaced by the checklist runner from `docs/testing/go_live_checklist.md`.

## Governance Decision
Deployment is allowed only if all required checklist items pass and no manual blockers remain open.

## CLI Usage
```bash
python app/governance/quant_runner.py --mode diagnostics
python app/governance/quant_runner.py --mode validation
python app/governance/quant_runner.py --mode predeploy
python app/governance/quant_runner.py --mode full
```
