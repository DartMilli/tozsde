# Roadmap Integrity Test Specification

This document translates each roadmap phase (P0–P9) into **conceptual integrity test assertions**.
These are *not unit tests* and *not implementation-level checks*.
They are **system-level, semantic guarantees** that must hold true if the roadmap is correctly implemented.

These assertions are intended to be used by:
- Humans during architectural/code reviews
- AI tools (e.g. GitHub Copilot) to generate concrete integrity / scenario tests
- Future maintainers to detect silent conceptual regressions

---

## P0 — Critical Stability & Determinism

**Integrity Assertions**
- The system must produce identical outputs when run twice with the same inputs and same historical data.
- No recommendation may be generated without a persisted state entry.
- A failure in one ticker must not abort processing of other tickers.
- All filesystem paths used during execution must be absolute and environment-independent.
- Secrets must never be hardcoded; absence of required secrets must fail fast.

**Failure Signals**
- Silent cron failures
- Duplicate recommendations for the same `(ticker, date)`
- Relative-path dependent crashes

---

## P1 — Unified Data Model & Persistence

**Integrity Assertions**
- There must be exactly one authoritative OHLCV data source.
- All reads and writes of market data must go through a single data-access layer.
- Database writes must be idempotent for identical `(ticker, date)` inputs.
- No runtime logic may depend on JSON files if a DB table exists for the same purpose.

**Failure Signals**
- Diverging results between backtest and live runs
- Inconsistent indicator values for identical data windows

---

## P2 — Realistic Simulation & Performance

**Integrity Assertions**
- Every simulated trade must include transaction costs.
- Backtest results must degrade measurably when costs are increased.
- Fitness scores must penalize overtrading and volatility spikes.
- Cached results must never change semantic outcomes.

**Failure Signals**
- Unrealistically smooth equity curves
- Identical results with and without transaction costs

---

## P3 — Signal Quality & Confidence

**Integrity Assertions**
- Every recommendation must include a confidence score.
- Confidence must be monotonic with signal strength.
- Low-confidence signals must be suppressible without breaking the pipeline.
- A "no-trade" outcome must be an explicit, explainable decision.

**Failure Signals**
- Trades executed with near-zero confidence
- Confidence values that do not affect outcomes

---

## P4 — Safety & Risk Constraints

**Integrity Assertions**
- The system must be able to refuse trades under adverse conditions.
- Cooldown and drawdown rules must override model output.
- Risk limits must cap exposure deterministically.

**Failure Signals**
- Trades during defined cooldown periods
- Capital allocation exceeding configured risk

---

## P5 — Walk-Forward Validity

**Integrity Assertions**
- Training data must never overlap with test windows.
- Walk-forward scores must influence downstream decisions.
- Removing walk-forward validation must measurably reduce robustness.

**Failure Signals**
- Identical performance with and without WF
- Future data leakage symptoms

---

## P6 — Model Reliability Tracking

**Integrity Assertions**
- Model performance must be evaluated over rolling historical windows.
- Reliability degradation must be detectable without retraining.
- Reliability scores must be persisted and comparable over time.

**Failure Signals**
- Models that never decay in reliability
- Missing historical reliability data

---

## P7 — Capital Allocation Logic

**Integrity Assertions**
- Capital allocation must respect confidence, risk, and correlation.
- Increasing available capital must not change trade selection, only sizing.
- Zero-allocation decisions must be explainable.

**Failure Signals**
- Capital magically appearing or disappearing
- Allocation without explicit decision linkage

---

## P8 — Explainability & Auditability

**Integrity Assertions**
- Every decision must be explainable post hoc.
- Removing explanation logic must reduce observability, not alter decisions.
- Audit records must be immutable once written.

**Failure Signals**
- Unexplainable trades
- Missing audit trails

---

## P9 — Operational Robustness

**Integrity Assertions**
- The system must tolerate restarts without state corruption.
- Partial failures must degrade functionality, not correctness.
- Monitoring/logs must reflect real system health.

**Failure Signals**
- State loss after restart
- Silent partial pipeline execution

---

## Usage Notes

These assertions should be validated via:
- Scenario tests
- End-to-end dry runs
- Fault injection
- Historical replay

They intentionally avoid:
- Function-level checks
- Mock-heavy testing
- Implementation assumptions

If an assertion cannot be validated, the roadmap item should be considered **not complete**.
