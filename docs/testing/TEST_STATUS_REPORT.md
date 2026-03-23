# Test & Code Quality Status Report

EN: This file is the canonical place for test/coverage execution workflow and latest verification notes.
HU: Ez a fajl a teszt/coverage futtatas kanonikus forrasa es a legutobbi ellenorzesi megjegyzesek helye.

## Current Usage

- Run local test suite:

```bash
pytest
```

- Run full scripted suite:

```bash
python scripts/run_all_tests.py
```

- Run coverage report:

```bash
pytest --cov=app --cov-report=term-missing
```

## Validation Commands

```bash
python main.py run-paper-history --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
python main.py validate --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
```

## Notes

- Do not treat older sprint-era numbers in historical documents as current truth.
- If test/coverage numbers are needed in release notes, regenerate them from fresh local run results.
- For go-live workflow checks, use `docs/testing/go_live_checklist.md`.

### Latest Verification (2026-03-22)

- Command: `pytest`
- Result: PASS (`1094 passed`, `0 failed`)
- Coverage run: `runTests(mode="coverage")` completed successfully in this workspace session

---

**Last Updated:** 2026-03-22

## Current Verification Policy

- Keep this document free of fixed test-count snapshots.
- When counts are needed, generate them from a fresh local run and update only this file.
- Historical sprint metrics remain in `docs/SPRINTS.md` as historical context only.

## Recommended Verification Sequence

1) Fast local check

```bash
pytest
```

2) Full scripted regression

```bash
python scripts/run_all_tests.py
```

3) Coverage snapshot

```bash
pytest --cov=app --cov-report=term-missing
```

4) Optional integrated report + validation

```bash
python scripts/run_tests_with_report.py --with-validation --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
```

## Related Documents

- Go-live validation checklist: `docs/testing/go_live_checklist.md`
- Operational docs index: `docs/INDEX.md`
- Troubleshooting for test/ops failures: `docs/TROUBLESHOOTING_GUIDE.md`
