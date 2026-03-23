# Go Live Checklist (EN + HU)

## English

### Goal
Ensure the trading system is safe, deterministic, observable, and usable before Raspberry Pi deployment.

### Scope
- Paper trading only
- Daily batch execution
- No live broker integration
- Focus on robustness, not new features

### Section 0 - Environment and Bootstrap
**0.1 Cold start**
- Delete database and cached data
- Run validation and daily pipeline

```bash
python main.py validate --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
python main.py daily --ticker VOO
```

**Acceptance:** DB created, schema initialized, no uncaught exceptions.

Status (2026-02-09): Completed with DB backup, validation, and daily dry-run.

**0.2 No internet mode**
- Disable internet
- Run daily pipeline

Manual step: must be executed offline by operator.

**Acceptance:** No crash, cache usage only, warning logged.

Status (2026-02-10): Completed offline daily at 09:18; cache-only behavior confirmed (no data update/download messages).

Manual step (offline required).

**0.3 No models available**
- Remove all models/*.zip
- Run historical paper runner

```bash
python main.py run-paper-history --ticker VOO --start-date 2022-01-01 --end-date 2022-01-10
```

**Acceptance:** HOLD decisions, no trades, decision_source=fallback.

Note: ALLOW_NO_MODEL_FALLBACK=true enables HOLD fallback when models are missing.

Status (2026-02-09): Completed (fallback HOLD decisions persisted).

### Section 1 - Daily Pipeline Integrity
**1.1 Single-ticker run**
```bash
python main.py daily --ticker VOO
```

**Acceptance:** decision_history row, position_sizing_json set, portfolio state updated, notifications sent (if enabled).

Status (2026-02-11): Completed (daily run executed for VOO).

**1.2 Idempotency**
Run twice on the same day.

**Acceptance:** No duplicate decision for the date.

Status (2026-02-09): Completed (idempotency guard confirmed).

**1.3 Restart safety**
Interrupt a run and rerun.

**Acceptance:** No corrupted DB state or duplicated effects.

Status (2026-02-10): Completed (interrupted run + re-run, no duplicates).

### Section 2 - Paper Execution and Outcomes
**2.1 Multi-day paper run**
```bash
python main.py run-paper-history --ticker VOO --start-date 2022-01-01 --end-date 2022-01-31
```

**Acceptance:** One decision per business day. Portfolio state evolves without NaN or infinite values.

Status (2026-02-10): Completed (21/21 business days have decisions).

**2.2 Outcomes linkage**
- Verify outcomes exist for closed positions.

**Acceptance:** outcomes table has entries linked to decision_history.

Status (2026-02-10): No outcomes in range (fallback HOLD only); recheck when trades exist.

Status (2026-02-10): Outcomes linkage check ran (trade_decisions=0, missing_outcomes=0).

Status (2026-02-11): Outcomes linkage check ran (trade_decisions=0, missing_outcomes=0).

### Section 3 - Validation
```bash
python main.py validate --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
python scripts/phase6_check.py --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
```

**Acceptance:** Phase 5 report built; Phase 6 checks run without errors.

Status (2026-02-10): Completed. Phase 5 ok. Phase 6 ok; no_data for effectiveness/model_trust (no outcomes yet).

## Magyar

### Cel
Biztonsag, determinizmus, megfigyelhetoseg es hasznalhatosag ellenorzese RPi telepites elott.

### Scope
- Csak paper trading
- Napi batch futas
- Nincs live broker integracio
- Fokusz a stabilitason

### 0. Szekcio - Kornyezet es bootstrap
**0.1 Cold start**
- DB torles
- Validacio es napi pipeline futtatas

```bash
python main.py validate --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
python main.py daily --ticker VOO
```

**Elfogadas:** DB letrejott, schema init, nincs hiba.

Statusz (2026-02-09): Keszen (DB backup, validacio, napi dry-run).

**0.2 Nincs internet**
- Kapcsold le az internetet
- Futtasd a napi pipeline-t

Manualis lepes: offline modban, operator futtassa.

**Elfogadas:** nincs crash, cache hasznalat, warning log.

Statusz (2026-02-10): Offline daily 09:18-kor kesz; cache-only viselkedes igazolt (nincs data update/download uzenet).

Manualis lepes (offline szukseges).

**0.3 Nincs modell**
- models/ mappaban nincs .zip
- Futtasd a historikus paper runnert

```bash
python main.py run-paper-history --ticker VOO --start-date 2022-01-01 --end-date 2022-01-10
```

**Elfogadas:** HOLD dontesek, nincs trade, decision_source=fallback.

Statusz (2026-02-09): Keszen (fallback HOLD dontesek mentve).

### 1. Szekcio - Napi pipeline integritas
**1.1 Egy ticker futas**
```bash
python main.py daily --ticker VOO
```

**Elfogadas:** decision_history sor, position_sizing_json, portfolio state frissules.

Statusz (2026-02-11): Keszen (VOO napi futas lefutott).

**1.2 Idempotencia**
Futtasd ugyanazon a napon ketszer.

**Elfogadas:** nincs duplikalt dontes.

Statusz (2026-02-09): Keszen (idempotencia ellenorzes ok).

**1.3 Ujrainditas biztonsag**
Szakitsd meg, majd futtasd ujra.

**Elfogadas:** nincs DB korrupcio vagy dupla hatas.

Statusz (2026-02-10): Keszen (megszakitas + ujrafutas, nincs duplikacio).

### 2. Szekcio - Paper execution es outcome
**2.1 Tobbnapi paper run**
```bash
python main.py run-paper-history --ticker VOO --start-date 2022-01-01 --end-date 2022-01-31
```

**Elfogadas:** napi egy dontes, portfolio state rendben.

Statusz (2026-02-10): Keszen (21/21 munkanapra van dontes).

**2.2 Outcome ellenorzes**
- Ellenorizd, hogy vannak outcome-ok es megfelelo linkek.

Statusz (2026-02-10): Nincs outcome a tartomanyban (csak fallback HOLD); trade eseten ellenorizd ujra.

Statusz (2026-02-10): Outcomes linkage ellenorzes lefutott (trade_decisions=0, missing_outcomes=0).

Statusz (2026-02-11): Outcomes linkage ellenorzes lefutott (trade_decisions=0, missing_outcomes=0).

### 3. Szekcio - Validacio
```bash
python main.py validate --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
python scripts/phase6_check.py --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
```

**Elfogadas:** Phase 5 riport keszul, Phase 6 ellenorzes lefut.

Statusz (2026-02-10): Keszen. Phase 5 ok. Phase 6 ok; no_data az effectiveness/model_trust reszben (nincs outcome).

Acceptance Criteria

Position sizing decreases with confidence

No runaway exposure

Safety rules activate when applicable

TODO

 Add drawdown summary to logs

 Add loss-streak counter

Status (2026-02-10): Implemented (GO_LIVE_METRICS logs).

Status (2026-02-10): Verified GO_LIVE_METRICS appears in daily logs.

 SECTION 3 - Explainability & Trust
3.1 Decision Explainability Audit

Task
Ensure every decision is human-understandable.

Steps

Pick 5 random decisions

Inspect stored explanation JSON

Run daily pipeline and check logs for EXPLAINABILITY_LINT warnings

Acceptance Criteria

Explanation answers:

Why this action?

Why this size?

What models agreed/disagreed?

No empty or placeholder text

TODO

 Explainability linter is enabled; resolve any EXPLAINABILITY_LINT warnings

Status (2026-02-10): Manual review report generated.
Status (2026-02-10): No EXPLAINABILITY_LINT warnings found in logs.

3.2 Email Usability Check

Task
Ensure daily emails are actionable.

Steps

Review 5 daily emails

Verify Summary and Details sections are present

Confirm emails are capped (EMAIL_MAX_BODY_CHARS) and truncated cleanly if needed

Acceptance Criteria

Clear subject line

Action + size visible

Confidence and rationale included

Not overly verbose

TODO

 Email length cap and summary/detail split are enabled

Status (2026-02-10): Manual review report generated.

 SECTION 4 - Validation & Monitoring
4.1 Phase 5 Validation

Task
Run full Phase 5 validation.

python main.py validate --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31


Acceptance Criteria

Validation completes

no_data statuses are explained

Report generated successfully

TODO

 Add validation summary banner

 Add run timestamp + git hash

Status (2026-02-10): Implemented (banner + timestamp + git hash).

4.2 Phase 6 Determinism & Safety

Task
Run Phase 6 checks.

python phase6_check.py --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31


Acceptance Criteria

Determinism checks pass

Position sizing monotonic

Reward shaping logged

Promotion gate enforced

TODO

 Fail build if determinism breaks

 Add CI hook (future)

Status (2026-02-10): Determinism gate implemented; CI hook added (GitHub Actions workflow).

 SECTION 5 - Go-Live Readiness Gate
Final Checklist (ALL must be true)

 Cold start passes

 No-internet safe

 No-models safe

 Daily pipeline idempotent

 Paper trading lifecycle complete

 Decisions explainable

 Emails understandable

 Validation reports stable

 No silent failures

 Logs are actionable

Status (2026-02-10)

 Cold start passes: OK

 No-internet safe: OK (offline run completed; cache-only behavior confirmed)

 No-models safe: OK (fallback HOLD)

 Daily pipeline idempotent: OK

 Paper trading lifecycle complete: OK (decisions present; outcomes pending)

 Decisions explainable: OK (manual review report generated)

 Emails understandable: OK (manual review report generated)

 Validation reports stable: OK (Phase 5/6 complete; determinism gate ok)

 No silent failures: OK (errors logged)

 Logs are actionable: OK (GO_LIVE_METRICS verified)

GO / NO-GO Decision:
 GO
 NO-GO

Recommendation (2026-02-10): GO for paper trading; recheck outcomes linkage once trades exist.

Magyar osszegzes (2026-02-10): GO paper trading-re; outcome ellenorzes ujra, ha lesznek trade-ek.

Notes:

 Usage Instruction for Copilot

Use this checklist as the authoritative source.
For each task:

generate helper scripts if needed

add missing guards/logs

do NOT introduce new features

do NOT change trading logic

prioritize safety and observability