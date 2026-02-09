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

**0.2 No internet mode**
- Disable internet
- Run daily pipeline

**Acceptance:** No crash, cache usage only, warning logged.

**0.3 No models available**
- Remove all models/*.zip
- Run historical paper runner

```bash
python main.py run-paper-history --ticker VOO --start-date 2022-01-01 --end-date 2022-01-10
```

**Acceptance:** HOLD decisions, no trades, decision_source=fallback.

### Section 1 - Daily Pipeline Integrity
**1.1 Single-ticker run**
```bash
python main.py daily --ticker VOO
```

**Acceptance:** decision_history row, position_sizing_json set, portfolio state updated, notifications sent (if enabled).

**1.2 Idempotency**
Run twice on the same day.

**Acceptance:** No duplicate decision for the date.

**1.3 Restart safety**
Interrupt a run and rerun.

**Acceptance:** No corrupted DB state or duplicated effects.

### Section 2 - Paper Execution and Outcomes
**2.1 Multi-day paper run**
```bash
python main.py run-paper-history --ticker VOO --start-date 2022-01-01 --end-date 2022-01-31
```

**Acceptance:** One decision per business day. Portfolio state evolves without NaN or infinite values.

**2.2 Outcomes linkage**
- Verify outcomes exist for closed positions.

**Acceptance:** outcomes table has entries linked to decision_history.

### Section 3 - Validation
```bash
python main.py validate --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
python scripts/phase6_check.py --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
```

**Acceptance:** Phase 5 report built; Phase 6 checks run without errors.

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

**0.2 Nincs internet**
- Kapcsold le az internetet
- Futtasd a napi pipeline-t

**Elfogadas:** nincs crash, cache hasznalat, warning log.

**0.3 Nincs modell**
- models/ mappaban nincs .zip
- Futtasd a historikus paper runnert

```bash
python main.py run-paper-history --ticker VOO --start-date 2022-01-01 --end-date 2022-01-10
```

**Elfogadas:** HOLD döntesek, nincs trade, decision_source=fallback.

### 1. Szekcio - Napi pipeline integritas
**1.1 Egy ticker futas**
```bash
python main.py daily --ticker VOO
```

**Elfogadas:** decision_history sor, position_sizing_json, portfolio state frissules.

**1.2 Idempotencia**
Futtasd ugyanazon a napon ketszer.

**Elfogadas:** nincs duplikalt dontes.

**1.3 Ujrainditas biztonsag**
Szakitsd meg, majd futtasd ujra.

**Elfogadas:** nincs DB korrupcio vagy dupla hatas.

### 2. Szekcio - Paper execution es outcome
**2.1 Tobbnapi paper run**
```bash
python main.py run-paper-history --ticker VOO --start-date 2022-01-01 --end-date 2022-01-31
```

**Elfogadas:** napi egy döntes, portfolio state rendben.

**2.2 Outcome ellenorzes**
- Ellenorizd, hogy vannak outcome-ok es megfelelo linkek.

### 3. Szekcio - Validacio
```bash
python main.py validate --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
python scripts/phase6_check.py --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
```

**Elfogadas:** Phase 5 riport keszul, Phase 6 ellenorzes lefut.

Acceptance Criteria

Position sizing decreases with confidence

No runaway exposure

Safety rules activate when applicable

TODO

 Add drawdown summary to logs

 Add loss-streak counter

🧠 SECTION 3 – Explainability & Trust
3.1 Decision Explainability Audit

Task
Ensure every decision is human-understandable.

Steps

Pick 5 random decisions

Inspect stored explanation JSON

Acceptance Criteria

Explanation answers:

Why this action?

Why this size?

What models agreed/disagreed?

No empty or placeholder text

TODO

 Add minimum explanation fields validation

 Add explainability linter

3.2 Email Usability Check

Task
Ensure daily emails are actionable.

Steps

Review 5 daily emails

Acceptance Criteria

Clear subject line

Action + size visible

Confidence and rationale included

Not overly verbose

TODO

 Add email length cap

 Add summary vs detail split

🧪 SECTION 4 – Validation & Monitoring
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

🚀 SECTION 5 – Go-Live Readiness Gate
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

GO / NO-GO Decision:
⬜ GO
⬜ NO-GO

Notes:

🧠 Usage Instruction for Copilot

Use this checklist as the authoritative source.
For each task:

generate helper scripts if needed

add missing guards/logs

do NOT introduce new features

do NOT change trading logic

prioritize safety and observability