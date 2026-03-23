# Use Case Contracts

This document defines the unified application-layer contract for use-cases under `app/application/use_cases`.

## Unified Result Schema

All `run(...)` methods return the same envelope:

```json
{
  "status": "ok|error",
  "use_case": "string",
  "data": "object|null",
  "meta": "object",
  "error": { "message": "string" }
}
```

Notes:
- `error` is present only when `status = "error"`.
- `meta` contains execution context (for example: `ticker`, `mode`, `dry_run`).

## Implemented Use Cases

### 1) RunDailyPipelineUseCase
- Class: `RunDailyPipelineUseCase`
- Input: `run(dry_run: bool = False, ticker: str = None)`
- Output:
  - `status=ok`
  - `use_case=run_daily_pipeline`
  - `data={"completed": true, "processed": int, "executed": int, "no_trade": int}`
  - `meta={"dry_run": bool, "ticker": str|null}`

Implementation detail (Phase 5.2):
- Daily orchestration is delegated to:
  - `DailyPipelineUseCase`
  - `ExecutionCoordinator`
  - `NotificationCoordinator`

### 2) RunWalkForwardUseCase
- Class: `RunWalkForwardUseCase`
- Input: `run(ticker: str | None)`
- Output:
  - `status=ok`
  - `use_case=run_walk_forward`
  - `data={ ticker -> wf_result }` for single-ticker mode, or `{ ticker -> wf_result, ... }` for batch mode
  - `meta` includes `ticker` and `total_tickers` in batch mode

### 3) TrainRLModelUseCase
- Class: `TrainRLModelUseCase`
- Input: `run(ticker: str, **kwargs)`
- Output:
  - `status=ok`
  - `use_case=train_rl_model`
  - `data={"completed": true}`
  - `meta={"ticker": str}`

### 4) ValidateModelUseCase
- Class: `ValidateModelUseCase`
- Input: `run(mode: str = "quick")`
- Output:
  - `status=ok`
  - `use_case=validate_model`
  - `data=<ValidationRunner.results>`
  - `meta={"mode": "quick|full|shadow"}`

## Usage Examples

### Python (bootstrap container)

```python
from app.bootstrap.bootstrap import build_application

container = build_application()

wf = container.walk_forward.run(ticker="VOO")
if wf["status"] == "ok":
    print(wf["data"])
```

### CLI envelope output

`app/main.py` emits JSON with the same envelope for both success and error paths.

Examples:

```bash
python -m app.main walk-forward VOO
python -m app.main daily-pipeline
python -m app.main validate-model quick
```

Error example shape:

```json
{
  "status": "error",
  "use_case": "cli",
  "data": null,
  "error": { "message": "unknown command: xyz" },
  "meta": {}
}
```

## Error Codes (CLI)

CLI error responses include a machine-readable `code` in `meta`.

### Root CLI (`main.py`)
- `NO_COMMAND`: command was not provided.
- `SUBPROCESS_FAILED`: governance subprocess returned non-zero exit code.
- `INTERRUPTED`: execution was interrupted by user.
- `UNHANDLED_EXCEPTION`: unexpected exception reached the top-level handler.

### Modular CLI (`app/main.py`)
- `NO_COMMAND`: no command arguments were provided.
- `UNKNOWN_COMMAND`: command is not recognized.
- `MISSING_TICKER`: required ticker argument is missing for `train-rl`.

### Web API (`app/interfaces/web/app.py`)
- `MISSING_TICKER`: required ticker is missing for `/train-rl`.
- `REQUEST_FAILED`: generic runtime failure for `/walk-forward`, `/daily-pipeline`, `/train-rl`.
- `VALIDATION_FAILED`: validation execution failed for `/validate-model`.

Example (`meta` section):

```json
{
  "meta": {
    "code": "UNKNOWN_COMMAND",
    "command": "foo"
  }
}
```

### Web endpoint mapping

- `/walk-forward` -> `container.walk_forward.run(...)`
- `/daily-pipeline` -> `container.daily_pipeline.run(...)`
- `/train-rl` -> `container.train_rl.run(...)`
- `/validate-model` -> `container.validate_model.run(...)`

## Export Surface

Use-case exports are centralized at:
- `app/application/use_cases/__init__.py`

Exported symbols:
- `RunDailyPipelineUseCase`
- `RunWalkForwardUseCase`
- `TrainRLModelUseCase`
- `ValidateModelUseCase`
- `ok`
- `error`
