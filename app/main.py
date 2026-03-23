# Main entrypoint for CLI

import json
import sys

from app.application.use_cases.result import error as result_error
from app.bootstrap.bootstrap import build_application


def _emit(payload):
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    app = build_application()

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "walk-forward":
            ticker = sys.argv[2] if len(sys.argv) > 2 else None
            result = app.walk_forward.run(ticker=ticker)
            _emit(result)
        elif command == "daily-pipeline":
            ticker = sys.argv[2] if len(sys.argv) > 2 else None
            result = app.daily_pipeline.run(ticker=ticker)
            _emit(result)
        elif command == "train-rl":
            ticker = sys.argv[2] if len(sys.argv) > 2 else None
            if not ticker:
                _emit(
                    result_error(
                        "train_rl_model",
                        "ticker is required",
                        code="MISSING_TICKER",
                    )
                )
                sys.exit(2)
            result = app.train_rl.run(ticker=ticker)
            _emit(result)
        elif command == "validate-model":
            mode = sys.argv[2] if len(sys.argv) > 2 else "quick"
            result = app.validate_model.run(mode=mode)
            _emit(result)
        else:
            _emit(
                result_error(
                    "cli",
                    f"unknown command: {command}",
                    code="UNKNOWN_COMMAND",
                    command=command,
                )
            )
            sys.exit(2)
    else:
        _emit(
            result_error(
                "cli",
                "usage: main.py [command] [args]",
                code="NO_COMMAND",
            )
        )
        sys.exit(2)
