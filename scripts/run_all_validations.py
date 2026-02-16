"""Run quick, shadow, and full validation modes in sequence."""

from __future__ import annotations

import subprocess
import sys


def _run(mode: str) -> int:
    cmd = [sys.executable, "-m", "app.validation.validation_runner", "--mode", mode]
    print("=" * 80)
    print(f"RUN {mode} validation")
    print("=" * 80)
    return subprocess.call(cmd)


def main() -> int:
    exit_code = 0
    for mode in ("quick", "shadow", "full"):
        code = _run(mode)
        if code != 0 and exit_code == 0:
            exit_code = code
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
