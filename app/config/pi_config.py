"""
Raspberry Pi specific configuration helpers.
This module detects Pi/ARM environments and applies conservative
resource settings suitable for low-power hardware. It is safe to
import on non-Pi machines; by default no changes are applied unless
PI_MODE env var is set to true or ARM Linux is detected.

Author: AI Assistant
Date: 2026-02-01
"""

import os
import platform
from pathlib import Path

from dataclasses import replace

from app.config.settings import Settings


def _parse_bool(value: str):
    if value is None:
        return None
    value = str(value).strip().lower()
    if value in ("1", "true", "yes", "y", "on"):
        return True
    if value in ("0", "false", "no", "n", "off"):
        return False
    return None


def detect_pi_mode(
    env=os.environ,
    platform_module=platform,
    model_path=Path("/proc/device-tree/model"),
):
    """
    Detect if running on Raspberry Pi.

    Priority:
    1) PI_MODE env var (true/false)
    2) Linux + ARM architecture
    3) /proc/device-tree/model contains 'Raspberry Pi'
    """
    env_override = _parse_bool(env.get("PI_MODE"))
    if env_override is not None:
        return env_override

    try:
        system = platform_module.system().lower()
        machine = platform_module.machine().lower()
    except Exception:
        system = ""
        machine = ""

    if system == "linux" and machine in ("armv7l", "aarch64", "arm64"):
        return True

    try:
        if model_path.exists():
            model = model_path.read_text().lower()
            if "raspberry pi" in model:
                return True
    except Exception:
        pass

    return False


def _build_pi_paths(base_dir: Path):
    return {
        "DATA_DIR": base_dir / "app" / "data",
        "LOG_DIR": base_dir / "logs",
        "MODEL_DIR": base_dir / "models",
        "CHART_DIR": base_dir / "charts",
        "TENSORBOARD_DIR": base_dir / "tensorboard",
        "MODEL_RELIABILITY_DIR": base_dir / "logs" / "model_reliability",
        "DECISION_LOG_DIR": base_dir / "decision_logs",
        "DECISION_OUTCOME_DIR": base_dir / "decision_outcomes",
        "HISTORY_DIR": base_dir / "decision_history",
    }


def apply_pi_config(
    settings: Settings,
    env=os.environ,
    platform_module=platform,
    model_path=Path("/proc/device-tree/model"),
):
    """Return Settings with Pi-specific overrides applied."""
    pi_mode = detect_pi_mode(
        env=env, platform_module=platform_module, model_path=model_path
    )
    if not pi_mode:
        return settings

    base_dir = Path(env.get("PI_BASE_DIR", "/home/pi/tozsde_webapp"))
    overrides = _build_pi_paths(base_dir)

    return replace(
        settings,
        PI_MODE=True,
        DATA_DIR=overrides["DATA_DIR"],
        LOG_DIR=overrides["LOG_DIR"],
        MODEL_DIR=overrides["MODEL_DIR"],
        CHART_DIR=overrides["CHART_DIR"],
        TENSORBOARD_DIR=overrides["TENSORBOARD_DIR"],
        MODEL_RELIABILITY_DIR=overrides["MODEL_RELIABILITY_DIR"],
        DECISION_LOG_DIR=overrides["DECISION_LOG_DIR"],
        DECISION_OUTCOME_DIR=overrides["DECISION_OUTCOME_DIR"],
        HISTORY_DIR=overrides["HISTORY_DIR"],
        DB_PATH=overrides["DATA_DIR"] / "market_data.db",
        PARAMS_FILE_PATH=overrides["DATA_DIR"] / "optimized_params.json",
        FAILED_DAYS_FILE_PATH=overrides["DATA_DIR"] / "failed_to_download.json",
    )


__all__ = ["detect_pi_mode", "apply_pi_config"]
