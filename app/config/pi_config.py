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

from app.config.config import Config


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
    config_cls=Config,
    env=os.environ,
    platform_module=platform,
    model_path=Path("/proc/device-tree/model"),
    ensure_dirs=False,
):
    """
    Apply Pi-specific overrides to a Config class.

    Returns:
        dict: {"pi_mode": bool, "applied": bool, "base_dir": str, "overrides": dict}
    """
    pi_mode = detect_pi_mode(env=env, platform_module=platform_module, model_path=model_path)

    result = {
        "pi_mode": pi_mode,
        "applied": False,
        "base_dir": None,
        "overrides": {},
    }

    if not pi_mode:
        setattr(config_cls, "PI_MODE", False)
        return result

    base_dir = Path(env.get("PI_BASE_DIR", "/home/pi/tozsde_webapp"))
    overrides = _build_pi_paths(base_dir)

    # Apply paths
    for key, value in overrides.items():
        setattr(config_cls, key, value)

    # Update dependent file paths
    config_cls.DB_PATH = config_cls.DATA_DIR / "market_data.db"
    config_cls.PARAMS_FILE_PATH = config_cls.DATA_DIR / "optimized_params.json"
    config_cls.FAILED_DAYS_FILE_PATH = config_cls.DATA_DIR / "failed_to_download.json"
    config_cls.MODEL_TEST_RESULT_FILE_PATH = config_cls.DATA_DIR / "model_test_result.json"

    # Resource-conservative settings
    setattr(config_cls, "PI_MODE", True)
    setattr(config_cls, "MAX_BACKTEST_WINDOW", 252)
    config_cls.OPTIMIZER_POPULATION = min(getattr(config_cls, "OPTIMIZER_POPULATION", 50), 30)
    setattr(config_cls, "SQLITE_SINGLE_CONNECTION", True)
    setattr(config_cls, "DISABLE_HEAVY_TASKS", True)

    if ensure_dirs:
        for d in overrides.values():
            try:
                d.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

    result["applied"] = True
    result["base_dir"] = str(base_dir)
    result["overrides"] = {k: str(v) for k, v in overrides.items()}
    return result


__all__ = ["detect_pi_mode", "apply_pi_config"]