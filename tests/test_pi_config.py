"""
Tests for Raspberry Pi configuration helpers.

Author: AI Assistant
Date: 2026-02-01
"""

import os
from pathlib import Path

from app.config.pi_config import detect_pi_mode, apply_pi_config


class DummyPlatform:
    def __init__(self, system_value, machine_value):
        self._system = system_value
        self._machine = machine_value

    def system(self):
        return self._system

    def machine(self):
        return self._machine


class DummyConfig:
    DATA_DIR = Path("/tmp/data")
    LOG_DIR = Path("/tmp/logs")
    MODEL_DIR = Path("/tmp/models")
    CHART_DIR = Path("/tmp/charts")
    TENSORBOARD_DIR = Path("/tmp/tensorboard")
    MODEL_RELIABILITY_DIR = Path("/tmp/logs/model_reliability")
    DECISION_LOG_DIR = Path("/tmp/decision_logs")
    DECISION_OUTCOME_DIR = Path("/tmp/decision_outcomes")
    HISTORY_DIR = Path("/tmp/decision_history")
    DB_PATH = DATA_DIR / "market_data.db"
    PARAMS_FILE_PATH = DATA_DIR / "optimized_params.json"
    FAILED_DAYS_FILE_PATH = DATA_DIR / "failed_to_download.json"
    MODEL_TEST_RESULT_FILE_PATH = DATA_DIR / "model_test_result.json"
    OPTIMIZER_POPULATION = 50


def test_detect_pi_mode_env_true(monkeypatch):
    monkeypatch.setenv("PI_MODE", "true")
    assert detect_pi_mode() is True


def test_detect_pi_mode_env_false(monkeypatch):
    monkeypatch.setenv("PI_MODE", "false")
    assert detect_pi_mode() is False


def test_detect_pi_mode_arm_linux(monkeypatch):
    monkeypatch.delenv("PI_MODE", raising=False)
    platform_stub = DummyPlatform("Linux", "aarch64")
    assert detect_pi_mode(platform_module=platform_stub) is True


def test_apply_pi_config_no_pi(monkeypatch):
    monkeypatch.setenv("PI_MODE", "false")
    cfg = DummyConfig
    result = apply_pi_config(config_cls=cfg, ensure_dirs=False)
    assert result["applied"] is False
    assert getattr(cfg, "PI_MODE") is False


def test_apply_pi_config_pi_mode(monkeypatch):
    monkeypatch.setenv("PI_MODE", "true")
    monkeypatch.setenv("PI_BASE_DIR", "/home/pi/tozsde_webapp")
    cfg = DummyConfig

    result = apply_pi_config(config_cls=cfg, ensure_dirs=False)
    assert result["applied"] is True
    assert result["pi_mode"] is True
    assert cfg.DATA_DIR.as_posix().endswith("/home/pi/tozsde_webapp/app/data")
    assert cfg.DB_PATH.name == "market_data.db"
    assert cfg.OPTIMIZER_POPULATION == 30
    assert getattr(cfg, "MAX_BACKTEST_WINDOW") == 252
    assert getattr(cfg, "SQLITE_SINGLE_CONNECTION") is True
    assert getattr(cfg, "DISABLE_HEAVY_TASKS") is True