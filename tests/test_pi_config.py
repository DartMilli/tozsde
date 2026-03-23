"""
Tests for Raspberry Pi configuration helpers.

Author: AI Assistant
Date: 2026-02-01
"""

import os
from dataclasses import replace
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


def test_detect_pi_mode_model_file(monkeypatch, tmp_path):
    monkeypatch.delenv("PI_MODE", raising=False)
    model_path = tmp_path / "model"
    model_path.write_text("Raspberry Pi 4 Model B")

    platform_stub = DummyPlatform("Linux", "x86_64")
    assert detect_pi_mode(platform_module=platform_stub, model_path=model_path) is True


def test_detect_pi_mode_invalid_env(monkeypatch):
    monkeypatch.setenv("PI_MODE", "maybe")
    platform_stub = DummyPlatform("Linux", "x86_64")
    assert detect_pi_mode(platform_module=platform_stub) is False


def test_apply_pi_config_no_pi(monkeypatch, test_settings):
    monkeypatch.setenv("PI_MODE", "false")
    settings = replace(test_settings, PI_MODE=False)
    result = apply_pi_config(settings=settings, env=os.environ)
    assert result is settings
    assert result.PI_MODE is False


def test_apply_pi_config_pi_mode(monkeypatch, test_settings):
    monkeypatch.setenv("PI_MODE", "true")
    monkeypatch.setenv("PI_BASE_DIR", "/home/pi/tozsde_webapp")
    settings = replace(test_settings, PI_MODE=False)

    result = apply_pi_config(settings=settings, env=os.environ)
    assert result.PI_MODE is True
    assert result.DATA_DIR.as_posix().endswith("/home/pi/tozsde_webapp/app/data")
    assert result.DB_PATH.name == "market_data.db"


def test_apply_pi_config_creates_dirs(monkeypatch, tmp_path, test_settings):
    monkeypatch.setenv("PI_MODE", "true")
    monkeypatch.setenv("PI_BASE_DIR", str(tmp_path))
    settings = replace(test_settings, PI_MODE=False)

    result = apply_pi_config(settings=settings, env=os.environ)
    assert result.PI_MODE is True
    assert str(result.DATA_DIR).startswith(str(tmp_path))
    assert str(result.LOG_DIR).startswith(str(tmp_path))
