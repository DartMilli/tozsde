import types
from unittest.mock import MagicMock, patch

import pytest

import main as cli_main


class _Settings:
    def __init__(self, enable_flask: bool, admin_api_key: str = "secure_key"):
        self.ENABLE_FLASK = enable_flask
        self.ADMIN_API_KEY = admin_api_key


def test_run_serve_returns_error_when_flask_disabled(monkeypatch):
    container = types.SimpleNamespace(settings=_Settings(enable_flask=False))
    monkeypatch.setattr(cli_main, "_get_container", lambda: container)

    result = cli_main.run_serve()

    assert result["ok"] is False
    assert "ENABLE_FLASK=true" in result["error"]


def test_run_serve_starts_flask_when_enabled(monkeypatch):
    container = types.SimpleNamespace(settings=_Settings(enable_flask=True))
    monkeypatch.setattr(cli_main, "_get_container", lambda: container)

    fake_app = MagicMock()
    with patch("app.interfaces.web.ui_app.build_default_ui_app", return_value=fake_app):
        result = cli_main.run_serve(host="0.0.0.0", port=5050, debug=True)

    fake_app.run.assert_called_once_with(host="0.0.0.0", port=5050, debug=True)
    assert result["ok"] is True
    assert result["port"] == 5050


def test_run_serve_warns_on_insecure_admin_key(monkeypatch):
    container = types.SimpleNamespace(
        settings=_Settings(enable_flask=True, admin_api_key="admin_key_12345")
    )
    monkeypatch.setattr(cli_main, "_get_container", lambda: container)

    fake_app = MagicMock()
    with patch(
        "app.interfaces.web.ui_app.build_default_ui_app", return_value=fake_app
    ), patch.object(cli_main.logger, "warning") as warn_mock:
        cli_main.run_serve()

    assert warn_mock.called


def test_cli_serve_emits_error_and_exits_when_disabled(monkeypatch):
    payloads = []

    monkeypatch.setattr(
        cli_main,
        "run_serve",
        lambda host, port, debug: {"ok": False, "error": "disabled"},
    )
    monkeypatch.setattr(cli_main, "_emit", lambda payload: payloads.append(payload))

    with patch("sys.argv", ["main.py", "serve"]):
        with pytest.raises(SystemExit) as exc:
            cli_main.main()

    assert exc.value.code == 1
    assert payloads
    assert payloads[-1]["status"] == "error"
    assert payloads[-1]["use_case"] == "cli.serve"


def test_cli_serve_emits_ok_when_started(monkeypatch):
    payloads = []

    monkeypatch.setattr(
        cli_main,
        "run_serve",
        lambda host, port, debug: {
            "ok": True,
            "host": host,
            "port": port,
            "debug": debug,
        },
    )
    monkeypatch.setattr(cli_main, "_emit", lambda payload: payloads.append(payload))

    with patch(
        "sys.argv",
        ["main.py", "serve", "--host", "0.0.0.0", "--port", "5051", "--debug"],
    ):
        cli_main.main()

    assert payloads
    assert payloads[-1]["status"] == "ok"
    assert payloads[-1]["use_case"] == "cli.serve"
    assert payloads[-1]["data"]["port"] == 5051
