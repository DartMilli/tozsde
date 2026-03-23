"""Edge case tests for Flask app routes in app.ui.app."""

from io import BytesIO
from datetime import datetime
from dataclasses import replace

import pandas as pd
import pytest

import app.ui.app as ui_app
from app.ui.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _make_df():
    dates = pd.date_range(start="2025-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "Open": [100, 101, 102, 103, 104],
            "High": [105, 106, 107, 108, 109],
            "Low": [99, 100, 101, 102, 103],
            "Close": [102, 103, 104, 105, 106],
            "Volume": [1000] * 5,
        },
        index=dates,
    )


def test_dev_status_denied_in_production(client):
    app.config["ENV"] = "production"
    res = client.get("/dev/status")
    assert res.status_code == 403


def test_dev_status_ok_in_development(monkeypatch, client):
    app.config["ENV"] = "development"

    class DummyDM:
        pass

    monkeypatch.setattr("app.ui.app.DataManager", lambda: DummyDM())

    res = client.get("/dev/status")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "OK"
    assert data["environment"] == "development"


def test_dev_config_ok_in_development(client):
    app.config["ENV"] = "development"
    res = client.get("/dev/config")
    assert res.status_code == 200
    data = res.get_json()
    assert "TICKERS" in data


def test_dev_db_init_ok(monkeypatch, client):
    app.config["ENV"] = "development"

    class DummyDM:
        def initialize_tables(self):
            return None

    monkeypatch.setattr("app.ui.app.DataManager", lambda: DummyDM())

    res = client.post("/dev/db-init")
    assert res.status_code == 200
    assert res.get_json()["status"] == "OK"


def test_dev_clear_recs_ok(client):
    app.config["ENV"] = "development"
    res = client.post("/dev/clear-recs")
    assert res.status_code == 200
    assert res.get_json()["status"] == "OK"


def test_dev_metrics_ok(monkeypatch, client):
    app.config["ENV"] = "development"

    class DummyProc:
        def cpu_percent(self, interval=1):
            return 10.0

        def memory_info(self):
            class Mem:
                rss = 1024 * 1024

            return Mem()

    class DummyPsutil:
        def Process(self, pid):
            return DummyProc()

    monkeypatch.setitem(__import__("sys").modules, "psutil", DummyPsutil())

    res = client.get("/dev/metrics")
    assert res.status_code == 200
    data = res.get_json()
    assert "system" in data


def test_index_renders_dashboard(monkeypatch, client):
    class DummyDM:
        def get_today_recommendations(self):
            return [{"ticker": "VOO"}]

    monkeypatch.setattr("app.ui.app.DataManager", lambda: DummyDM())
    monkeypatch.setattr("app.ui.app.get_supported_tickers", lambda: {"VOO": {}})
    monkeypatch.setattr("app.ui.app.render_template", lambda *a, **k: "OK")

    res = client.get("/")
    assert res.status_code == 200


def test_chart_returns_404_on_empty_data(monkeypatch, client):
    def _fake_load(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr("app.ui.app.load_data", _fake_load)

    res = client.get("/chart?ticker=SPY&start=2025-01-01&end=2025-01-02")
    assert res.status_code == 404


def test_chart_success_returns_png(monkeypatch, client):
    df = _make_df()

    monkeypatch.setattr("app.ui.app.load_data", lambda *a, **k: df)
    monkeypatch.setattr("app.ui.app.sanitize_dataframe", lambda x: x)
    monkeypatch.setattr("app.ui.app.compute_signals", lambda *a, **k: ([], {}))
    monkeypatch.setattr(
        "app.ui.app.get_candle_img_buffer", lambda *a, **k: BytesIO(b"png")
    )

    res = client.get("/chart?ticker=SPY&start=2025-01-01&end=2025-01-05")
    assert res.status_code == 200
    assert res.mimetype == "image/png"


def test_history_requires_ticker(client):
    res = client.get("/history")
    assert res.status_code == 400


def test_history_returns_records(monkeypatch, client):
    class DummyDM:
        def get_ticker_historical_recommendations(self, ticker, start, end):
            return [{"ticker": ticker}]

    monkeypatch.setattr("app.ui.app.DataManager", lambda: DummyDM())

    res = client.get("/history?ticker=SPY&start=2025-01-01&end=2025-01-02")
    assert res.status_code == 200
    data = res.get_json()
    assert data["history"][0]["ticker"] == "SPY"


def test_indicators_returns_payload(client):
    res = client.get("/indicators")
    assert res.status_code == 200


def test_report_returns_404_on_empty_data(monkeypatch, client):
    monkeypatch.setattr("app.ui.app.load_data", lambda *a, **k: pd.DataFrame())

    res = client.get("/report?ticker=SPY&start=2025-01-01&end=2025-01-02")
    assert res.status_code == 404


def test_report_success(monkeypatch, client):
    df = _make_df()

    class DummyReport:
        diagnostics = {"equity_curve": {"date": [], "portfolio_value": []}}

    class DummyBacktester:
        def __init__(self, df, ticker):
            pass

        def run(self, params):
            return DummyReport()

    monkeypatch.setattr("app.ui.app.load_data", lambda *a, **k: df)
    monkeypatch.setattr("app.ui.app.sanitize_dataframe", lambda x: x)
    monkeypatch.setattr("app.ui.app.get_params", lambda t: {})
    monkeypatch.setattr("app.ui.app.Backtester", DummyBacktester)
    monkeypatch.setattr("app.ui.app.get_equity_curve_buffer", lambda *a, **k: "img")
    monkeypatch.setattr("app.ui.app.get_drawdown_curve_buffer", lambda *a, **k: "img")
    monkeypatch.setattr("app.ui.app.render_template", lambda *a, **k: "OK")

    res = client.get("/report?ticker=SPY&start=2025-01-01&end=2025-01-05")
    assert res.status_code == 200


def test_admin_health_unauthorized(client):
    res = client.get("/admin/health")
    assert res.status_code == 401


def test_admin_health_authorized(monkeypatch, client, test_settings):
    from app.ui import set_settings as set_ui_settings

    class DummyMetrics:
        def get_health_status(self):
            return {"status": "healthy"}

    new_settings = replace(test_settings, ADMIN_API_KEY="key")
    set_ui_settings(new_settings)
    monkeypatch.setattr(ui_app, "settings", new_settings)
    monkeypatch.setattr(
        "app.infrastructure.metrics.get_metrics", lambda: DummyMetrics()
    )

    res = client.get("/admin/health", headers={"X-Admin-Key": "key"})
    assert res.status_code == 200
    assert res.get_json()["status"] == "healthy"


def test_admin_metrics_recent(monkeypatch, client, test_settings):
    from app.ui import set_settings as set_ui_settings

    class DummyMetrics:
        def get_recent_metrics(self, hours=24):
            return {"total_executions": 1}

        def get_health_status(self):
            return {"status": "healthy"}

    new_settings = replace(test_settings, ADMIN_API_KEY="key")
    set_ui_settings(new_settings)
    monkeypatch.setattr(ui_app, "settings", new_settings)
    monkeypatch.setattr(
        "app.infrastructure.metrics.get_metrics", lambda: DummyMetrics()
    )

    res = client.get("/admin/metrics", headers={"X-Admin-Key": "key"})
    assert res.status_code == 200
    data = res.get_json()
    assert "metrics" in data


def test_admin_force_rebalance_authorized(monkeypatch, client, test_settings):
    from app.ui import set_settings as set_ui_settings

    new_settings = replace(test_settings, ADMIN_API_KEY="key")
    set_ui_settings(new_settings)
    monkeypatch.setattr(ui_app, "settings", new_settings)
    res = client.post("/admin/force-rebalance", headers={"X-Admin-Key": "key"})
    assert res.status_code == 200
