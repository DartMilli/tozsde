"""S17: Web API, bootstrap, and model_promotion_gate coverage boost.

Targets:
- app/interfaces/web/app.py (38%)
- app/models/model_promotion_gate.py (65%)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import json
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Web API (Flask Blueprint)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def web_client():
    """Provide a Flask test client with mocked container."""
    from app.interfaces.web.app import create_api_blueprint
    from flask import Flask

    mock_container = MagicMock()
    blueprint = create_api_blueprint(container=mock_container)
    app = Flask(__name__)
    app.register_blueprint(blueprint)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client, mock_container


class TestWalkForwardEndpoint:
    def test_success(self, web_client):
        client, container = web_client
        container.walk_forward.run.return_value = {"status": "ok", "data": {}}
        resp = client.get("/walk-forward?ticker=VOO")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["status"] == "ok"

    def test_exception_returns_500(self, web_client):
        client, container = web_client
        container.walk_forward.run.side_effect = RuntimeError("boom")
        resp = client.get("/walk-forward?ticker=VOO")
        assert resp.status_code == 500

    def test_no_ticker_runs_all(self, web_client):
        client, container = web_client
        container.walk_forward.run.return_value = {"status": "ok", "data": {}}
        resp = client.get("/walk-forward")
        assert resp.status_code == 200


class TestDailyPipelineEndpoint:
    def test_post_success(self, web_client):
        client, container = web_client
        container.daily_pipeline.run.return_value = {
            "status": "ok",
            "data": {"processed": 1},
        }
        resp = client.post("/daily-pipeline", json={"ticker": "VOO", "dry_run": True})
        assert resp.status_code == 200

    def test_post_exception_returns_500(self, web_client):
        client, container = web_client
        container.daily_pipeline.run.side_effect = RuntimeError("pipeline error")
        resp = client.post("/daily-pipeline", json={})
        assert resp.status_code == 500

    def test_post_empty_body(self, web_client):
        client, container = web_client
        container.daily_pipeline.run.return_value = {"status": "ok", "data": {}}
        resp = client.post("/daily-pipeline")
        assert resp.status_code == 200


class TestTrainRlEndpoint:
    def test_missing_ticker_returns_400(self, web_client):
        client, container = web_client
        resp = client.post("/train-rl", json={})
        assert resp.status_code == 400
        data = json.loads(resp.data)
        assert "ticker" in data.get("error", {}).get("message", "").lower()

    def test_success(self, web_client):
        client, container = web_client
        container.train_rl.run.return_value = {"status": "ok", "data": {}}
        resp = client.post("/train-rl", json={"ticker": "VOO"})
        assert resp.status_code == 200

    def test_exception_returns_500(self, web_client):
        client, container = web_client
        container.train_rl.run.side_effect = RuntimeError("training error")
        resp = client.post("/train-rl", json={"ticker": "VOO"})
        assert resp.status_code == 500


class TestValidateModelEndpoint:
    def test_success(self, web_client):
        client, container = web_client
        container.validate_model.run.return_value = {"status": "ok"}
        resp = client.post("/validate-model", json={"mode": "quick"})
        assert resp.status_code == 200

    def test_exception_returns_500(self, web_client):
        client, container = web_client
        container.validate_model.run.side_effect = RuntimeError("validation failed")
        resp = client.post("/validate-model", json={})
        assert resp.status_code == 500


# ─────────────────────────────────────────────────────────────────────────────
# ModelPromotionGate
# ─────────────────────────────────────────────────────────────────────────────


class TestModelPromotionGate:
    def _make_gate(self):
        from app.models.model_promotion_gate import (
            ModelPromotionGate,
            PromotionDecision,
        )

        with patch(
            "app.models.model_promotion_gate._create_data_repository"
        ) as mock_repo:
            dm = MagicMock()
            mock_repo.return_value = dm
            gate = ModelPromotionGate()
        gate.dm = dm
        return gate, dm

    def test_evaluate_candidate_passes(self):
        gate, dm = self._make_gate()
        dm.update_model_status.return_value = None
        with patch.object(
            gate,
            "_load_baseline_metrics",
            return_value={
                "wf_stability": 0.5,
                "max_drawdown": 0.10,
                "effectiveness": 0.5,
            },
        ):
            from app.config.config import Config

            result = gate.evaluate_candidate(
                ticker="AAPL",
                candidate_model_id="m1",
                candidate_metrics={
                    "wf_stability": 0.9,
                    "max_drawdown": 0.05,
                    "effectiveness": 0.9,
                    "safety_override_rate": 0.0,
                },
            )
        assert result.allow is True

    def test_evaluate_candidate_fails_drawdown(self):
        gate, dm = self._make_gate()
        dm.update_model_status.return_value = None
        with patch.object(
            gate,
            "_load_baseline_metrics",
            return_value={
                "wf_stability": 0.9,
                "max_drawdown": 0.05,
                "effectiveness": 0.9,
            },
        ):
            result = gate.evaluate_candidate(
                ticker="AAPL",
                candidate_model_id="m2",
                candidate_metrics={
                    "wf_stability": 0.9,
                    "max_drawdown": 0.99,
                    "effectiveness": 0.9,
                    "safety_override_rate": 0.0,
                },
            )
        assert result.allow is False
        assert "MAX_DRAWDOWN_REGRESSION" in result.reasons

    def test_compute_candidate_metrics_returns_dict(self):
        gate, dm = self._make_gate()
        with patch.object(gate, "_load_wf_stability", return_value=0.7), patch.object(
            gate, "_load_effectiveness", return_value=0.6
        ), patch.object(gate, "_load_max_drawdown", return_value=0.08), patch.object(
            gate, "_load_safety_override_rate", return_value=0.02
        ):
            metrics = gate.compute_candidate_metrics("AAPL")
        assert metrics["wf_stability"] == 0.7
        assert metrics["max_drawdown"] == 0.08
