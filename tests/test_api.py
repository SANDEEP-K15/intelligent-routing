import pytest
from fastapi.testclient import TestClient

from service import app as app_module


@pytest.fixture()
def client(model_path, monkeypatch):
    monkeypatch.setenv("KESTREL_MODEL_PATH", str(model_path))
    app_module.get_model.cache_clear()
    yield TestClient(app_module.app)
    app_module.get_model.cache_clear()


@pytest.fixture()
def client_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_MODEL_PATH", str(tmp_path / "missing.joblib"))
    app_module.get_model.cache_clear()
    yield TestClient(app_module.app)
    app_module.get_model.cache_clear()


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["cost_per_prediction_rs"] == 0
    assert "product_family" in body["inputs_used"]


def test_route_happy_path(client):
    r = client.post("/api/route", json={
        "request_text": "good morning, my emi conversion for the air fryer is still pending",
        "channel": "chat", "warranty_status": "in_warranty", "product_family": "Air Fryer",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["team"] == "Billing"
    assert set(body) >= {"team", "confidence", "confidence_band", "reasons", "alternatives", "latency_ms"}
    assert 0 <= body["confidence"] <= 1
    assert body["reasons"]


def test_route_text_only(client):
    r = client.post("/api/route", json={"request_text": "the installer never came for my new fan"})
    assert r.status_code == 200
    assert r.json()["team"] == "Installs & Demo"
    assert any("was not provided" in x for x in r.json()["reasons"])


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"request_text": ""},
        {"request_text": "   "},
        {"request_text": 123},
        {"request_text": "x" * 2001},
        {"request_text": "fan noise", "channel": "fax"},
        {"request_text": "fan noise", "product_family": "Toaster"},
        {"request_text": "fan noise", "warranty_status": "expired"},
        {"request_text": "fan noise", "final_team": "Repairs"},
    ],
)
def test_invalid_input_rejected(client, payload):
    r = client.post("/api/route", json=payload)
    assert r.status_code == 422


def test_missing_model_fails_politely(client_without_model):
    assert client_without_model.get("/api/health").json()["status"] == "model_missing"
    r = client_without_model.post("/api/route", json={"request_text": "fan noise"})
    assert r.status_code == 503
    assert "scripts.train" in r.json()["detail"]


def test_ui_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "/api/route" in r.text
