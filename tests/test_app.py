"""
Unit Tests for SpaceNetra FastAPI Web Application & REST API Service.
"""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data


def test_index_endpoint():
    response = client.get("/")
    assert response.status_code == 200


def test_predict_endpoint():
    response = client.post("/api/predict", json={"architecture": "siamese_unet", "threshold": 0.5, "dry_run": True})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "probability_map_shape" in data


def test_multitemporal_endpoint():
    response = client.post("/api/multitemporal", json={})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["scenes_processed"] == 4


def test_search_endpoint():
    response = client.post("/api/search", json={"query": "solar panels in desert", "top_k": 3})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["query"] == "solar panels in desert"
    assert len(data["results"]) == 3


def test_score_endpoint():
    response = client.post("/api/score", json={"dry_run": True})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "quality_report" in data
