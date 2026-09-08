"""
Unit tests for SpaceNetra Telemetry, Prometheus Metrics, & Deep Health Diagnostics (Phase 29).
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.monitoring.telemetry import TelemetryMonitor, default_telemetry


@pytest.fixture
def client():
    return TestClient(app)


def test_telemetry_monitor_recording():
    telemetry = TelemetryMonitor()
    telemetry.record_latency(120.5)
    telemetry.record_latency(150.0)
    telemetry.record_error()
    telemetry.record_processed_tiles(10)

    summary = telemetry.get_metrics_summary()
    assert summary["total_requests"] == 2
    assert summary["total_errors"] == 1
    assert summary["processed_tiles"] == 10
    assert summary["latency_ms"]["avg"] > 0.0
    assert summary["system_resources"]["ram_usage_mb"] > 0.0


def test_prometheus_exporter_format():
    telemetry = TelemetryMonitor()
    telemetry.record_latency(85.0)
    telemetry.record_processed_tiles(5)

    prom_text = telemetry.export_prometheus_format()
    assert "# HELP spacenetra_uptime_seconds" in prom_text
    assert "spacenetra_requests_total 1" in prom_text
    assert "spacenetra_processed_tiles_total 5" in prom_text
    assert "spacenetra_ram_usage_mb" in prom_text


def test_metrics_and_deep_health_endpoints(client):
    # Test Prometheus metrics endpoint
    metrics_resp = client.get("/api/metrics")
    assert metrics_resp.status_code == 200
    assert "spacenetra_uptime_seconds" in metrics_resp.text

    # Test Deep Health Check endpoint
    health_resp = client.get("/api/health/deep")
    assert health_resp.status_code == 200
    data = health_resp.json()
    assert data["status"] == "HEALTHY"
    assert data["components"]["database_orm"]["status"] == "UP"
    assert data["components"]["vector_search"]["status"] == "UP"
    assert "metrics" in data
