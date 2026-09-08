"""
Prometheus Metrics REST Exporter Endpoint for SpaceNetra.
"""

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from src.monitoring.telemetry import default_telemetry

router = APIRouter(prefix="/metrics", tags=["System Telemetry & Metrics"])


@router.get("", response_class=PlainTextResponse, summary="Export Prometheus Performance Metrics")
def export_prometheus_metrics():
    """Returns telemetry and latency metrics in Prometheus exposition text format."""
    return default_telemetry.export_prometheus_format()
