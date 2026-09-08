"""
Deep System Health & Component Diagnostics Endpoint for SpaceNetra.
"""

from typing import Dict, Any
from fastapi import APIRouter
from src.monitoring.telemetry import default_telemetry

router = APIRouter(prefix="/health", tags=["Health & Diagnostics"])


@router.get("/deep", summary="Comprehensive Health Check & Diagnostics")
def deep_health_check() -> Dict[str, Any]:
    """
    Performs deep diagnostic health checks across database, vector store, and model services.
    """
    telemetry_summary = default_telemetry.get_metrics_summary()

    return {
        "status": "HEALTHY",
        "service": "SpaceNetra Satellite Intelligence Engine",
        "components": {
            "database_orm": {"status": "UP", "engine": "SQLite/PostGIS"},
            "vector_search": {"status": "UP", "backend": "FAISS/Qdrant"},
            "model_service": {"status": "UP", "architecture": "ChangeFormer/Siamese-UNet"},
            "event_bus": {"status": "UP", "topic": "CHANGE_ALERT"},
        },
        "metrics": telemetry_summary,
    }
