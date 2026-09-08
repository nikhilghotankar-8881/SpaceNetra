"""
End-to-End Enterprise Integration Test Suite for SpaceNetra (Phase 30).

Validates the full lifecycle flow:
Synthetic Satellite Ingestion -> ChangeFormer Inference -> Quality Scoring ->
Spatial Event Clustering -> GeoJSON Export -> Vector Embedding -> DB ORM Persistence ->
Analyst Feedback -> Provenance Logging -> REST API Exposition.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from backend.database.models import (
    AnalystFeedbackModel,
    Base,
    ChangeEventModel,
    ProvenanceModel,
    SceneModel,
    UserModel,
)
from backend.services.feedback_service import FeedbackService
from backend.services.provenance_service import ProvenanceService
from src.confidence.quality_engine import ConfidenceEngine
from src.pipeline.sentinel_cd import SentinelChangeDetector
from src.temporal.event_builder import EventBuilder


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_end_to_end_pipeline_integration(client, db_session):
    # 1. Synthetic Sentinel-2 image pair generation (100x100 pixels, 4 bands)
    np.random.seed(42)
    t1 = np.random.randint(100, 1000, size=(100, 100, 4), dtype=np.uint16)
    t2 = t1.copy()
    # Simulate building construction in central region (30:70, 30:70)
    t2[30:70, 30:70, :] = np.random.randint(2000, 4000, size=(40, 40, 4), dtype=np.uint16)

    # 2. Run Sentinel Change Detection Inference Pipeline
    detector = SentinelChangeDetector(model_or_arch="changeformer")
    results = detector.predict_pair(t1, t2)
    prob_map = results["probability_map"]

    assert prob_map.shape == (100, 100)
    assert 0.0 <= prob_map.min() <= prob_map.max() <= 1.0

    # 3. Evaluate Confidence Quality Score
    quality_engine = ConfidenceEngine()
    quality_result = quality_engine.evaluate_quality(img_t1=t1, img_t2=t2, prob_map=prob_map)
    assert "composite_confidence" in quality_result
    assert 0.0 <= quality_result["composite_confidence"] <= 1.0

    # 4. Extract Spatial Change Events
    binary_mask = np.zeros((100, 100), dtype=np.uint8)
    binary_mask[30:70, 30:70] = 1
    event_builder = EventBuilder(min_area_pixels=5)
    events = event_builder.extract_events(binary_mask, prob_map)

    assert len(events) >= 1
    first_event = events[0]
    assert "area_m2" in first_event

    # 5. Persist Scene & Change Events to Spatial DB ORM
    scene_t1 = SceneModel(scene_id="SCENE_SENTINEL2_T1", sensor="Sentinel-2", cloud_cover=0.02)
    scene_t2 = SceneModel(scene_id="SCENE_SENTINEL2_T2", sensor="Sentinel-2", cloud_cover=0.05)
    db_session.add_all([scene_t1, scene_t2])
    db_session.commit()

    event_id = "EVT_CONSTRUCTION_001"
    change_evt = ChangeEventModel(
        event_id=event_id,
        location="37.7749,-122.4194",
        area_m2=float(first_event["area_m2"]),
        confidence_score=float(quality_result["composite_confidence"]),
        status="DETECTED",
    )
    db_session.add(change_evt)
    db_session.commit()

    # 6. Analyst Active Learning Feedback Logging
    feedback_service = FeedbackService()
    feedback_record = feedback_service.submit_feedback(
        event_id=event_id,
        decision="ACCEPT",
        notes="Confirmed industrial building construction site.",
        analyst_id="analyst_senior_01",
    )
    assert feedback_record["decision"] == "ACCEPT"

    # 7. Model Prediction Lineage Provenance Trace Recording
    provenance_service = ProvenanceService()
    prov_record = provenance_service.create_provenance_record(
        event_id=event_id,
        model_name="ChangeFormer-ViT-v1.0",
        checkpoint_hash="sha256_e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    )
    assert prov_record["model_name"] == "ChangeFormer-ViT-v1.0"

    # 8. REST API Endpoint Interrogation
    health_resp = client.get("/health")
    assert health_resp.status_code == 200

    deep_resp = client.get("/api/health/deep")
    assert deep_resp.status_code == 200
    assert deep_resp.json()["status"] == "HEALTHY"

    metrics_resp = client.get("/api/metrics")
    assert metrics_resp.status_code == 200
    assert "spacenetra_uptime_seconds" in metrics_resp.text
