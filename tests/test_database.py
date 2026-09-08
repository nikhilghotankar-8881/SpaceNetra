"""
Unit Tests for SpaceNetra Relational & Spatial Database ORM Models.
"""

import pytest
from backend.database.connection import DatabaseManager
from backend.database.models import (
    AnalystFeedbackModel,
    ChangeEventModel,
    ProvenanceModel,
    SceneModel,
    TileModel,
)


@pytest.fixture
def db_manager():
    manager = DatabaseManager("sqlite:///:memory:")
    manager.create_tables()
    return manager


def test_scene_model_crud(db_manager: DatabaseManager):
    with db_manager.get_session() as session:
        scene = SceneModel(
            scene_id="S2A_20230101_AOI1",
            sensor="Sentinel-2A",
            acquisition_date="2023-01-01",
            cloud_cover=2.5,
            filepath="data/raw/scene1.tif",
        )
        session.add(scene)

    with db_manager.get_session() as session:
        queried = session.query(SceneModel).filter_by(scene_id="S2A_20230101_AOI1").first()
        assert queried is not None
        assert queried.sensor == "Sentinel-2A"
        assert queried.cloud_cover == 2.5
        d = queried.to_dict()
        assert d["scene_id"] == "S2A_20230101_AOI1"


def test_change_event_model_crud(db_manager: DatabaseManager):
    with db_manager.get_session() as session:
        event = ChangeEventModel(
            event_id="EVENT_0001",
            location="28.61°N, 77.21°E",
            area_m2=250000.0,
            first_observed="2023-01-15",
            confidence_score=0.92,
            status="DETECTED",
        )
        session.add(event)

    with db_manager.get_session() as session:
        queried = session.query(ChangeEventModel).filter_by(event_id="EVENT_0001").first()
        assert queried is not None
        assert queried.area_m2 == 250000.0
        assert queried.status == "DETECTED"


def test_analyst_feedback_and_provenance(db_manager: DatabaseManager):
    with db_manager.get_session() as session:
        event = ChangeEventModel(
            event_id="EVENT_0002",
            area_m2=120000.0,
        )
        session.add(event)

        feedback = AnalystFeedbackModel(
            feedback_id="FB_001",
            event_id="EVENT_0002",
            analyst_id="analyst_007",
            decision="ACCEPT",
            notes="Confirmed construction site development.",
        )
        session.add(feedback)

        provenance = ProvenanceModel(
            provenance_id="PROV_001",
            event_id="EVENT_0002",
            model_name="siamese_unet",
            checkpoint_hash="sha256_abc123",
        )
        session.add(provenance)

    with db_manager.get_session() as session:
        fb = session.query(AnalystFeedbackModel).filter_by(feedback_id="FB_001").first()
        prov = session.query(ProvenanceModel).filter_by(provenance_id="PROV_001").first()

        assert fb is not None
        assert fb.decision == "ACCEPT"
        assert prov is not None
        assert prov.model_name == "siamese_unet"
