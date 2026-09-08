"""
Unit Tests for Analyst Feedback Review & Provenance Tracking Services.
"""

import numpy as np
import pytest
from backend.database.connection import DatabaseManager
from backend.services.feedback_service import FeedbackService
from backend.services.provenance_service import ProvenanceService


@pytest.fixture
def db_manager():
    manager = DatabaseManager("sqlite:///:memory:")
    manager.create_tables()
    return manager


def test_feedback_service_submission_and_stats(db_manager: DatabaseManager):
    service = FeedbackService(db_manager=db_manager)

    # Empty stats initial check
    empty_stats = service.get_feedback_stats()
    assert empty_stats["total_reviews"] == 0
    assert empty_stats["acceptance_rate_pct"] == 0.0

    # Submit 3 feedback decisions
    res1 = service.submit_feedback(event_id="EVENT_001", decision="ACCEPT", notes="Valid change")
    res2 = service.submit_feedback(event_id="EVENT_002", decision="REJECT", notes="Cloud artifact")
    res3 = service.submit_feedback(event_id="EVENT_003", decision="REVIEW", notes="Needs 2nd review")

    assert res1["decision"] == "ACCEPT"
    assert res2["decision"] == "REJECT"

    # Verify invalid decision raises ValueError
    with pytest.raises(ValueError):
        service.submit_feedback(event_id="EVENT_004", decision="INVALID_DECISION")

    # Verify aggregate statistics
    stats = service.get_feedback_stats()
    assert stats["total_reviews"] == 3
    assert stats["accepted_count"] == 1
    assert stats["rejected_count"] == 1
    assert stats["review_count"] == 1
    assert np.isclose(stats["acceptance_rate_pct"], 33.33, atol=1e-2)


def test_provenance_service_creation_and_trace(db_manager: DatabaseManager):
    service = ProvenanceService(db_manager=db_manager)

    res = service.create_provenance_record(
        event_id="EVENT_010",
        model_name="changeformer",
        checkpoint_hash="sha256_checkpoint_123",
    )

    assert "provenance_id" in res
    assert res["event_id"] == "EVENT_010"

    trace = service.get_provenance_trace(event_id="EVENT_010")
    assert trace is not None
    assert trace["model_name"] == "changeformer"
    assert trace["checkpoint_hash"] == "sha256_checkpoint_123"

    # Non-existent trace returns None
    missing = service.get_provenance_trace(event_id="EVENT_999")
    assert missing is None
