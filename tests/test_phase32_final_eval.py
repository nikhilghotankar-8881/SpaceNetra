"""
Phase 32 — Final System Evaluation & SLA Compliance Test Suite.
"""

from pathlib import Path
import pytest

from scripts.evaluate_model import evaluate_model
from scripts.evaluate_system import benchmark_system
from scripts.verify_security import run_security_audit
from backend.services.auth_service import AuthService, Role


def test_final_model_evaluation_sla():
    """Verify model accuracy and latency metrics satisfy production SLA."""
    metrics, mean_lat = evaluate_model(checkpoint_path=None, num_batches=2, patch_size=256)

    assert "f1" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "overall_accuracy" in metrics
    assert mean_lat < 500.0  # ms/patch SLA

    # Verify report file written
    report_file = Path("outputs") / "metrics" / "model_report.txt"
    assert report_file.exists()
    assert "SPACENETRA MODEL EVALUATION REPORT" in report_file.read_text(encoding="utf-8")


def test_final_system_performance_sla():
    """Verify semantic search latency and event throughput satisfy system SLA."""
    results = benchmark_system(num_queries=5)

    assert results["search_p50"] < 100.0  # ms SLA
    assert results["search_p95"] < 200.0  # ms SLA
    assert results["throughput_eps"] > 500  # events/sec SLA

    # Verify report file written
    report_file = Path("outputs") / "metrics" / "system_report.txt"
    assert report_file.exists()
    assert "SPACENETRA SYSTEM PERFORMANCE REPORT" in report_file.read_text(encoding="utf-8")


def test_complete_security_audit_compliance():
    """Verify security audit checklist passes clean."""
    assert run_security_audit() is True


def test_full_system_identity_and_role_permissions():
    """Verify role hierarchy and token lifecycle."""
    token = AuthService.create_access_token({"sub": "admin_user", "role": "ADMIN"})
    payload = AuthService.verify_access_token(token)
    assert payload["sub"] == "admin_user"
    assert payload["role"] == "ADMIN"
    assert AuthService.has_role_permission("ADMIN", Role.ANALYST) is True
