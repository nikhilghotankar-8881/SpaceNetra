"""
Unit Tests for Phase 31 Security Hardening Middleware, Verifier, and RBAC.
"""

from pathlib import Path
import tempfile
import time
import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

from app.middleware.auth import get_current_user, require_role
from app.middleware.audit import AuditMiddleware
from app.middleware.security import SecurityHeadersMiddleware, RateLimitMiddleware
from backend.services.auth_service import AuthService, Role
from src.pipeline.integrity import ModelIntegrityVerifier
from scripts.verify_security import run_security_audit


def test_unauthenticated_request_rejected():
    """Test that protected endpoint rejects requests without JWT token."""
    app = FastAPI()

    @app.get("/protected")
    def protected(user=Depends(get_current_user)):
        return {"user": user}

    client = TestClient(app)
    response = client.get("/protected")
    assert response.status_code == 401
    assert "token required" in response.json()["detail"].lower()


def test_valid_jwt_passes_auth():
    """Test that valid JWT token allows access and returns payload."""
    app = FastAPI()

    @app.get("/protected")
    def protected(user=Depends(get_current_user)):
        return {"sub": user["sub"], "role": user["role"]}

    token = AuthService.create_access_token({"sub": "analyst1", "role": "ANALYST"})
    client = TestClient(app)
    response = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["sub"] == "analyst1"
    assert response.json()["role"] == "ANALYST"


def test_expired_jwt_rejected():
    """Test that expired JWT token is rejected with 401."""
    app = FastAPI()

    @app.get("/protected")
    def protected(user=Depends(get_current_user)):
        return {"user": user}

    token = AuthService.create_access_token(
        {"sub": "analyst1", "role": "ANALYST"},
        expires_delta_seconds=-10,  # Already expired
    )
    client = TestClient(app)
    response = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401
    assert "expired" in response.json()["detail"].lower()


def test_rbac_permission_allowed_and_denied():
    """Test RBAC decorator allows higher role and rejects lower role."""
    app = FastAPI()

    @app.get("/analyst-only", dependencies=[Depends(require_role(Role.ANALYST))])
    def analyst_only():
        return {"status": "ok"}

    client = TestClient(app)

    # Observer -> 403
    obs_token = AuthService.create_access_token({"sub": "obs1", "role": "OBSERVER"})
    res_obs = client.get("/analyst-only", headers={"Authorization": f"Bearer {obs_token}"})
    assert res_obs.status_code == 403

    # Analyst -> 200
    ana_token = AuthService.create_access_token({"sub": "ana1", "role": "ANALYST"})
    res_ana = client.get("/analyst-only", headers={"Authorization": f"Bearer {ana_token}"})
    assert res_ana.status_code == 200

    # Admin -> 200
    adm_token = AuthService.create_access_token({"sub": "adm1", "role": "ADMIN"})
    res_adm = client.get("/analyst-only", headers={"Authorization": f"Bearer {adm_token}"})
    assert res_adm.status_code == 200


def test_security_headers_present():
    """Test that security headers middleware adds expected HTTP headers."""
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/ping")
    def ping():
        return {"msg": "pong"}

    client = TestClient(app)
    res = client.get("/ping")
    assert res.status_code == 200
    assert res.headers["X-Content-Type-Options"] == "nosniff"
    assert res.headers["X-Frame-Options"] == "DENY"
    assert res.headers["X-XSS-Protection"] == "1; mode=block"
    assert "Strict-Transport-Security" in res.headers


def test_audit_middleware_logs():
    """Test audit middleware captures request and writes JSONL log entry."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "audit.jsonl"
        app = FastAPI()
        app.add_middleware(AuditMiddleware, log_filepath=log_path)

        @app.get("/test-audit")
        def test_audit():
            return {"ok": True}

        client = TestClient(app)
        res = client.get("/test-audit")
        assert res.status_code == 200
        assert "X-Request-ID" in res.headers

        assert log_path.exists()
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        assert "test-audit" in lines[0]
        assert "latency_ms" in lines[0]


def test_rate_limit_enforcement():
    """Test rate limiting middleware blocks after threshold exceeded."""
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, window_seconds=10, enabled=True, limits={"DEFAULT": 3})

    @app.get("/limited")
    def limited():
        return {"ok": True}

    client = TestClient(app)

    # First 3 succeed
    for _ in range(3):
        assert client.get("/limited").status_code == 200

    # 4th fails with 429
    res_blocked = client.get("/limited")
    assert res_blocked.status_code == 429
    assert "Retry-After" in res_blocked.headers


def test_model_integrity_verifier():
    """Test SHA-256 model checksum calculation and manifest verification."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir)
        dummy_model = ckpt_dir / "model.pth"
        dummy_model.write_bytes(b"dummy_weights_12345")

        checksum = ModelIntegrityVerifier.compute_checksum(dummy_model)
        assert len(checksum) == 64  # SHA-256 hex length

        assert ModelIntegrityVerifier.verify_checksum(dummy_model, checksum)
        assert not ModelIntegrityVerifier.verify_checksum(dummy_model, "wrong_hash")

        manifest_file = ckpt_dir / "CHECKSUMS.sha256"
        manifest = ModelIntegrityVerifier.generate_manifest(ckpt_dir, manifest_file)
        assert "model.pth" in manifest

        assert ModelIntegrityVerifier.verify_manifest(manifest_file, ckpt_dir)


def test_verify_security_script():
    """Test running security audit script returns True."""
    success = run_security_audit()
    assert success is True
