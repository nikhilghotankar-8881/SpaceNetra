"""
Unit tests for SpaceNetra Authentication, JWT Security, & RBAC Engine (Phase 27).
"""

import time
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from backend.database.models import Base, UserModel
from backend.services.auth_service import AuthService, Role


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


def test_password_hashing_and_verification():
    password = "SuperSecretPassword123!"
    hashed = AuthService.hash_password(password)

    assert hashed != password
    assert "$" in hashed
    assert AuthService.verify_password(password, hashed) is True
    assert AuthService.verify_password("WrongPassword", hashed) is False


def test_jwt_token_generation_and_validation():
    payload = {"sub": "analyst_01", "email": "analyst@spacenetra.ai", "role": "ANALYST"}
    token = AuthService.create_access_token(payload, expires_delta_seconds=60)

    decoded = AuthService.verify_access_token(token)
    assert decoded["sub"] == "analyst_01"
    assert decoded["email"] == "analyst@spacenetra.ai"
    assert decoded["role"] == "ANALYST"
    assert "exp" in decoded

    # Invalid signature check
    with pytest.raises(ValueError, match="signature"):
        AuthService.verify_access_token(token, secret_key="wrong_secret_key")

    # Expired token check
    expired_token = AuthService.create_access_token(payload, expires_delta_seconds=-10)
    with pytest.raises(ValueError, match="expired"):
        AuthService.verify_access_token(expired_token)


def test_role_permissions_hierarchy():
    assert AuthService.has_role_permission("ADMIN", Role.ADMIN) is True
    assert AuthService.has_role_permission("ADMIN", Role.ANALYST) is True
    assert AuthService.has_role_permission("ADMIN", Role.OBSERVER) is True

    assert AuthService.has_role_permission("ANALYST", Role.ADMIN) is False
    assert AuthService.has_role_permission("ANALYST", Role.ANALYST) is True
    assert AuthService.has_role_permission("ANALYST", Role.OBSERVER) is True

    assert AuthService.has_role_permission("OBSERVER", Role.ADMIN) is False
    assert AuthService.has_role_permission("OBSERVER", Role.ANALYST) is False
    assert AuthService.has_role_permission("OBSERVER", Role.OBSERVER) is True


def test_user_model_crud(db_session):
    user = UserModel(
        username="john_doe",
        email="john@spacenetra.ai",
        hashed_password=AuthService.hash_password("password123"),
        role="ANALYST",
    )
    db_session.add(user)
    db_session.commit()

    retrieved = db_session.query(UserModel).filter_by(username="john_doe").first()
    assert retrieved is not None
    assert retrieved.email == "john@spacenetra.ai"
    assert retrieved.role == "ANALYST"

    user_dict = retrieved.to_dict()
    assert user_dict["username"] == "john_doe"
    assert user_dict["role"] == "ANALYST"
    assert user_dict["is_active"] is True


def test_auth_endpoints_login_register_me(client):
    # 1. Login with dev credentials
    response = client.post("/api/auth/login", json={"username": "analyst", "password": "analyst123"})
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["role"] == "ANALYST"

    token = data["access_token"]

    # 2. Test /me with valid token
    me_resp = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 200
    me_data = me_resp.json()
    assert me_data["username"] == "analyst"
    assert me_data["permissions"]["can_run_inference"] is True
    assert me_data["permissions"]["can_manage_users"] is False

    # 3. Test /register new user
    reg_resp = client.post("/api/auth/register", json={
        "username": "new_analyst",
        "email": "new@spacenetra.ai",
        "password": "secure_pass_123",
        "role": "ANALYST"
    })
    assert reg_resp.status_code == 200
    assert reg_resp.json()["status"] == "SUCCESS"

    # 4. Login with newly registered user
    login_new = client.post("/api/auth/login", json={"username": "new_analyst", "password": "secure_pass_123"})
    assert login_new.status_code == 200
