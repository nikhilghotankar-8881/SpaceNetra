"""
FastAPI Authentication & RBAC REST Endpoints for SpaceNetra.
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Header, Depends
from pydantic import BaseModel, EmailStr
from backend.services.auth_service import AuthService, Role

router = APIRouter(prefix="/auth", tags=["Authentication & Security"])

# Default in-memory seed user for dev/testing
DEV_USERS = {
    "admin": {
        "username": "admin",
        "email": "admin@spacenetra.ai",
        "hashed_password": AuthService.hash_password("admin123"),
        "role": "ADMIN",
        "is_active": True,
    },
    "analyst": {
        "username": "analyst",
        "email": "analyst@spacenetra.ai",
        "hashed_password": AuthService.hash_password("analyst123"),
        "role": "ANALYST",
        "is_active": True,
    },
}


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    role: Optional[str] = "ANALYST"


@router.post("/login", summary="Authenticate User & Obtain JWT Access Token")
def login(request: LoginRequest) -> Dict[str, Any]:
    """Verify credentials and return JWT bearer token."""
    user = DEV_USERS.get(request.username.lower())
    if not user or not AuthService.verify_password(request.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    token = AuthService.create_access_token({
        "sub": user["username"],
        "email": user["email"],
        "role": user["role"],
    })

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "username": user["username"],
            "email": user["email"],
            "role": user["role"],
        },
    }


@router.post("/register", summary="Register New User Account")
def register(request: RegisterRequest) -> Dict[str, Any]:
    """Register a new user account."""
    username = request.username.lower()
    if username in DEV_USERS:
        raise HTTPException(status_code=400, detail="Username already registered.")

    role_str = request.role.upper() if request.role else "ANALYST"
    if role_str not in [r.value for r in Role]:
        raise HTTPException(status_code=400, detail=f"Invalid role. Must be one of {[r.value for r in Role]}")

    DEV_USERS[username] = {
        "username": username,
        "email": request.email,
        "hashed_password": AuthService.hash_password(request.password),
        "role": role_str,
        "is_active": True,
    }

    return {
        "status": "SUCCESS",
        "message": f"User {username} registered successfully.",
        "user": {
            "username": username,
            "email": request.email,
            "role": role_str,
        },
    }


@router.get("/me", summary="Get Current Authenticated User Profile")
def get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Validate bearer token and return user profile with role permissions."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")

    token = authorization.split(" ")[1]
    try:
        payload = AuthService.verify_access_token(token)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    role = payload.get("role", "OBSERVER")
    return {
        "username": payload.get("sub"),
        "email": payload.get("email"),
        "role": role,
        "permissions": {
            "can_run_inference": AuthService.has_role_permission(role, Role.ANALYST),
            "can_submit_feedback": AuthService.has_role_permission(role, Role.ANALYST),
            "can_manage_users": AuthService.has_role_permission(role, Role.ADMIN),
        },
    }
