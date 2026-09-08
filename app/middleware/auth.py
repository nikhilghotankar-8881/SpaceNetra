"""
Authentication & Role-Based Access Control (RBAC) Dependency Injectors for FastAPI.
"""

from typing import Callable, Dict, Any, Optional
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer

from backend.services.auth_service import AuthService, Role

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login", auto_error=False)


async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """
    Extract and validate JWT token from request header.
    Returns decoded token payload if valid.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        payload = AuthService.verify_access_token(token)
        return payload
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_role(required_role: Role) -> Callable:
    """
    FastAPI dependency factory enforcing Role-Based Access Control (RBAC).
    Usage: @app.post('/endpoint', dependencies=[Depends(require_role(Role.ANALYST))])
    """
    async def role_checker(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        user_role = current_user.get("role", "OBSERVER")
        if not AuthService.has_role_permission(user_role, required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user_role}' insufficient. Required role: '{required_role.value}'.",
            )
        return current_user

    return role_checker
