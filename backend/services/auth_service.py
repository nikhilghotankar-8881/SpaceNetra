"""
Authentication & Role-Based Access Control (RBAC) Service for SpaceNetra.

Provides password hashing, JWT token creation/validation, and RBAC authorization routines.
"""

import base64
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import os
from enum import Enum
from typing import Any, Dict, Optional

SECRET_KEY = os.getenv("SPACENETRA_SECRET_KEY", "spacenetra_dev_jwt_secret_key_2026")
ALGORITHM = "HS256"


class Role(str, Enum):
    ADMIN = "ADMIN"
    ANALYST = "ANALYST"
    OBSERVER = "OBSERVER"


# Role priority hierarchy for authorization checks
ROLE_HIERARCHY = {
    Role.ADMIN: 30,
    Role.ANALYST: 20,
    Role.OBSERVER: 10,
}


class AuthService:
    """Service handling credential verification, JWT generation, and RBAC permissions."""

    @staticmethod
    def hash_password(password: str, salt: Optional[bytes] = None) -> str:
        """Hash a password using PBKDF2-HMAC-SHA256 with a salt."""
        if salt is None:
            salt = os.urandom(16)
        pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
        return f"{salt.hex()}${pwd_hash.hex()}"

    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        """Verify a plain password against a salt$hash string."""
        try:
            salt_hex, hash_hex = hashed_password.split("$")
            salt = bytes.fromhex(salt_hex)
            expected_hash = bytes.fromhex(hash_hex)
            computed_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
            return hmac.compare_digest(computed_hash, expected_hash)
        except Exception:
            return False

    @staticmethod
    def create_access_token(
        data: Dict[str, Any],
        secret_key: str = SECRET_KEY,
        expires_delta_seconds: int = 3600,
    ) -> str:
        """Create a signed JWT bearer token (HS256)."""
        payload = data.copy()
        now = datetime.now(timezone.utc)
        payload.update({
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(seconds=expires_delta_seconds)).timestamp()),
        })

        header = {"alg": "HS256", "typ": "JWT"}
        
        # Base64URL encode header and payload
        header_b64 = base64.urlsafe_b64encode(
            json.dumps(header, separators=(",", ":")).encode("utf-8")
        ).decode("utf-8").rstrip("=")
        
        payload_b64 = base64.urlsafe_b64encode(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
        ).decode("utf-8").rstrip("=")

        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
        signature = hmac.new(secret_key.encode("utf-8"), signing_input, hashlib.sha256).digest()
        signature_b64 = base64.urlsafe_b64encode(signature).decode("utf-8").rstrip("=")

        return f"{header_b64}.{payload_b64}.{signature_b64}"

    @staticmethod
    def verify_access_token(token: str, secret_key: str = SECRET_KEY) -> Dict[str, Any]:
        """Verify JWT token signature and expiration, returning decoded payload."""
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid JWT token format.")

        header_b64, payload_b64, signature_b64 = parts

        # Verify signature
        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
        expected_sig = hmac.new(secret_key.encode("utf-8"), signing_input, hashlib.sha256).digest()
        
        # Restore padding for base64 decode
        rem = len(signature_b64) % 4
        padded_sig_b64 = signature_b64 + ("=" * (4 - rem) if rem else "")
        actual_sig = base64.urlsafe_b64decode(padded_sig_b64.encode("utf-8"))

        if not hmac.compare_digest(actual_sig, expected_sig):
            raise ValueError("Invalid JWT token signature.")

        # Decode payload
        rem_payload = len(payload_b64) % 4
        padded_payload_b64 = payload_b64 + ("=" * (4 - rem_payload) if rem_payload else "")
        payload_bytes = base64.urlsafe_b64decode(padded_payload_b64.encode("utf-8"))
        payload = json.loads(payload_bytes.decode("utf-8"))

        # Check expiration
        exp = payload.get("exp")
        if exp is not None:
            now_ts = int(datetime.now(timezone.utc).timestamp())
            if now_ts > exp:
                raise ValueError("JWT token has expired.")

        return payload

    @staticmethod
    def has_role_permission(user_role: str, required_role: Role) -> bool:
        """Check if user_role satisfies required_role in the hierarchy."""
        try:
            u_role_enum = Role(user_role.upper())
            user_level = ROLE_HIERARCHY.get(u_role_enum, 0)
            req_level = ROLE_HIERARCHY.get(required_role, 100)
            return user_level >= req_level
        except Exception:
            return False
