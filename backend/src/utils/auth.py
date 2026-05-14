"""
LexVed Authentication Module
JWT-based auth with role-based access control (user vs admin).
"""

import os
import json
import hashlib
import hmac
import time
import base64
from typing import Optional
from fastapi import HTTPException, Request

# ─── Configuration ────────────────────────────────────────────────

AUTH_SECRET = os.getenv("LEXVED_AUTH_SECRET", "lexved-prestige-auth-secret-2025-xK9mP2")
TOKEN_EXPIRY = 86400  # 24 hours
USERS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "users.json")

# ─── Password Hashing (HMAC-SHA256, no external deps) ────────────

def hash_password(password: str) -> str:
    """Hash a password using HMAC-SHA256 with the auth secret."""
    return hmac.new(
        AUTH_SECRET.encode(),
        password.encode(),
        hashlib.sha256
    ).hexdigest()

def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash."""
    return hmac.compare_digest(hash_password(password), password_hash)

# ─── JWT Implementation (Minimal, no PyJWT dependency) ────────────

def _b64_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

def _b64_decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    if padding != 4:
        s += "=" * padding
    return base64.urlsafe_b64decode(s)

def create_token(username: str, role: str) -> str:
    """Create a signed JWT token with username, role, and expiry."""
    header = _b64_encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload_data = {
        "sub": username,
        "role": role,
        "exp": int(time.time()) + TOKEN_EXPIRY,
        "iat": int(time.time())
    }
    payload = _b64_encode(json.dumps(payload_data).encode())
    signature_input = f"{header}.{payload}"
    signature = _b64_encode(
        hmac.new(AUTH_SECRET.encode(), signature_input.encode(), hashlib.sha256).digest()
    )
    return f"{header}.{payload}.{signature}"

def verify_token(token: str) -> Optional[dict]:
    """Verify and decode a JWT token. Returns payload dict or None."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None

        header_b64, payload_b64, signature_b64 = parts

        # Verify signature
        expected_sig = _b64_encode(
            hmac.new(
                AUTH_SECRET.encode(),
                f"{header_b64}.{payload_b64}".encode(),
                hashlib.sha256
            ).digest()
        )
        if not hmac.compare_digest(expected_sig, signature_b64):
            return None

        # Decode payload
        payload = json.loads(_b64_decode(payload_b64))

        # Check expiry
        if payload.get("exp", 0) < time.time():
            return None

        return payload
    except Exception:
        return None

# ─── User Management ─────────────────────────────────────────────

def _load_users() -> list:
    """Load users from users.json."""
    if os.path.exists(USERS_PATH):
        try:
            with open(USERS_PATH, "r") as f:
                data = json.load(f)
            return data.get("users", [])
        except Exception:
            pass
    return []

def _save_users(users: list):
    """Save users to users.json."""
    with open(USERS_PATH, "w") as f:
        json.dump({"users": users}, f, indent=2)

def authenticate_user(username: str, password: str) -> Optional[dict]:
    """Authenticate a user by username/password. Returns user dict or None."""
    users = _load_users()
    for user in users:
        if user["username"] == username and verify_password(password, user["password_hash"]):
            return {
                "username": user["username"],
                "role": user["role"],
                "display_name": user.get("display_name", username)
            }
    return None

def get_all_users() -> list:
    """Return all users (without password hashes) for admin view."""
    users = _load_users()
    return [
        {
            "username": u["username"],
            "role": u["role"],
            "display_name": u.get("display_name", u["username"])
        }
        for u in users
    ]

def initialize_users():
    """Create default users if users.json does not exist."""
    if not os.path.exists(USERS_PATH):
        default_users = {
            "users": [
                {
                    "username": "user",
                    "password_hash": hash_password("lexved2025"),
                    "role": "user",
                    "display_name": "Legal Researcher"
                },
                {
                    "username": "admin",
                    "password_hash": hash_password("lexved@admin"),
                    "role": "admin",
                    "display_name": "System Administrator"
                }
            ]
        }
        with open(USERS_PATH, "w") as f:
            json.dump(default_users, f, indent=2)
        print("[LexVed Auth] Default users created (user/admin)")

# ─── FastAPI Dependencies ─────────────────────────────────────────

async def get_current_user(request: Request) -> dict:
    """FastAPI dependency: Extract and validate the current user from the Authorization header."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required")

    token = auth_header[7:]
    payload = verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return {"username": payload["sub"], "role": payload["role"]}

async def require_admin(request: Request) -> dict:
    """FastAPI dependency: Require admin role."""
    user = await get_current_user(request)
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Administrator access required")
    return user
