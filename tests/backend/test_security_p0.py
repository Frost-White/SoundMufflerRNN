import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import jwt


ROOT = Path(__file__).resolve().parents[2]
BACKEND_PATH = ROOT / "backend"
if str(BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PATH))

from app.config import Settings  # noqa: E402
from app.services.security import decode_token, verify_password  # noqa: E402


def test_verify_password_returns_false_for_malformed_hash() -> None:
    assert verify_password("secret", "not-a-bcrypt-hash") is False


def test_decode_token_returns_none_when_sub_missing() -> None:
    settings = Settings(jwt_secret="test-secret-key-with-32-plus-bytes", jwt_algorithm="HS256")
    token = jwt.encode({"exp": datetime.now(UTC) + timedelta(minutes=5)}, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    assert decode_token(token, settings) is None


def test_decode_token_returns_none_for_expired_token() -> None:
    settings = Settings(jwt_secret="test-secret-key-with-32-plus-bytes", jwt_algorithm="HS256")
    token = jwt.encode(
        {"sub": str(uuid4()), "exp": datetime.now(UTC) - timedelta(minutes=1)},
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )
    assert decode_token(token, settings) is None
