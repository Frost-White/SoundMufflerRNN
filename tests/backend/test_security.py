import uuid

from app.config import Settings
from app.services.security import (
    create_access_token,
    decode_token,
    generate_api_key_material,
    hash_api_key,
    hash_password,
    mask_api_key,
    verify_password,
)


def _settings() -> Settings:
    return Settings(jwt_secret="test-secret", jwt_algorithm="HS256", access_token_expire_minutes=5)


def test_password_hash_and_verify() -> None:
    hashed = hash_password("super-secret-pass")
    assert hashed != "super-secret-pass"
    assert verify_password("super-secret-pass", hashed) is True
    assert verify_password("wrong-pass", hashed) is False


def test_token_create_and_decode_round_trip() -> None:
    user_id = uuid.uuid4()
    token = create_access_token(user_id, _settings())
    decoded = decode_token(token, _settings())
    assert decoded == user_id


def test_decode_token_returns_none_for_invalid_token() -> None:
    assert decode_token("bad.token.value", _settings()) is None


def test_api_key_helpers() -> None:
    full_key, secret = generate_api_key_material()
    assert full_key.startswith("sk-live-")
    assert len(secret) == 32
    assert hash_api_key(full_key) == hash_api_key(full_key)
    assert mask_api_key("1234").endswith("1234")
