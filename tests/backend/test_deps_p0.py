import sys
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials


ROOT = Path(__file__).resolve().parents[2]
BACKEND_PATH = ROOT / "backend"
if str(BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PATH))

import app.deps as deps_mod  # noqa: E402
from app.config import Settings  # noqa: E402


class DummyDb:
    def __init__(self, user=None, api_key_row=None):
        self._user = user
        self._api_key_row = api_key_row

    def get(self, _model, _pk):
        return self._user

    def scalar(self, _query):
        return self._api_key_row


def test_get_current_user_rejects_missing_credentials() -> None:
    with pytest.raises(HTTPException) as exc:
        deps_mod.get_current_user(db=DummyDb(), settings=Settings(), creds=None)
    assert exc.value.status_code == 401


def test_get_current_user_rejects_non_bearer_scheme() -> None:
    creds = HTTPAuthorizationCredentials(scheme="Basic", credentials="token")
    with pytest.raises(HTTPException) as exc:
        deps_mod.get_current_user(db=DummyDb(), settings=Settings(), creds=creds)
    assert exc.value.status_code == 401


def test_get_current_user_rejects_when_token_decoding_fails(monkeypatch) -> None:
    monkeypatch.setattr(deps_mod, "decode_token", lambda _token, _settings: None)
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="token")
    with pytest.raises(HTTPException) as exc:
        deps_mod.get_current_user(db=DummyDb(), settings=Settings(), creds=creds)
    assert exc.value.status_code == 401


def test_get_current_user_rejects_when_user_missing(monkeypatch) -> None:
    monkeypatch.setattr(deps_mod, "decode_token", lambda _token, _settings: uuid4())
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="token")
    with pytest.raises(HTTPException) as exc:
        deps_mod.get_current_user(db=DummyDb(user=None), settings=Settings(), creds=creds)
    assert exc.value.status_code == 401


def test_get_active_api_key_rejects_missing_header() -> None:
    with pytest.raises(HTTPException) as exc:
        deps_mod.get_active_api_key(db=DummyDb(api_key_row=object()), x_api_key=None)
    assert exc.value.status_code == 401


def test_get_active_api_key_rejects_invalid_key() -> None:
    with pytest.raises(HTTPException) as exc:
        deps_mod.get_active_api_key(db=DummyDb(api_key_row=None), x_api_key="sk-live-bad")
    assert exc.value.status_code == 403


def test_get_active_api_key_accepts_valid_key_and_trims_whitespace() -> None:
    row = object()
    out = deps_mod.get_active_api_key(db=DummyDb(api_key_row=row), x_api_key="  sk-live-ok  ")
    assert out is row
