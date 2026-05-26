import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app import deps


class _Db:
    def __init__(self, user):
        self._user = user

    def get(self, _model, _id):
        return self._user


def test_get_current_user_requires_bearer() -> None:
    with pytest.raises(HTTPException) as exc:
        deps.get_current_user(db=_Db(user=None), settings=SimpleNamespace(), creds=None)
    assert exc.value.status_code == 401


def test_get_current_user_rejects_bad_token(monkeypatch) -> None:
    monkeypatch.setattr(deps, "decode_token", lambda *_: None)
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="bad")
    with pytest.raises(HTTPException) as exc:
        deps.get_current_user(db=_Db(user=None), settings=SimpleNamespace(), creds=creds)
    assert exc.value.status_code == 401


def test_get_current_user_returns_user(monkeypatch) -> None:
    user = SimpleNamespace(id=uuid.uuid4())
    monkeypatch.setattr(deps, "decode_token", lambda *_: user.id)
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="ok")
    got = deps.get_current_user(db=_Db(user=user), settings=SimpleNamespace(), creds=creds)
    assert got is user
