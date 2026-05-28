import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException


ROOT = Path(__file__).resolve().parents[2]
BACKEND_PATH = ROOT / "backend"
if str(BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PATH))

import app.routers.auth as auth_mod  # noqa: E402
from app.config import Settings  # noqa: E402
from app.schemas import LoginRequest, RegisterRequest  # noqa: E402


class _ExecuteResult:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class _Db:
    def __init__(self, existing_user=None):
        self.existing_user = existing_user
        self.added = []
        self.commits = 0
        self.refreshed = []

    def execute(self, _query):
        return _ExecuteResult(self.existing_user)

    def add(self, obj):
        self.added.append(obj)

    def flush(self):
        if self.added and getattr(self.added[0], "id", None) is None:
            self.added[0].id = uuid4()

    def commit(self):
        self.commits += 1

    def refresh(self, obj):
        self.refreshed.append(obj)


def test_user_out_sets_email_verified_flag() -> None:
    user = SimpleNamespace(
        id=uuid4(),
        email="a@b.com",
        full_name="A",
        email_verified_at=datetime.now(UTC),
    )
    out = auth_mod._user_out(user)  # pylint: disable=protected-access
    assert out.email_verified is True


def test_register_rejects_duplicate_email() -> None:
    db = _Db(existing_user=SimpleNamespace(id=uuid4()))
    body = RegisterRequest(email="x@example.com", password="1234567890", full_name="X Y")
    with pytest.raises(HTTPException) as exc:
        auth_mod.register(body=body, db=db, settings=Settings())
    assert exc.value.status_code == 409


def test_register_normalizes_email_and_strips_name(monkeypatch) -> None:
    monkeypatch.setattr(auth_mod, "hash_password", lambda _p: "hashed")
    monkeypatch.setattr(auth_mod, "create_access_token", lambda _uid, _s: "token")
    db = _Db(existing_user=None)
    body = RegisterRequest(email="  TeSt@Example.com  ", password="1234567890", full_name="  Full Name  ")

    out = auth_mod.register(body=body, db=db, settings=Settings())

    user = db.added[0]
    sub = db.added[1]
    assert user.email == "test@example.com"
    assert user.full_name == "Full Name"
    assert user.password_hash == "hashed"
    assert sub.plan_id == "free"
    assert out.access_token == "token"
    assert db.commits == 1


def test_login_rejects_when_user_missing() -> None:
    db = _Db(existing_user=None)
    body = LoginRequest(email="x@example.com", password="123")
    with pytest.raises(HTTPException) as exc:
        auth_mod.login(body=body, db=db, settings=Settings())
    assert exc.value.status_code == 401


def test_login_rejects_when_password_invalid(monkeypatch) -> None:
    monkeypatch.setattr(auth_mod, "verify_password", lambda _p, _h: False)
    user = SimpleNamespace(id=uuid4(), email="x@example.com", full_name="X", password_hash="h", email_verified_at=None)
    db = _Db(existing_user=user)
    body = LoginRequest(email="x@example.com", password="bad")
    with pytest.raises(HTTPException) as exc:
        auth_mod.login(body=body, db=db, settings=Settings())
    assert exc.value.status_code == 401


def test_login_returns_token_on_success(monkeypatch) -> None:
    monkeypatch.setattr(auth_mod, "verify_password", lambda _p, _h: True)
    monkeypatch.setattr(auth_mod, "create_access_token", lambda _uid, _s: "ok-token")
    user = SimpleNamespace(id=uuid4(), email="x@example.com", full_name="X", password_hash="h", email_verified_at=None)
    db = _Db(existing_user=user)
    body = LoginRequest(email=" x@example.com ", password="good")
    out = auth_mod.login(body=body, db=db, settings=Settings())
    assert out.access_token == "ok-token"
    assert out.user.email == "x@example.com"


def test_logout_returns_ok() -> None:
    assert auth_mod.logout() == {"ok": True}


def test_me_returns_user_out() -> None:
    user = SimpleNamespace(id=uuid4(), email="x@example.com", full_name="X", email_verified_at=None)
    out = auth_mod.me(user=user)
    assert out.email == "x@example.com"
