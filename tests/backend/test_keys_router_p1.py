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

import app.routers.keys as keys_mod  # noqa: E402
from app.schemas import ApiKeyCreateRequest  # noqa: E402


class _Scalars:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _Db:
    def __init__(self, rows=None, key_row=None):
        self.rows = rows or []
        self.key_row = key_row
        self.added = []
        self.commits = 0

    def scalars(self, _query):
        return _Scalars(self.rows)

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.commits += 1

    def refresh(self, obj):
        if getattr(obj, "id", None) is None:
            obj.id = uuid4()
        if getattr(obj, "created_at", None) is None:
            obj.created_at = datetime.now(UTC)

    def get(self, _model, _key_id):
        return self.key_row


def test_list_keys_masks_secret_values() -> None:
    row = SimpleNamespace(id=uuid4(), name="prod", created_at=datetime.now(UTC), key_last_four="1234")
    db = _Db(rows=[row])
    user = SimpleNamespace(id=uuid4())
    out = keys_mod.list_keys(db=db, user=user)
    assert len(out) == 1
    assert out[0].masked.endswith("1234")
    assert out[0].name == "prod"


def test_create_key_persists_hashed_material(monkeypatch) -> None:
    monkeypatch.setattr(keys_mod, "generate_api_key_material", lambda: ("sk-live-abcdef1234567890", "abcdef1234567890"))
    monkeypatch.setattr(keys_mod, "hash_api_key", lambda _k: "hash")
    db = _Db()
    user = SimpleNamespace(id=uuid4())
    body = ApiKeyCreateRequest(name="  key name  ")
    out = keys_mod.create_key(body=body, db=db, user=user)
    saved = db.added[0]
    assert saved.name == "key name"
    assert saved.key_prefix == "sk-live-abcdef12"
    assert saved.key_last_four == "7890"
    assert saved.key_hash == "hash"
    assert out.key.startswith("sk-live-")


def test_revoke_key_returns_404_when_missing() -> None:
    db = _Db(key_row=None)
    user = SimpleNamespace(id=uuid4())
    with pytest.raises(HTTPException) as exc:
        keys_mod.revoke_key(key_id=uuid4(), db=db, user=user)
    assert exc.value.status_code == 404


def test_revoke_key_is_idempotent_when_already_revoked() -> None:
    row = SimpleNamespace(user_id=uuid4(), revoked_at=datetime.now(UTC))
    db = _Db(key_row=row)
    user = SimpleNamespace(id=row.user_id)
    out = keys_mod.revoke_key(key_id=uuid4(), db=db, user=user)
    assert out is None
    assert db.commits == 0


def test_revoke_key_sets_revoked_at_for_active_key() -> None:
    row = SimpleNamespace(user_id=uuid4(), revoked_at=None)
    db = _Db(key_row=row)
    user = SimpleNamespace(id=row.user_id)
    keys_mod.revoke_key(key_id=uuid4(), db=db, user=user)
    assert row.revoked_at is not None
    assert db.commits == 1
