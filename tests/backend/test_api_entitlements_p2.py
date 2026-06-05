import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest


ROOT = Path(__file__).resolve().parents[2]
BACKEND_PATH = ROOT / "backend"
if str(BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PATH))

from app.config import Settings  # noqa: E402
from app.services import api_entitlements as ent_mod  # noqa: E402


class _Db:
    def __init__(self, sub):
        self._sub = sub

    def scalar(self, _query):
        return self._sub


def test_api_enhance_rate_limit_free_defaults() -> None:
    s = Settings()
    limit, window = ent_mod.api_enhance_rate_limit("free", s)
    assert limit == 30
    assert window == 900


def test_api_enhance_rate_limit_pro_defaults() -> None:
    s = Settings()
    limit, window = ent_mod.api_enhance_rate_limit("pro", s)
    assert limit == 15
    assert window == 60


def test_resolve_user_plan_id_returns_free_without_subscription() -> None:
    db = _Db(None)
    assert ent_mod.resolve_user_plan_id(db, uuid4()) == "free"


def test_resolve_user_plan_id_returns_pro_for_active_pro() -> None:
    sub = SimpleNamespace(plan_id="pro", status="active")
    db = _Db(sub)
    assert ent_mod.resolve_user_plan_id(db, uuid4()) == "pro"


def test_resolve_user_plan_id_returns_free_for_inactive() -> None:
    sub = SimpleNamespace(plan_id="pro", status="canceled")
    db = _Db(sub)
    assert ent_mod.resolve_user_plan_id(db, uuid4()) == "free"


def test_resolve_user_plan_id_treats_unknown_plan_as_free() -> None:
    sub = SimpleNamespace(plan_id="enterprise", status="active")
    db = _Db(sub)
    assert ent_mod.resolve_user_plan_id(db, uuid4()) == "free"
