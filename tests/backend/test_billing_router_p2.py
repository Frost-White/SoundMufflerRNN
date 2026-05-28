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

import app.routers.billing as billing_mod  # noqa: E402
from app.schemas import PaymentMethodCreate, SubscriptionPatch  # noqa: E402


class _Scalars:
    def __init__(self, first=None, all_rows=None):
        self._first = first
        self._all_rows = all_rows or []

    def first(self):
        return self._first

    def all(self):
        return self._all_rows


class _Db:
    def __init__(self):
        self.sub = None
        self.plan = None
        self.pm_rows = []
        self.pm_get = None
        self.added = []
        self.commits = 0

    def scalars(self, _query):
        if self.pm_rows:
            return _Scalars(all_rows=self.pm_rows)
        return _Scalars(first=self.sub, all_rows=self.pm_rows)

    def get(self, model, _id):
        name = getattr(model, "__name__", "")
        if name == "SubscriptionPlan":
            return self.plan
        return self.pm_get

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.commits += 1

    def refresh(self, _obj):
        if getattr(_obj, "id", None) is None:
            _obj.id = uuid4()
        if getattr(_obj, "created_at", None) is None:
            _obj.created_at = datetime.now(UTC)

    def delete(self, _obj):
        return None

    def flush(self):
        return None


def test_get_subscription_returns_404_when_missing() -> None:
    db = _Db()
    user = SimpleNamespace(id=uuid4())
    with pytest.raises(HTTPException) as exc:
        billing_mod.get_subscription(db=db, user=user)
    assert exc.value.status_code == 404


def test_get_subscription_returns_500_when_plan_missing() -> None:
    db = _Db()
    db.sub = SimpleNamespace(
        user_id=uuid4(),
        plan_id="pro",
        status="active",
        current_period_start=datetime.now(UTC),
        current_period_end=datetime.now(UTC),
        cancel_at_period_end=False,
    )
    db.plan = None
    user = SimpleNamespace(id=uuid4())
    with pytest.raises(HTTPException) as exc:
        billing_mod.get_subscription(db=db, user=user)
    assert exc.value.status_code == 500


def test_patch_subscription_rejects_invalid_plan_id() -> None:
    db = _Db()
    db.sub = SimpleNamespace(
        user_id=uuid4(),
        plan_id="free",
        status="active",
        current_period_start=datetime.now(UTC),
        current_period_end=datetime.now(UTC),
        cancel_at_period_end=False,
    )
    db.plan = None
    user = SimpleNamespace(id=uuid4())
    body = SubscriptionPatch(plan_id="unknown")
    with pytest.raises(HTTPException) as exc:
        billing_mod.patch_subscription(body=body, db=db, user=user)
    assert exc.value.status_code == 400


def test_create_payment_method_first_card_becomes_default() -> None:
    db = _Db()
    db.pm_rows = []
    user = SimpleNamespace(id=uuid4())
    body = PaymentMethodCreate(brand="VISA", last4="1234", exp_month=12, exp_year=2030)
    out = billing_mod.create_payment_method(body=body, db=db, user=user)
    created = db.added[0]
    assert created.is_default is True
    assert created.brand == "visa"
    assert out.last4 == "1234"


def test_delete_payment_method_rejects_other_users_card() -> None:
    db = _Db()
    db.pm_get = SimpleNamespace(user_id=uuid4(), is_default=True)
    user = SimpleNamespace(id=uuid4())
    with pytest.raises(HTTPException) as exc:
        billing_mod.delete_payment_method(pm_id=uuid4(), db=db, user=user)
    assert exc.value.status_code == 404
