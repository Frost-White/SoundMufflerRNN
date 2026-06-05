import sys
from datetime import UTC, datetime, timedelta
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

    def get(self, model, key):
        name = getattr(model, "__name__", "")
        if name == "SubscriptionPlan":
            if key == "free":
                return SimpleNamespace(
                    id="free",
                    display_name="Free",
                    price_cents=0,
                    billing_interval="month",
                )
            if key == "pro":
                return SimpleNamespace(
                    id="pro",
                    display_name="Pro",
                    price_cents=1000,
                    billing_interval="month",
                )
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
        plan_id="enterprise",
        scheduled_plan_id=None,
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


def test_patch_subscription_schedules_downgrade_at_period_end() -> None:
    db = _Db()
    period_end = datetime.now(UTC) + timedelta(days=12)
    db.sub = SimpleNamespace(
        user_id=uuid4(),
        plan_id="pro",
        scheduled_plan_id=None,
        status="active",
        current_period_start=datetime.now(UTC),
        current_period_end=period_end,
        cancel_at_period_end=False,
    )
    db.plan = SimpleNamespace(
        id="pro",
        display_name="Pro",
        price_cents=1000,
        billing_interval="month",
    )
    user = SimpleNamespace(id=uuid4())
    out = billing_mod.patch_subscription(
        body=SubscriptionPatch(plan_id="free"),
        db=db,
        user=user,
    )
    assert db.sub.plan_id == "pro"
    assert db.sub.scheduled_plan_id == "free"
    assert db.sub.current_period_end == period_end
    assert out.scheduled_plan_id == "free"
    assert out.scheduled_plan_display_name == "Free"


def test_apply_pending_period_changes_applies_scheduled_downgrade() -> None:
    past_end = datetime.now(UTC) - timedelta(days=1)
    sub = SimpleNamespace(
        plan_id="pro",
        scheduled_plan_id="free",
        status="active",
        current_period_start=datetime.now(UTC) - timedelta(days=31),
        current_period_end=past_end,
        cancel_at_period_end=False,
    )
    billing_mod._apply_pending_period_changes(sub)  # pylint: disable=protected-access
    assert sub.plan_id == "free"
    assert sub.scheduled_plan_id is None


def test_patch_subscription_rejects_cancel_on_free_plan() -> None:
    db = _Db()
    db.sub = SimpleNamespace(
        user_id=uuid4(),
        plan_id="free",
        scheduled_plan_id=None,
        status="active",
        current_period_start=datetime.now(UTC),
        current_period_end=datetime.now(UTC),
        cancel_at_period_end=False,
    )
    db.plan = SimpleNamespace(
        id="free",
        display_name="Free",
        price_cents=0,
        billing_interval="month",
    )
    user = SimpleNamespace(id=uuid4())
    body = SubscriptionPatch(cancel_at_period_end=True)
    with pytest.raises(HTTPException) as exc:
        billing_mod.patch_subscription(body=body, db=db, user=user)
    assert exc.value.status_code == 400
    assert exc.value.detail == "Free plan cannot be canceled"


def test_patch_subscription_upgrades_free_to_pro_immediately() -> None:
    db = _Db()
    db.sub = SimpleNamespace(
        user_id=uuid4(),
        plan_id="free",
        scheduled_plan_id=None,
        status="active",
        current_period_start=datetime.now(UTC) - timedelta(days=10),
        current_period_end=datetime.now(UTC) + timedelta(days=20),
        cancel_at_period_end=False,
    )
    user = SimpleNamespace(id=uuid4())
    out = billing_mod.patch_subscription(
        body=SubscriptionPatch(plan_id="pro"),
        db=db,
        user=user,
    )
    assert db.sub.plan_id == "pro"
    assert db.sub.scheduled_plan_id is None
    assert db.sub.cancel_at_period_end is False
    assert out.plan_id == "pro"


def test_patch_subscription_clears_scheduled_downgrade_when_reselecting_pro() -> None:
    db = _Db()
    period_end = datetime.now(UTC) + timedelta(days=12)
    db.sub = SimpleNamespace(
        user_id=uuid4(),
        plan_id="pro",
        scheduled_plan_id="free",
        status="active",
        current_period_start=datetime.now(UTC),
        current_period_end=period_end,
        cancel_at_period_end=False,
    )
    user = SimpleNamespace(id=uuid4())
    out = billing_mod.patch_subscription(
        body=SubscriptionPatch(plan_id="pro"),
        db=db,
        user=user,
    )
    assert db.sub.plan_id == "pro"
    assert db.sub.scheduled_plan_id is None
    assert out.scheduled_plan_id is None


def test_patch_subscription_resume_clears_cancel_at_period_end() -> None:
    db = _Db()
    db.sub = SimpleNamespace(
        user_id=uuid4(),
        plan_id="pro",
        scheduled_plan_id=None,
        status="active",
        current_period_start=datetime.now(UTC),
        current_period_end=datetime.now(UTC) + timedelta(days=12),
        cancel_at_period_end=True,
    )
    user = SimpleNamespace(id=uuid4())
    out = billing_mod.patch_subscription(
        body=SubscriptionPatch(cancel_at_period_end=False),
        db=db,
        user=user,
    )
    assert db.sub.cancel_at_period_end is False
    assert db.sub.plan_id == "pro"
    assert out.cancel_at_period_end is False


def test_patch_subscription_clears_cancel_when_reselecting_pro() -> None:
    db = _Db()
    db.sub = SimpleNamespace(
        user_id=uuid4(),
        plan_id="pro",
        scheduled_plan_id=None,
        status="active",
        current_period_start=datetime.now(UTC),
        current_period_end=datetime.now(UTC) + timedelta(days=12),
        cancel_at_period_end=True,
    )
    user = SimpleNamespace(id=uuid4())
    billing_mod.patch_subscription(
        body=SubscriptionPatch(plan_id="pro"),
        db=db,
        user=user,
    )
    assert db.sub.cancel_at_period_end is False


def test_patch_subscription_rejects_invalid_plan_id() -> None:
    db = _Db()
    db.sub = SimpleNamespace(
        user_id=uuid4(),
        plan_id="free",
        scheduled_plan_id=None,
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
