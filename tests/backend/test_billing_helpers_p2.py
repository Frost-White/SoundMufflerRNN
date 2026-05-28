import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[2]
BACKEND_PATH = ROOT / "backend"
if str(BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PATH))

import app.routers.billing as billing_mod  # noqa: E402


def test_price_to_usd_month_returns_none_when_missing_price() -> None:
    plan = SimpleNamespace(price_cents=None)
    assert billing_mod._price_to_usd_month(plan) is None  # pylint: disable=protected-access


def test_price_to_usd_month_rounds_two_decimals() -> None:
    plan = SimpleNamespace(price_cents=999)
    assert billing_mod._price_to_usd_month(plan) == 9.99  # pylint: disable=protected-access


def test_format_expiry_zero_pads_month_and_uses_short_year() -> None:
    pm = SimpleNamespace(exp_month=3, exp_year=2031)
    assert billing_mod._format_expiry(pm) == "03/31"  # pylint: disable=protected-access


def test_set_default_payment_method_returns_404_for_other_user() -> None:
    pm_id = uuid4()
    user = SimpleNamespace(id=uuid4())
    other_pm = SimpleNamespace(id=pm_id, user_id=uuid4())

    class _Db:
        def get(self, _model, _id):
            return other_pm

    import pytest
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        billing_mod.set_default_payment_method(pm_id=pm_id, db=_Db(), user=user)
    assert exc.value.status_code == 404
