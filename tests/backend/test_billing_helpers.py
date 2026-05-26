from types import SimpleNamespace

from app.routers.billing import _format_expiry, _price_to_usd_month


def test_price_to_usd_month_handles_none() -> None:
    plan = SimpleNamespace(price_cents=None)
    assert _price_to_usd_month(plan) is None


def test_price_to_usd_month_rounds() -> None:
    plan = SimpleNamespace(price_cents=1299)
    assert _price_to_usd_month(plan) == 12.99


def test_format_expiry() -> None:
    pm = SimpleNamespace(exp_month=3, exp_year=2030)
    assert _format_expiry(pm) == "03/30"
