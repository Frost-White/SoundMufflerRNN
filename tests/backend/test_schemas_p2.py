import sys
from pathlib import Path

import pytest
from pydantic import ValidationError


ROOT = Path(__file__).resolve().parents[2]
BACKEND_PATH = ROOT / "backend"
if str(BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PATH))

from app.schemas import (  # noqa: E402
    ApiKeyCreateRequest,
    LoginRequest,
    PaymentMethodCreate,
    RegisterRequest,
)


def test_register_request_rejects_short_password() -> None:
    with pytest.raises(ValidationError):
        RegisterRequest(email="x@example.com", password="short", full_name="User Name")


def test_login_request_rejects_invalid_email() -> None:
    with pytest.raises(ValidationError):
        LoginRequest(email="not-email", password="x")


def test_payment_method_create_rejects_non_digit_last4() -> None:
    with pytest.raises(ValidationError):
        PaymentMethodCreate(brand="visa", last4="12a4", exp_month=1, exp_year=2030)


def test_payment_method_create_rejects_invalid_month() -> None:
    with pytest.raises(ValidationError):
        PaymentMethodCreate(brand="visa", last4="1234", exp_month=13, exp_year=2030)


def test_api_key_create_request_rejects_empty_name() -> None:
    with pytest.raises(ValidationError):
        ApiKeyCreateRequest(name="")


def test_valid_payment_method_create_passes() -> None:
    pm = PaymentMethodCreate(brand="visa", last4="1234", exp_month=12, exp_year=2030)
    assert pm.last4 == "1234"
