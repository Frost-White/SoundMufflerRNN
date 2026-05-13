from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


class UserOut(BaseModel):
    id: UUID
    email: str
    full_name: str
    email_verified: bool

    model_config = {"from_attributes": True}


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=10, max_length=128)
    full_name: str = Field(min_length=2, max_length=200)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class SubscriptionOut(BaseModel):
    plan_id: str
    plan_display_name: str
    billing_cycle: str
    price_usd_per_month: float | None
    status: str
    current_period_start: datetime
    current_period_end: datetime
    cancel_at_period_end: bool


class SubscriptionPatch(BaseModel):
    plan_id: str | None = None
    cancel_at_period_end: bool | None = None


class PaymentMethodOut(BaseModel):
    id: UUID
    brand: str
    last4: str
    expiry: str
    is_default: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class PaymentMethodCreate(BaseModel):
    brand: str = Field(min_length=2, max_length=32)
    last4: str = Field(min_length=4, max_length=4, pattern=r"^\d{4}$")
    exp_month: int = Field(ge=1, le=12)
    exp_year: int = Field(ge=2000, le=2100)


class ApiKeyRowOut(BaseModel):
    id: UUID
    name: str
    created: str
    masked: str


class ApiKeyCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class ApiKeyCreateResponse(BaseModel):
    id: UUID
    name: str
    created_at: datetime
    key: str
