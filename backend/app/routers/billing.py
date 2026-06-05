from datetime import UTC, datetime, timedelta
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.deps import get_current_user
from app.db.session import get_db
from app.models.payment_method import PaymentMethod
from app.models.subscription import SubscriptionPlan, UserSubscription
from app.models.user import User
from app.schemas import (
    PaymentMethodCreate,
    PaymentMethodOut,
    SubscriptionOut,
    SubscriptionPatch,
)

router = APIRouter()

_PLAN_TIER = {"free": 0, "pro": 1}


def _plan_tier(plan_id: str) -> int:
    return _PLAN_TIER.get(plan_id, 0)


def _is_downgrade(from_plan: str, to_plan: str) -> bool:
    return _plan_tier(to_plan) < _plan_tier(from_plan)


def _price_to_usd_month(plan: SubscriptionPlan) -> float | None:
    if plan.price_cents is None:
        return None
    return round(plan.price_cents / 100.0, 2)


def _format_expiry(pm: PaymentMethod) -> str:
    return f"{pm.exp_month:02d}/{str(pm.exp_year)[-2:]}"


def _advance_free_period(sub: UserSubscription) -> None:
    sub.current_period_start = sub.current_period_end
    sub.current_period_end = sub.current_period_end + timedelta(days=30)


def _apply_pending_period_changes(sub: UserSubscription) -> None:
    now = datetime.now(UTC)
    if now < sub.current_period_end:
        return

    if sub.scheduled_plan_id:
        sub.plan_id = sub.scheduled_plan_id
        sub.scheduled_plan_id = None
        sub.cancel_at_period_end = False
        sub.status = "active"
        if sub.plan_id == "free":
            _advance_free_period(sub)
        else:
            sub.current_period_start = now
            sub.current_period_end = now + timedelta(days=30)
        return

    if sub.cancel_at_period_end and sub.plan_id == "pro":
        sub.plan_id = "free"
        sub.cancel_at_period_end = False
        sub.scheduled_plan_id = None
        sub.status = "active"
        _advance_free_period(sub)


def _subscription_out(sub: UserSubscription, db: Session) -> SubscriptionOut:
    plan = db.get(SubscriptionPlan, sub.plan_id)
    if plan is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Plan missing")

    scheduled_name: str | None = None
    if sub.scheduled_plan_id:
        scheduled_plan = db.get(SubscriptionPlan, sub.scheduled_plan_id)
        scheduled_name = scheduled_plan.display_name if scheduled_plan else sub.scheduled_plan_id

    return SubscriptionOut(
        plan_id=plan.id,
        plan_display_name=plan.display_name,
        billing_cycle="Monthly" if plan.billing_interval == "month" else plan.billing_interval,
        price_usd_per_month=_price_to_usd_month(plan),
        status=sub.status,
        current_period_start=sub.current_period_start,
        current_period_end=sub.current_period_end,
        cancel_at_period_end=sub.cancel_at_period_end,
        scheduled_plan_id=sub.scheduled_plan_id,
        scheduled_plan_display_name=scheduled_name,
    )


def _payment_methods_for_user(db: Session, user: User) -> list[PaymentMethodOut]:
    rows = db.scalars(
        select(PaymentMethod)
        .where(PaymentMethod.user_id == user.id)
        .order_by(PaymentMethod.created_at.desc())
    ).all()
    return [
        PaymentMethodOut(
            id=r.id,
            brand=r.brand,
            last4=r.last4,
            expiry=_format_expiry(r),
            is_default=r.is_default,
            created_at=r.created_at,
        )
        for r in rows
    ]


@router.get("/subscription", response_model=SubscriptionOut)
def get_subscription(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> SubscriptionOut:
    sub = db.scalars(
        select(UserSubscription).where(UserSubscription.user_id == user.id)
    ).first()
    if sub is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No subscription")
    _apply_pending_period_changes(sub)
    db.add(sub)
    db.commit()
    db.refresh(sub)
    return _subscription_out(sub, db)


@router.patch("/subscription", response_model=SubscriptionOut)
def patch_subscription(
    body: SubscriptionPatch,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> SubscriptionOut:
    sub = db.scalars(
        select(UserSubscription).where(UserSubscription.user_id == user.id)
    ).first()
    if sub is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No subscription")

    _apply_pending_period_changes(sub)

    if body.plan_id is not None:
        plan = db.get(SubscriptionPlan, body.plan_id)
        if plan is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid plan_id")

        if body.plan_id == sub.plan_id:
            sub.scheduled_plan_id = None
            if body.plan_id == "pro":
                sub.cancel_at_period_end = False
        elif _is_downgrade(sub.plan_id, body.plan_id):
            sub.scheduled_plan_id = body.plan_id
            sub.cancel_at_period_end = False
        else:
            sub.plan_id = body.plan_id
            sub.scheduled_plan_id = None
            sub.cancel_at_period_end = False
            sub.status = "active"
            sub.current_period_start = datetime.now(UTC)
            sub.current_period_end = datetime.now(UTC) + timedelta(days=30)

    if body.cancel_at_period_end is not None:
        if body.cancel_at_period_end and sub.plan_id == "free":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Free plan cannot be canceled",
            )
        if body.cancel_at_period_end:
            sub.scheduled_plan_id = None
        sub.cancel_at_period_end = body.cancel_at_period_end

    db.add(sub)
    db.commit()
    db.refresh(sub)
    return _subscription_out(sub, db)


@router.get("/payment-methods", response_model=list[PaymentMethodOut])
def list_payment_methods(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[PaymentMethodOut]:
    return _payment_methods_for_user(db, user)


@router.post("/payment-methods", response_model=PaymentMethodOut, status_code=status.HTTP_201_CREATED)
def create_payment_method(
    body: PaymentMethodCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> PaymentMethodOut:
    existing = db.scalars(
        select(PaymentMethod).where(PaymentMethod.user_id == user.id)
    ).all()
    is_default = len(existing) == 0

    pm = PaymentMethod(
        user_id=user.id,
        is_default=is_default,
        brand=body.brand.lower().strip(),
        last4=body.last4,
        exp_month=body.exp_month,
        exp_year=body.exp_year,
    )
    db.add(pm)
    db.commit()
    db.refresh(pm)
    return PaymentMethodOut(
        id=pm.id,
        brand=pm.brand,
        last4=pm.last4,
        expiry=_format_expiry(pm),
        is_default=pm.is_default,
        created_at=pm.created_at,
    )


@router.patch("/payment-methods/{pm_id}/default", response_model=list[PaymentMethodOut])
def set_default_payment_method(
    pm_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[PaymentMethodOut]:
    target = db.get(PaymentMethod, pm_id)
    if target is None or target.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")

    all_pm = db.scalars(select(PaymentMethod).where(PaymentMethod.user_id == user.id)).all()
    for p in all_pm:
        p.is_default = p.id == target.id
    db.commit()
    return _payment_methods_for_user(db, user)


@router.delete("/payment-methods/{pm_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_payment_method(
    pm_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> None:
    pm = db.get(PaymentMethod, pm_id)
    if pm is None or pm.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")

    was_default = pm.is_default
    db.delete(pm)
    db.flush()

    if was_default:
        rest = db.scalars(
            select(PaymentMethod)
            .where(PaymentMethod.user_id == user.id)
            .order_by(PaymentMethod.created_at.asc())
        ).all()
        if rest:
            rest[0].is_default = True
            db.add(rest[0])
    db.commit()
