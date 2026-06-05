from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.models.subscription import UserSubscription


def resolve_user_plan_id(db: Session, user_id: UUID) -> str:
    sub = db.scalar(select(UserSubscription).where(UserSubscription.user_id == user_id))
    if sub is None or sub.status != "active":
        return "free"
    return "pro" if sub.plan_id == "pro" else "free"


def api_enhance_rate_limit(plan_id: str, settings: Settings) -> tuple[int, int]:
    if plan_id == "pro":
        return (
            settings.enhance_api_pro_rate_limit,
            settings.enhance_api_pro_rate_window_seconds,
        )
    return (
        settings.enhance_api_free_rate_limit,
        settings.enhance_api_free_rate_window_seconds,
    )
