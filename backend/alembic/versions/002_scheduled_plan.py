"""scheduled plan at period end

Revision ID: 002_scheduled_plan
Revises: 001_initial
Create Date: 2026-06-04

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "002_scheduled_plan"
down_revision: Union[str, None] = "001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "user_subscriptions",
        sa.Column("scheduled_plan_id", sa.String(length=32), nullable=True),
    )
    op.create_foreign_key(
        "fk_user_subscriptions_scheduled_plan_id",
        "user_subscriptions",
        "subscription_plans",
        ["scheduled_plan_id"],
        ["id"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_user_subscriptions_scheduled_plan_id",
        "user_subscriptions",
        type_="foreignkey",
    )
    op.drop_column("user_subscriptions", "scheduled_plan_id")
