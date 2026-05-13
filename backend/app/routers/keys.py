from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.deps import get_current_user
from app.db.session import get_db
from app.models.api_key import ApiKey
from app.models.user import User
from app.schemas import ApiKeyCreateRequest, ApiKeyCreateResponse, ApiKeyRowOut
from app.services.security import generate_api_key_material, hash_api_key, mask_api_key

router = APIRouter()


@router.get("", response_model=list[ApiKeyRowOut])
def list_keys(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[ApiKeyRowOut]:
    rows = db.scalars(
        select(ApiKey)
        .where(ApiKey.user_id == user.id, ApiKey.revoked_at.is_(None))
        .order_by(ApiKey.created_at.desc())
    ).all()
    out: list[ApiKeyRowOut] = []
    for r in rows:
        created = r.created_at.isoformat()
        out.append(
            ApiKeyRowOut(
                id=r.id,
                name=r.name,
                created=created,
                masked=mask_api_key(r.key_last_four),
            )
        )
    return out


@router.post("", response_model=ApiKeyCreateResponse, status_code=status.HTTP_201_CREATED)
def create_key(
    body: ApiKeyCreateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ApiKeyCreateResponse:
    full_key, secret = generate_api_key_material()
    key_hash = hash_api_key(full_key)
    prefix = "sk-live-" + secret[:8]
    last_four = secret[-4:]

    row = ApiKey(
        user_id=user.id,
        name=body.name.strip(),
        key_prefix=prefix,
        key_last_four=last_four,
        key_hash=key_hash,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return ApiKeyCreateResponse(id=row.id, name=row.name, created_at=row.created_at, key=full_key)


@router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_key(
    key_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> None:
    row = db.get(ApiKey, key_id)
    if row is None or row.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    if row.revoked_at is not None:
        return None

    row.revoked_at = datetime.now(UTC)
    db.add(row)
    db.commit()
