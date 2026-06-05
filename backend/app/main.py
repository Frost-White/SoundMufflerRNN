import logging
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.session import get_db
from app.deps import get_active_api_key, get_current_user
from app.inference import InferenceError, enhance_audio_bytes
from app.models.api_key import ApiKey
from app.models.user import User
from app.routers import auth, billing, keys
from app.services.api_entitlements import api_enhance_rate_limit, resolve_user_plan_id
from app.services.rate_limit import rate_limiter

settings = get_settings()
app = FastAPI(title="Sound Muffler API")
logger = logging.getLogger("uvicorn.error")

_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(billing.router, prefix="/api/billing", tags=["billing"])
app.include_router(keys.router, prefix="/api/keys", tags=["keys"])


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "Sound Muffler API", "docs": "/docs", "health": "/api/health"}


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _max_upload_bytes(settings: Settings) -> int:
    return settings.enhance_max_upload_bytes


def _reject_oversized_upload(size: int | None, max_bytes: int) -> None:
    if size is not None and size > max_bytes:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum upload size is 2 MB.",
        )


async def _read_upload_capped(file: UploadFile, max_bytes: int) -> bytes:
    _reject_oversized_upload(file.size, max_bytes)
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail="File too large. Maximum upload size is 2 MB.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _enforce_rate_limit(key: str, limit: int, window_seconds: int) -> None:
    allowed, retry_after = rate_limiter.check(key=key, limit=limit, window_seconds=window_seconds)
    if allowed:
        return
    raise HTTPException(
        status_code=429,
        detail="Rate limit exceeded.",
        headers={"Retry-After": str(retry_after)},
    )


async def _run_enhance(
    file: UploadFile,
    route_kind: str,
    req_settings: Settings,
) -> Response:
    max_bytes = _max_upload_bytes(req_settings)
    content = await _read_upload_capped(file, max_bytes)
    if not content:
        logger.warning("Enhance request received with empty file route=%s", route_kind)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    filename = file.filename or "audio.bin"
    try:
        enhanced_bytes = enhance_audio_bytes(content)
    except FileNotFoundError as exc:
        logger.exception("Enhance weights missing: %s", exc)
        raise HTTPException(status_code=500, detail="Model weights are not available.") from exc
    except InferenceError as exc:
        logger.warning(
            "Enhance request rejected: route=%s filename=%s reason=%s",
            route_kind,
            filename,
            exc,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Enhance failed: route=%s filename=%s", route_kind, filename)
        raise HTTPException(status_code=500, detail="Audio processing failed.") from exc

    logger.info(
        "Enhance request completed: route=%s filename=%s input_bytes=%s output_bytes=%s status=200 device=cpu",
        route_kind,
        filename,
        len(content),
        len(enhanced_bytes),
    )
    print(
        f'ENHANCE 200 route="{route_kind}" filename="{filename}" in={len(content)} out={len(enhanced_bytes)} device=cpu',
        flush=True,
    )
    out_name = f"processed-{Path(filename).stem}.wav"
    return Response(
        content=enhanced_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": f'inline; filename="{out_name}"'},
    )


@app.post("/enhance")
async def enhance_api(
    file: UploadFile = File(...),
    api_key: ApiKey = Depends(get_active_api_key),
    db: Session = Depends(get_db),
    req_settings: Settings = Depends(get_settings),
) -> Response:
    plan_id = resolve_user_plan_id(db, api_key.user_id)
    limit, window_seconds = api_enhance_rate_limit(plan_id, req_settings)
    _enforce_rate_limit(
        key=f"enhance:api:{api_key.id}",
        limit=limit,
        window_seconds=window_seconds,
    )
    return await _run_enhance(file=file, route_kind="api", req_settings=req_settings)


@app.post("/enhance/web")
async def enhance_web(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    req_settings: Settings = Depends(get_settings),
) -> Response:
    _enforce_rate_limit(
        key=f"enhance:web:{user.id}",
        limit=req_settings.enhance_web_rate_limit,
        window_seconds=req_settings.enhance_web_rate_window_seconds,
    )
    return await _run_enhance(file=file, route_kind="web", req_settings=req_settings)
