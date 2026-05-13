"""Create application database if missing (connects via maintenance DB 'postgres')."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
os.chdir(_BACKEND)
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.engine.url import make_url  # noqa: E402

from app.config import get_settings  # noqa: E402


def main() -> None:
    settings = get_settings()
    u = make_url(settings.database_url)
    target_db = u.database
    if not target_db:
        raise SystemExit("DATABASE_URL must include a database name")

    admin = u.set(database="postgres")
    engine = create_engine(admin, isolation_level="AUTOCOMMIT")
    with engine.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :name"),
            {"name": target_db},
        ).scalar()
        if exists:
            print(f"Database '{target_db}' already exists.")
            return
        conn.execute(text(f'CREATE DATABASE "{target_db}"'))
        print(f"Created database '{target_db}'.")


if __name__ == "__main__":
    main()
