#!/bin/sh
set -e

python scripts/ensure_database.py
alembic upgrade head
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
