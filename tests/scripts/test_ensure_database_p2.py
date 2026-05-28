import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# pylint: disable=import-error


ROOT = Path(__file__).resolve().parents[2]
BACKEND_PATH = ROOT / "backend"
SCRIPTS_PATH = BACKEND_PATH / "scripts"
for p in (str(BACKEND_PATH), str(SCRIPTS_PATH)):
    if p not in sys.path:
        sys.path.insert(0, p)

import ensure_database as ensure_db  # noqa: E402


class _Url:
    def __init__(self, database: str | None):
        self.database = database

    def set(self, database: str):
        return f"admin://{database}"


class _Conn:
    def __init__(self, exists_value):
        self._exists_value = exists_value
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, stmt, params=None):
        self.executed.append((str(stmt), params))
        if params and params.get("name") is not None:
            return SimpleNamespace(scalar=lambda: self._exists_value)
        return SimpleNamespace()


class _Engine:
    def __init__(self, conn: _Conn):
        self._conn = conn

    def connect(self):
        return self._conn


def test_main_exits_when_database_name_missing(monkeypatch) -> None:
    monkeypatch.setattr(ensure_db, "get_settings", lambda: SimpleNamespace(database_url="postgresql://host"))
    monkeypatch.setattr(ensure_db, "make_url", lambda _url: _Url(database=None))
    with pytest.raises(SystemExit):
        ensure_db.main()


def test_main_does_not_create_when_database_exists(monkeypatch, capsys) -> None:
    conn = _Conn(exists_value=1)
    monkeypatch.setattr(ensure_db, "get_settings", lambda: SimpleNamespace(database_url="postgresql://host/db1"))
    monkeypatch.setattr(ensure_db, "make_url", lambda _url: _Url(database="db1"))
    monkeypatch.setattr(ensure_db, "create_engine", lambda *_a, **_kw: _Engine(conn))
    ensure_db.main()
    out = capsys.readouterr().out
    assert "already exists" in out
    assert not any("CREATE DATABASE" in stmt for stmt, _ in conn.executed)


def test_main_creates_database_when_missing(monkeypatch, capsys) -> None:
    conn = _Conn(exists_value=None)
    monkeypatch.setattr(ensure_db, "get_settings", lambda: SimpleNamespace(database_url="postgresql://host/db2"))
    monkeypatch.setattr(ensure_db, "make_url", lambda _url: _Url(database="db2"))
    monkeypatch.setattr(ensure_db, "create_engine", lambda *_a, **_kw: _Engine(conn))
    ensure_db.main()
    out = capsys.readouterr().out
    assert "Created database" in out
    assert any('CREATE DATABASE "db2"' in stmt for stmt, _ in conn.executed)
