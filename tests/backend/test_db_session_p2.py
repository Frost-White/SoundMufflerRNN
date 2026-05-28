import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BACKEND_PATH = ROOT / "backend"
if str(BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PATH))

import app.db.session as session_mod  # noqa: E402


def test_get_db_yields_and_closes_session(monkeypatch) -> None:
    class _Session:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    holder = {}

    def _factory():
        s = _Session()
        holder["session"] = s
        return s

    monkeypatch.setattr(session_mod, "SessionLocal", _factory)

    gen = session_mod.get_db()
    db = next(gen)
    assert db is holder["session"]
    try:
        next(gen)
    except StopIteration:
        pass
    assert holder["session"].closed is True
