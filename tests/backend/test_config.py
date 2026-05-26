from app.config import Settings


def test_settings_accept_overrides() -> None:
    s = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        jwt_secret="abc",
        cors_origins="http://a.com,http://b.com",
        enhance_api_rate_limit=5,
    )
    assert s.database_url.startswith("sqlite")
    assert s.jwt_secret == "abc"
    assert "http://a.com" in s.cors_origins
    assert s.enhance_api_rate_limit == 5
