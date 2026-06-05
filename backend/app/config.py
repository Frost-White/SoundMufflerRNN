from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/soundmuffler"
    jwt_secret: str = "change-me-in-production-use-long-random-string"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 7
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    enhance_web_rate_limit: int = 30
    enhance_web_rate_window_seconds: int = 60
    enhance_api_free_rate_limit: int = 30
    enhance_api_free_rate_window_seconds: int = 900
    enhance_api_pro_rate_limit: int = 15
    enhance_api_pro_rate_window_seconds: int = 60
    enhance_max_upload_bytes: int = 2 * 1024 * 1024


def get_settings() -> Settings:
    return Settings()
