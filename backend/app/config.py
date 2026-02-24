from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    app_name: str = "Persi API"
    debug: bool = True

    # Database
    database_url: str = "postgresql+asyncpg://persi:persi_dev@localhost:5433/persi"

    # CORS
    frontend_url: str = "http://localhost:3000"

    # AI
    anthropic_api_key: str = ""

    # Dev mode: hardcoded user ID (replaced by Supabase JWT in Phase 2)
    dev_user_id: str = "dev-user-001"
    dev_user_email: str = "dev@persi.app"

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
