from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RECRUITAIR_")

    model_uri: str = "models:/criteria-evaluation/2"  # default MLflow model URI; override via env
    use_dummy: bool = False
    device: str | None = None  # "cuda" or "cpu" or None to auto-select
    api_version: str = "v1"


settings = Settings()
