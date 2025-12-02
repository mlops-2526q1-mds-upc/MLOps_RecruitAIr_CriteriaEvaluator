from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RECRUITAIR_")

    model: str = "criteria-evaluation"  # default MLflow model URI; override via env
    model_version: str = "2"  # or specify a version like "1"
    # Don't forget to set MLFLOW_ARTIFACT_URI env var if needed
    device: str | None = None  # "cuda" or "cpu" or None to auto-select


settings = Settings()
