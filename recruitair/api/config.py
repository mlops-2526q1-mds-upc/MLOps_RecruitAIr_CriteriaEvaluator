import os
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RECRUITAIR_")
    # Don't forget to set MLFLOW_TRACKING_URI in your environment variables
    model: str = Field("criteria-evaluation", description="default MLflow model URI; override via env")
    model_version: str = Field("latest", description="Model version to load")
    device: Literal["cuda", "cpu", None] = Field(None, description='"cuda" or "cpu" or None to auto-select')


settings = Settings()

if not os.getenv("MLFLOW_TRACKING_URI"):
    raise EnvironmentError("Please set the MLFLOW_TRACKING_URI environment variable.")
