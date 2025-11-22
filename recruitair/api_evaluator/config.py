from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_uri: str = ""           # e.g., "models:/criteria-evaluation/2"
    use_dummy: bool = True        # default True for local development
    api_version: str = "v1"

    class Config:
        env_prefix = "EVAL_"


settings = Settings()
