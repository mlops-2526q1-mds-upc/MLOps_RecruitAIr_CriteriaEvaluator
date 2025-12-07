from functools import lru_cache
import os

from .config import settings
from .model import BaseEvaluatorModel, DummyEvaluator, TorchMLflowEvaluator


@lru_cache()
def get_model() -> BaseEvaluatorModel:
    """Get the evaluator model instance, loading it if necessary."""
    if os.getenv("MLFLOW_TRACKING_URI") is None:
        raise EnvironmentError("Please set the MLFLOW_TRACKING_URI environment variable.")
    return TorchMLflowEvaluator(
        model_name=settings.model,
        model_version=settings.model_version,
        cache_dir=settings.cache_dir,
        device=settings.device,
    )
