from functools import lru_cache

from .config import settings
from .model import BaseEvaluatorModel, DummyEvaluator, TorchMLflowEvaluator


@lru_cache()
def get_default_model() -> BaseEvaluatorModel:
    return TorchMLflowEvaluator(
        model_uri=f"models:/{settings.model}/{settings.model_version}",
        device=settings.device,
    )
    # return DummyEvaluator()
