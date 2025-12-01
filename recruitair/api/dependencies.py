from functools import lru_cache

from .config import settings
from .model import BaseEvaluatorModel, DummyEvaluator, TorchMLflowEvaluator


@lru_cache()
def get_default_model() -> BaseEvaluatorModel:
    return TorchMLflowEvaluator(
        model_uri=settings.model_uri,
        device=settings.device,
    )
    # return DummyEvaluator()
    # return DummyEvaluator()
    # return DummyEvaluator()
    # return DummyEvaluator()
