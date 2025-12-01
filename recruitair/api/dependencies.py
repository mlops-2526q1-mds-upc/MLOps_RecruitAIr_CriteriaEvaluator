from functools import lru_cache
from .config import settings
from .model import DummyEvaluator, TorchMLflowEvaluator, BaseEvaluatorModel


@lru_cache()
def get_default_model() -> BaseEvaluatorModel:
    return TorchMLflowEvaluator(
        model_uri=settings.model_uri,
        device=settings.device,
    )
    # return DummyEvaluator()
