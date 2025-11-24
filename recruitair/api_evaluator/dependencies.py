from functools import lru_cache
from .config import settings
from .model import DummyEvaluator, TorchMLflowEvaluator, BaseEvaluatorModel


@lru_cache(maxsize=1)
def get_default_model() -> BaseEvaluatorModel:
    """
    Returns the loaded model instance. Cached so we only load it once per process.
    For dev, if settings.use_dummy is True, returns the dummy evaluator.
    """
    if settings.use_dummy:
        return DummyEvaluator()
    return TorchMLflowEvaluator(model_uri=settings.model_uri, device=settings.device)
