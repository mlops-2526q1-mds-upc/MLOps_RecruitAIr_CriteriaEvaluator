import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class BaseEvaluatorModel:
    """Abstract interface for evaluator models. Implement predict(criteria, resume) -> float."""
    def predict(self, criteria: str, resume: str) -> float:
        raise NotImplementedError()

    @property
    def version(self) -> Optional[str]:
        return None


class DummyEvaluator(BaseEvaluatorModel):
    """
    Deterministic lightweight scorer used for local testing or when real model isn't available.

    Algorithm:
    - Lowercase both texts
    - Score = (#shared words) / (sqrt(len(criteria_words) * len(resume_words)) + eps)
    - Clamp to [0,1]
    This yields a reproducible score useful for tests and local dev.
    """
    def __init__(self):
        self._version = "dummy-v1"

    @property
    def version(self) -> str:
        return self._version

    def _tokenize(self, text: str):
        # very lightweight tokenization (for demo/testing only)
        return [w for w in "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in text.lower()).split() if len(w) > 1]

    def predict(self, criteria: str, resume: str) -> float:
        t0 = time.time()
        c_tokens = set(self._tokenize(criteria))
        r_tokens = set(self._tokenize(resume))

        if not c_tokens or not r_tokens:
            return 0.0

        shared = c_tokens.intersection(r_tokens)
        # cosine-sim-like normalized by sqrt(|A||B|)
        import math
        score = len(shared) / (math.sqrt(len(c_tokens) * len(r_tokens)) + 1e-9)
        # clamp 0..1
        score = max(0.0, min(1.0, score))
        elapsed = time.time() - t0
        logger.debug("Dummy predict finished in %.4fs, score=%.4f", elapsed, score)
        return score


# Example MLflow loader stub (real implementation would call mlflow.pytorch.load_model(...))
def load_model_from_mlflow(model_uri: str):
    """
    Placeholder: implement actual mlflow load in production.
    This stub demonstrates where to add MLflow or torch loading logic.
    """
    
