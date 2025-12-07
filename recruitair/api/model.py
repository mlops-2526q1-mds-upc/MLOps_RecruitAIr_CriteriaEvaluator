# recruitair/api_evaluator/model.py
import logging
import os
from tempfile import TemporaryDirectory
import time
from typing import Optional

from filelock import FileLock
import mlflow.artifacts
import torch

from recruitair.modeling.tokenize import ResumeAndCriteriaTokenizer

logger = logging.getLogger("uvicorn.error")


class BaseEvaluatorModel:
    """Interface for evaluator models"""

    def predict(self, criteria: str, resume: str) -> float:
        raise NotImplementedError()

    @property
    def version(self) -> Optional[str]:
        return None


class DummyEvaluator(BaseEvaluatorModel):
    """Deterministic lightweight scorer for local dev & tests."""

    def __init__(self):
        self._version = "dummy-v1"

    @property
    def version(self) -> str:
        return self._version

    def _tokenize(self, text: str):
        return [
            w
            for w in "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in text.lower()).split()
            if len(w) > 1
        ]

    def predict(self, criteria: str, resume: str) -> float:
        c_tokens = set(self._tokenize(criteria))
        r_tokens = set(self._tokenize(resume))
        if not c_tokens or not r_tokens:
            return 0.0
        import math

        score = len(c_tokens.intersection(r_tokens)) / (math.sqrt(len(c_tokens) * len(r_tokens)) + 1e-9)
        return max(0.0, min(1.0, float(score)))


class TorchMLflowEvaluator(BaseEvaluatorModel):
    """
    Loads a PyTorch model from Mlflow and the custom tokenizer artifact saved by your training script.
    Expects the same call signature used in training notebooks:
        encoded = tokenizer(resume, criteria, padding=True, return_tensors="pt", padding_side="left").to(device)
        preds = model(**encoded)   # returns tensor shape (batch, 1)
    """

    def __init__(
        self, model_name: str, model_version: str, device: Optional[str] = None, *, cache_dir: Optional[str] = None
    ):
        assert isinstance(model_name, str) and model_name, "model_name must be a non-empty string"
        assert isinstance(model_version, str) and model_version, "model_version must be a non-empty string"
        self.model_name = model_name
        self.model_version = model_version
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._cache_dir = cache_dir
        self._model = None
        self._tokenizer = None
        self._version = None
        self._load()

    @property
    def version(self) -> Optional[str]:
        return self._version

    @property
    def model_uri(self) -> str:
        return f"models:/{self.model_name}/{self.model_version}"

    @property
    def cache_dir(self) -> Optional[str]:
        if self._cache_dir:
            res = os.path.join(self._cache_dir, "mlflow_models", self.model_name, self.model_version)
            return res
        return None

    def _load_model(self) -> float:
        """
        Load the model from MLflow.
        Returns the time taken to load the model in seconds.
        """
        t0 = time.monotonic()
        logger.info("Loading model from mlflow uri=%s", self.model_uri)
        if self.cache_dir:
            model_dir = os.path.join(self.cache_dir, "model")
            os.makedirs(model_dir, exist_ok=True)
            # Check if the model is already cached
            if os.listdir(model_dir) != []:
                logger.info("Loading model from cache directory: %s", model_dir)
                self._model = mlflow.pytorch.load_model(model_uri=model_dir, map_location=self._device)
            else:
                logger.info("Downloading model into cache directory: %s", model_dir)
                with FileLock(os.path.join(self.cache_dir, "model.lock")):
                    self._model = mlflow.pytorch.load_model(
                        model_uri=self.model_uri, map_location=self._device, dst_path=model_dir
                    )
        else:
            # mlflow.pytorch.load_model will return the model (same approach used in your notebook)
            self._model = mlflow.pytorch.load_model(model_uri=self.model_uri, map_location=self._device)
        logger.info("Model loaded from %s", self.model_uri)
        return time.monotonic() - t0

    def _load_tokenizer(self):
        """
        Load the tokenizer from MLflow artifacts.

        Returns the time taken to load the tokenizer in seconds.
        """
        t0 = time.time()
        # fetch tokenizer extra_files/artifacts the training logged
        if not self.cache_dir:
            tmpdir = TemporaryDirectory()
            tokenizer_dir = tmpdir.name
            logger.info("Downloading model artifacts to %s", tokenizer_dir)
            try:
                mlflow.artifacts.download_artifacts(artifact_uri=self.model_uri, dst_path=tokenizer_dir)
            except Exception as e:
                logger.exception("Failed to download model artifacts: %s", e)
        else:
            tokenizer_dir = os.path.join(self.cache_dir, "tokenizer")
            os.makedirs(tokenizer_dir, exist_ok=True)
            if os.listdir(tokenizer_dir) == []:
                logger.info("Downloading model artifacts into cache directory: %s", tokenizer_dir)
                os.makedirs(tokenizer_dir, exist_ok=True)
                with FileLock(os.path.join(self.cache_dir, "tokenizer.lock")):
                    try:
                        mlflow.artifacts.download_artifacts(artifact_uri=self.model_uri, dst_path=tokenizer_dir)
                    except Exception as e:
                        logger.exception("Failed to download model artifacts: %s", e)
            else:
                logger.info("Loading tokenizer from cache directory: %s", tokenizer_dir)

        try:
            self._tokenizer = ResumeAndCriteriaTokenizer.from_pretrained(
                os.path.join(tokenizer_dir, "extra_files", "tokenizer"),
            )
            logger.info("Loaded tokenizer from %s", tokenizer_dir)
        except Exception as e:
            logger.exception("Failed to load tokenizer: %s", e)
            # continue; tokenizer may not be strictly required if you supply an alternate encoding strategy
        if not self.cache_dir:
            tmpdir.cleanup()
        logger.info("Tokenizer loaded in %.2fs", time.time() - t0)
        return time.time() - t0

    def _load(self):
        t0 = time.time()
        self._load_model()
        self._load_tokenizer()
        self._version = f"mlflow-{self.model_name}-v{self.model_version}"
        logger.info("Model and tokenizer loaded in %.2fs", time.time() - t0)

    def predict(self, criteria: str, resume: str) -> float:
        if self._model is None:
            raise RuntimeError("Model not loaded")

        # Prepare input with tokenizer
        if self._tokenizer is None:
            # If tokenizer missing, fall back to naive string input (not ideal)
            raise RuntimeError("Tokenizer not found. Please ensure tokenizer was logged with the model.")
        # tokenizer usage matches the training:
        encoded = self._tokenizer([resume], [criteria], padding=True, return_tensors="pt", padding_side="left")
        # move tensors to device
        device = torch.device(self._device)
        encoded = {k: v.to(device) for k, v in encoded.items()}

        self._model.eval()
        with torch.no_grad():
            preds = self._model(**encoded)
            # In training, model returned (batch, 1) tensor directly
            # If the loaded model returns a single tensor or a dict, handle both
            if torch.is_tensor(preds):
                out = preds
            elif isinstance(preds, (tuple, list)):
                out = preds[0]
            elif isinstance(preds, dict):
                # try common keys
                for k in ("logits", "preds", "output"):
                    if k in preds:
                        out = preds[k]
                        break
                else:
                    # pick first tensor-like value
                    first = next(iter(preds.values()))
                    out = first
            else:
                raise RuntimeError(f"Unsupported model output type: {type(preds)}")
            # ensure shape (batch, 1) or (batch,)
            out = out.detach()
            if out.ndim == 2 and out.shape[1] == 1:
                out = out.squeeze(1)
            # take first (single input)
            score = float(out[0].cpu().item())
            # clamp 0..1
            score = max(0.0, min(1.0, score))
            return score
            return score
