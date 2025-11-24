# recruitair/api_evaluator/model.py
import os
import time
import logging
from typing import Optional

import torch
import mlflow
from tempfile import TemporaryDirectory
from recruitair.modeling.tokenize import ResumeAndCriteriaTokenizer

logger = logging.getLogger(__name__)


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
            for w in "".join(
                ch if (ch.isalnum() or ch.isspace()) else " " for ch in text.lower()
            ).split()
            if len(w) > 1
        ]

    def predict(self, criteria: str, resume: str) -> float:
        c_tokens = set(self._tokenize(criteria))
        r_tokens = set(self._tokenize(resume))
        if not c_tokens or not r_tokens:
            return 0.0
        import math

        score = len(c_tokens.intersection(r_tokens)) / (
            math.sqrt(len(c_tokens) * len(r_tokens)) + 1e-9
        )
        return max(0.0, min(1.0, float(score)))


class TorchMLflowEvaluator(BaseEvaluatorModel):
    """
    Loads a PyTorch model from Mlflow and the custom tokenizer artifact saved by your training script.
    Expects the same call signature used in training notebooks:
        encoded = tokenizer(resume, criteria, padding=True, return_tensors="pt", padding_side="left").to(device)
        preds = model(**encoded)   # returns tensor shape (batch, 1)
    """

    def __init__(self, model_uri: str, device: Optional[str] = None):
        self.model_uri = model_uri
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._tokenizer = None
        self._version = None
        self._load()

    @property
    def version(self) -> Optional[str]:
        return self._version

    def _load(self):
        t0 = time.time()
        # load model via mlflow
        logger.info("Loading model from mlflow uri=%s", self.model_uri)
        # mlflow.pytorch.load_model will return the model (same approach used in your notebook)
        self._model = mlflow.pytorch.load_model(
            model_uri=self.model_uri, map_location=self._device
        )
        # try to fetch model version from uri or model metadata
        self._version = self.model_uri

        # fetch tokenizer extra_files/artifacts the training logged
        try:
            with TemporaryDirectory() as tmpdir:
                logger.info("Downloading model artifacts to %s", tmpdir)
                mlflow.artifacts.download_artifacts(artifact_uri=self.model_uri, dst_path=tmpdir)
                # your training saved the tokenizer under extra_files/tokenizer or a tokenizer dir
                # try common candidate locations
                cand = [
                    os.path.join(tmpdir, "tokenizer"),
                    os.path.join(tmpdir, "extra_files", "tokenizer"),
                    os.path.join(tmpdir, "extra", "tokenizer"),
                ]
                tokenizer_dir = None
                for c in cand:
                    if os.path.isdir(c):
                        tokenizer_dir = c
                        break
                if tokenizer_dir is None:
                    # sometimes mlflow stores tokenizer inside model artifact folder
                    # try to find any folder containing "tokenizer" substring
                    for root, dirs, _ in os.walk(tmpdir):
                        for d in dirs:
                            if "tokenizer" in d.lower():
                                tokenizer_dir = os.path.join(root, d)
                                break
                        if tokenizer_dir:
                            break
                if tokenizer_dir is None:
                    logger.warning(
                        "Could not find tokenizer directory among downloaded artifacts: %s", tmpdir
                    )
                else:
                    self._tokenizer = ResumeAndCriteriaTokenizer.from_pretrained(tokenizer_dir)
                    logger.info("Loaded tokenizer from %s", tokenizer_dir)
        except Exception as e:
            logger.exception("Failed to download/load tokenizer: %s", e)
            # continue; tokenizer may not be strictly required if you supply an alternate encoding strategy

        # send model to proper device
        # if mlflow returned a module on CPU, move to requested device
        try:
            if self._device.startswith("cuda"):
                self._model.to(self._device)
        except Exception:
            pass

        logger.info("Model loaded in %.2fs", time.time() - t0)

    def predict(self, criteria: str, resume: str) -> float:
        if self._model is None:
            raise RuntimeError("Model not loaded")

        # Prepare input with tokenizer
        if self._tokenizer is None:
            # If tokenizer missing, fall back to naive string input (not ideal)
            raise RuntimeError(
                "Tokenizer not found. Please ensure tokenizer was logged with the model."
            )
        # tokenizer usage matches the training:
        encoded = self._tokenizer(
            [resume], [criteria], padding=True, return_tensors="pt", padding_side="left"
        )
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
