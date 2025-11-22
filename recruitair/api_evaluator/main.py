import time
import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from .schemas import EvalRequest, EvalResponse
from .model import DummyEvaluator, BaseEvaluatorModel
from .config import settings
from typing import Callable

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Applicant Evaluator API", version=settings.api_version)

# CORS — adjust origins as needed for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# Dependency resolver for model: allows overrides in tests
def get_model() -> BaseEvaluatorModel:
    """
    Return an instance of a model implementing BaseEvaluatorModel.
    Replace the DummyEvaluator with your real loader (MLflow/PyTorch) in production.
    """
    if settings.use_dummy:
        return DummyEvaluator()
    try:
         model = load_model_from_mlflow(settings.model_uri)
         return model
    except Exception as e:
         logger.exception("Failed to load model: %s", e)
         raise RuntimeError("Model loading failed")
    


@app.post("/eval", response_model=EvalResponse)
def evaluate(request: EvalRequest, model: BaseEvaluatorModel = Depends(get_model)):
    """
    Evaluate the suitability of applicant_cv for a single criterion.
    Returns score in [0,1], model version (if available), and elapsed_seconds.
    """
    t0 = time.time()
    try:
        score = model.predict(request.criteria_description, request.applicant_cv)
    except Exception as exc:
        logger.exception("Model prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Model prediction failed")

    elapsed = time.time() - t0
    # Ensure score in [0,1]
    score = max(0.0, min(1.0, float(score)))
    return EvalResponse(score=score, model_version=getattr(model, "version", None), elapsed_seconds=elapsed)


@app.get("/health")
def health():
    return {"status": "ok"}
