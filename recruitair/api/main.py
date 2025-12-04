from contextlib import asynccontextmanager
import logging
import time

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .dependencies import get_model
from .model import BaseEvaluatorModel
from .schemas import EvalRequest, EvalResponse

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Preloading model at startup...")
    _get_model = app.dependency_overrides.get(get_model, get_model)
    model: BaseEvaluatorModel = _get_model()
    logger.info("Loaded model version: %s", getattr(model, "version", "unknown"))
    yield


app = FastAPI(title="Applicant Evaluator API", version="v1", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


@app.post("/eval", response_model=EvalResponse)
def evaluate(request: EvalRequest, model: BaseEvaluatorModel = Depends(get_model)):
    """Evaluate a single applicant CV against a single criterion and return score in [0,1]."""
    t0 = time.time()
    try:
        score = model.predict(request.criteria_description, request.applicant_cv)
    except Exception as exc:
        logger.exception("Model prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Model prediction failed")
    elapsed = time.time() - t0
    return EvalResponse(
        score=score, model_version=getattr(model, "version", None), elapsed_seconds=elapsed
    )


@app.get("/health")
def health():
    return {"status": "ok"}
