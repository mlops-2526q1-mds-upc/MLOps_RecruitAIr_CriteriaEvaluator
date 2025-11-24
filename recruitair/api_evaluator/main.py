import time
import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

from .schemas import EvalRequest, EvalResponse
from .dependencies import get_default_model
from .model import BaseEvaluatorModel

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Applicant Evaluator API", version="v1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


@app.post("/eval", response_model=EvalResponse)
def evaluate(request: EvalRequest, model: BaseEvaluatorModel = Depends(get_default_model)):
    """Evaluate a single applicant CV against a single criterion and return score in [0,1]."""
    t0 = time.time()
    try:
        score = model.predict(request.criteria_description, request.applicant_cv)
    except Exception as exc:
        logger.exception("Model prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Model prediction failed")
    elapsed = time.time() - t0
    return EvalResponse(score=score, model_version=getattr(model, "version", None), elapsed_seconds=elapsed)


@app.get("/health")
def health():
    return {"status": "ok"}
