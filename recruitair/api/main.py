from contextlib import asynccontextmanager
import logging
import time

from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import (
    Counter,
    Histogram,
    CONTENT_TYPE_LATEST,
    generate_latest,
)

from .dependencies import get_model
from .model import BaseEvaluatorModel
from .schemas import EvalRequest, EvalResponse

logger = logging.getLogger("uvicorn.error")


# -------------------------------------------------------------
# PROMETHEUS METRICS
# -------------------------------------------------------------

CRIT_EVAL_REQUESTS_TOTAL = Counter(
    "recruitair_criteria_eval_requests_total",
    "Total number of /eval requests received by CriteriaEvaluator",
)

CRIT_EVAL_REQUESTS_FAILED_TOTAL = Counter(
    "recruitair_criteria_eval_requests_failed_total",
    "Total number of /eval requests that failed",
)

CRIT_EVAL_LATENCY_SECONDS = Histogram(
    "recruitair_criteria_eval_latency_seconds",
    "Latency of /eval endpoint in seconds",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)

CRIT_MODEL_PREDICTION_LATENCY_SECONDS = Histogram(
    "recruitair_criteria_model_prediction_latency_seconds",
    "Time spent inside model.predict()",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)

CRITERION_LENGTH_CHARS = Histogram(
    "recruitair_criteria_description_length_chars",
    "Length of criteria_description input",
    buckets=(32, 64, 128, 256, 512, 1024),
)

APPLICANT_CV_LENGTH_CHARS = Histogram(
    "recruitair_applicant_cv_length_chars",
    "Length of applicant CV input",
    buckets=(128, 256, 512, 1024, 2048, 4096, 8192),
)


# -------------------------------------------------------------
# APP LIFECYCLE
# -------------------------------------------------------------

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


# -------------------------------------------------------------
# /eval ENDPOINT WITH MONITORING
# -------------------------------------------------------------

@app.post("/eval", response_model=EvalResponse)
def evaluate(request: EvalRequest, model: BaseEvaluatorModel = Depends(get_model)):
    """
    Evaluate a CV against a criterion and return a score in [0,1].
    Now monitored with Prometheus metrics.
    """
    CRIT_EVAL_REQUESTS_TOTAL.inc()

    # Observing input sizes
    CRITERION_LENGTH_CHARS.observe(len(request.criteria_description or ""))
    APPLICANT_CV_LENGTH_CHARS.observe(len(request.applicant_cv or ""))

    # Measure endpoint latency
    endpoint_start = time.time()

    try:
        # Measure model prediction time
        model_start = time.time()
        score = model.predict(request.criteria_description, request.applicant_cv)
        CRIT_MODEL_PREDICTION_LATENCY_SECONDS.observe(time.time() - model_start)

    except Exception as exc:
        CRIT_EVAL_REQUESTS_FAILED_TOTAL.inc()
        logger.exception("Model prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Model prediction failed")

    finally:
        CRIT_EVAL_LATENCY_SECONDS.observe(time.time() - endpoint_start)

    return EvalResponse(
        score=score,
        model_version=getattr(model, "version", None),
        elapsed_seconds=time.time() - endpoint_start,
    )


# -------------------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# -------------------------------------------------------------
# /metrics ENDPOINT FOR PROMETHEUS
# -------------------------------------------------------------

@app.get("/metrics")
def metrics():
    """
    Expose Prometheus metrics for scraping.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
