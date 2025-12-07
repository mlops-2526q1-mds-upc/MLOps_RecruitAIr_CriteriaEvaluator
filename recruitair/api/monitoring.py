"""
Prometheus metrics for the RecruitAIr Criteria Evaluator API.

Import and use these in the /eval endpoint and wherever you call the model.
"""

from prometheus_client import Counter, Histogram

# --- API-level metrics ---

CRIT_EVAL_REQUESTS_TOTAL = Counter(
    "recruitair_criteria_eval_requests_total",
    "Total number of /eval requests received by CriteriaEvaluator",
)

CRIT_EVAL_REQUESTS_FAILED_TOTAL = Counter(
    "recruitair_criteria_eval_requests_failed_total",
    "Total number of /eval requests that resulted in an error",
)

CRIT_EVAL_REQUEST_LATENCY_SECONDS = Histogram(
    "recruitair_criteria_eval_request_latency_seconds",
    "Latency of /eval endpoint in CriteriaEvaluator, in seconds",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)

# --- Model-level metrics ---

CRIT_EVAL_PREDICTION_LATENCY_SECONDS = Histogram(
    "recruitair_criteria_model_evaluation_latency_seconds",
    "Time spent in evaluator model (criteria evaluation) in seconds",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)

# --- Payload / business metrics ---


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
