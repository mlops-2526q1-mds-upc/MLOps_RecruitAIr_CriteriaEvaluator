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

CRIT_MODEL_EVAL_LATENCY_SECONDS = Histogram(
    "recruitair_criteria_model_evaluation_latency_seconds",
    "Time spent in evaluator model (criteria evaluation) in seconds",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)

CRIT_MODEL_EVAL_ERRORS_TOTAL = Counter(
    "recruitair_criteria_model_evaluation_errors_total",
    "Total number of exceptions raised by criteria evaluation model",
)

# --- Payload / business metrics ---

REQUEST_NUM_CRITERIA = Histogram(
    "recruitair_criteria_request_num_criteria",
    "Number of criteria provided in /eval requests",
    buckets=(1, 3, 5, 10, 20, 50),
)

PROFILE_TEXT_LENGTH = Histogram(
    "recruitair_criteria_profile_text_length_chars",
    "Length of profile text provided for evaluation (chars)",
    buckets=(128, 256, 512, 1024, 2048, 4096, 8192),
)
