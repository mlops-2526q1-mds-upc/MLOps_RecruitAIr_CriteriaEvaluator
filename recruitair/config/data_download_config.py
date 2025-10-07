"""Config for downloading data."""

from .config_base import PROJ_ROOT

# Paths for data download
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# HuggingFace dataset info
HF_RESUME_SCORE_DETAILS_REPO = "netsol/resume-score-details"
HF_RESUME_SCORE_DETAILS_REVISION = "f2b4938"
