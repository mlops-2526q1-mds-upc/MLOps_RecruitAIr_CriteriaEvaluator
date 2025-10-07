"""Config for data pre-processing."""

from .config_base import PROJ_ROOT

# Paths for preprocessing
DATA_DIR = PROJ_ROOT / "data"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_DIR = DATA_DIR / "raw"
