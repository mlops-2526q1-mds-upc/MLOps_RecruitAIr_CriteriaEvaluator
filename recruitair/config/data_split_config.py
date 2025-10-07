"""Config for splitting the data."""

from .config_base import PROJ_ROOT

DATA_DIR = PROJ_ROOT / "data"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

SEED = 42
TRAIN_SPLIT = 0.7
VALIDATION_SPLIT = 0.2
