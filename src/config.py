"""
config.py

Central configuration file for the Heat Sink Surrogate Modeling project.
"""

from pathlib import Path

# =============================================================================
# Project Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SPLIT_DATA_DIR = DATA_DIR / "splits"
MODEL_DIR = PROJECT_ROOT / "models"

# =============================================================================
# Dataset Configuration
# =============================================================================

DATASET_SIZE = 10_000

TEST_SIZE = 0.20

RANDOM_STATE = 42

# =============================================================================
# Parameter Sweep Ranges
# =============================================================================

TDP_RANGE = (30.0, 250.0)

AIR_VELOCITY_RANGE = (0.5, 15.0)

TIM_CONDUCTIVITY_RANGE = (1.0, 12.0)

# =============================================================================
# Machine Learning
# =============================================================================

RF_ESTIMATORS = 300

N_JOBS = -1

# =============================================================================
# Create Required Directories
# =============================================================================

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)