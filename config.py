"""
Central configuration for the Alzheimer's Early Detection pipeline.

Change these values (or override via environment variables / CLI flags in the
individual scripts) instead of hardcoding paths inside notebooks or modules.
"""

import os

# --- Data ---------------------------------------------------------------
# Expected folder layout:
#   data/
#     MildDemented/*.jpg
#     ModerateDemented/*.jpg
#     NonDemented/*.jpg
#     VeryMildDemented/*.jpg
DATA_DIR = os.environ.get("AD_XAI_DATA_DIR", "data")

CATEGORIES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented",
]

IMG_HEIGHT = 208
IMG_WIDTH = 176
IMG_CHANNELS = 3
NUM_CLASSES = len(CATEGORIES)

# --- Training -------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# --- Output ---------------------------------------------------------------
OUTPUT_DIR = os.environ.get("AD_XAI_OUTPUT_DIR", "outputs")
MODEL_PATH = os.path.join(OUTPUT_DIR, "model", "ad_xai_model")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics")
EXPLANATIONS_PATH = os.path.join(OUTPUT_DIR, "explanations")
