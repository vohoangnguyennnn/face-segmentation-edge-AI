"""Shared constants across training, deployment, and visualization scripts.

All paths are relative to the repo root. Scripts must be run from the repo root
— a RuntimeError is raised if requirements.txt is not found in CWD.
"""

import os

# ─── Guard: must run from repo root ────────────────────────────────────────
_ROOT_GUARD = "requirements.txt"
if not os.path.isfile(_ROOT_GUARD):
    raise RuntimeError(
        f"'{_ROOT_GUARD}' not found — run this script from the repo root.\n"
        f"Current directory: {os.getcwd()}"
    )

# ─── Dataset ────────────────────────────────────────────────────────────────
DATASET_SPLIT_DIR = os.path.join("dataset_split")
REP_IMAGES_DIR    = "rep_images"

# ─── Model artifacts ────────────────────────────────────────────────────────
MODEL_DIR       = os.path.join("artifacts", "models")
KERAS_MODEL     = os.path.join(MODEL_DIR, "unet_face_segmentation.keras")
KERAS_BEST      = os.path.join(MODEL_DIR, "unet_face_segmentation_best.keras")
TFLITE_FP32     = os.path.join(MODEL_DIR, "unet_fp32.tflite")
TFLITE_INT8     = os.path.join(MODEL_DIR, "unet_int8.tflite")