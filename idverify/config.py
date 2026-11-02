"""Paths, environment overrides and tuning thresholds for IDVerify."""

from __future__ import annotations

import os
from pathlib import Path

# Repository root - the directory that contains the ``idverify`` package.
ROOT = Path(__file__).resolve().parent.parent

ENV_HOME = "IDVERIFY_HOME"
ENV_MODELS = "IDVERIFY_MODELS"

# Vendored face-model filenames (OpenCV Zoo, downloaded once at build time).
DETECTOR_FILE = "face_detection_yunet_2023mar.onnx"
RECOGNIZER_FILE = "face_recognition_sface_2021dec.onnx"

# --- Face matching thresholds -------------------------------------------------
# Cosine similarity between two SFace 128-D embeddings.
# Measured while building this system on the bundled sample faces:
#   same person, card portrait vs live capture .... 0.83
#   different people .............................. 0.05 - 0.10
# 0.45 sits well above OpenCV's LFW-calibrated 0.363 floor (ID prints are lower
# quality than LFW) while 0.30 stays far above the impostor population.
FACE_ACCEPT_COSINE = 0.45
FACE_REJECT_COSINE = 0.30

# --- Capture quality gates ----------------------------------------------------
# A frame that fails these is rejected before any matching happens, which is
# where most false accepts in a real deployment come from.
MIN_DETECTION_SCORE = 0.80  # YuNet confidence required to trust a face
MIN_FACE_PX = 80  # smallest accepted live-face edge, in pixels
MIN_CARD_FACE_PX = 60  # smallest accepted printed-portrait edge, in pixels

# --- OCR ----------------------------------------------------------------------
OCR_MIN_SCORE = 0.50  # discard OCR lines the engine is unsure about
OCR_MIN_LINE_SCORE = 0.35  # keep more lines for field-value fallback parsing


def models_dir() -> Path:
    """Directory holding the two vendored ONNX face models."""
    env = os.environ.get(ENV_MODELS)
    if env:
        return Path(env).expanduser()
    packaged = Path(__file__).resolve().parent / "models"
    if packaged.is_dir():
        return packaged
    return ROOT / "models"


def home_dir() -> Path:
    """Writable data directory: local SQLite database plus captured images."""
    env = os.environ.get(ENV_HOME)
    if env:
        return Path(env).expanduser()
    return Path.home() / ".idverify"