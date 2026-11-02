"""Small shared helpers: image IO and timestamps."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


def load_image(src: "str | os.PathLike | np.ndarray") -> np.ndarray:
    """Load an image as a BGR ``uint8`` array from a path or pass an array through.

    Uses ``imdecode`` rather than ``imread`` so paths with non-ASCII characters
    work on Windows.
    """
    if isinstance(src, np.ndarray):
        img = src
    else:
        path = str(src)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not decode image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return np.ascontiguousarray(img)


def save_image(path: "str | os.PathLike", img: np.ndarray) -> Path:
    """Write an image, tolerating non-ASCII paths."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix or ".png"
    ok, buf = cv2.imencode(suffix, img)
    if not ok:
        raise ValueError(f"Could not encode image for {target}")
    target.write_bytes(buf.tobytes())
    return target


def now_iso() -> str:
    """UTC timestamp, second resolution, ISO-8601 with offset."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: "str | os.PathLike") -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target