"""Webcam capture. Every other part of the system also accepts image files,
so a machine without a camera stays fully usable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .util import save_image


class CameraError(RuntimeError):
    """The camera could not be opened or delivered a frame."""


class Camera:
    """A single OpenCV capture device, opened lazily."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720) -> None:
        self.index = index
        self.width = width
        self.height = height
        self._capture: Optional[cv2.VideoCapture] = None

    # -- lifecycle -----------------------------------------------------
    def open(self) -> "Camera":
        if self._capture is not None:
            return self
        capture = cv2.VideoCapture(self.index)
        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"No camera available on index {self.index}. "
                "Use a saved image instead, or pass --camera to pick another device."
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture = capture
        return self

    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    # -- capture -------------------------------------------------------
    def read(self) -> np.ndarray:
        """Grab one frame. Warms the sensor up first to avoid a dark first frame."""
        if self._capture is None:
            self.open()
        assert self._capture is not None
        for _ in range(3):  # auto-exposure warm-up
            self._capture.read()
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraError("The camera did not return a frame.")
        return frame

    def snapshot(self, path: "str | Path") -> Path:
        return save_image(path, self.read())

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    # -- context manager ----------------------------------------------
    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.release()