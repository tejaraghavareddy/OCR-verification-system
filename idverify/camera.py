"""Webcam capture for IDVerify.

A single OpenCV capture device. Other parts of the system also accept image
files, so a machine without a camera stays fully usable.

Two ways to use a camera:

* One-shot ``read()`` — the CLI path. Opens the device, warms it briefly, and
  returns a single ``uint8 BGR`` frame. Good for ``python main.py attend`` or
  ``python main.py verify --camera 0``.

* Live stream — the GUI path. Call ``start_stream()`` and repeatedly call
  ``stream_frame()`` from a timer or loop until ``stop_stream()``. This keeps
  the device open and feeds a continuous preview. It is the only reliable way
  to show a moving webcam view inside a Tkinter window.
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
    """A single OpenCV capture device, opened lazily.

    Create one per camera index and reuse it while the user is looking at that
    camera. Call ``release()`` once the camera is no longer needed, or use it
    as a context manager for one-shot captures.
    """

    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        warmup: int = 3,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.warmup = max(0, warmup)
        self._capture: Optional[cv2.VideoCapture] = None
        self._streaming = False

    # -- lifecycle -----------------------------------------------------\
    def open(self) -> "Camera":
        """Open the device if it is not already open."""
        if self._capture is not None and self._capture.isOpened():
            return self
        capture = cv2.VideoCapture(self.index)
        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"No camera available on index {self.index}. "
                "Use a saved image instead, or pass a different camera index."
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # Some backends ignore the request; carry on with whatever is available.
        self._capture = capture
        return self

    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    # -- one-shot capture ------------------------------------------------
    def read(self) -> np.ndarray:
        """Grab one frame.

        Warms the sensor briefly so the first frame is not a dark, auto-exposure
        settling frame. Raises :class:`CameraError` when the device is missing
        or stops returning frames.
        """
        if self._capture is None:
            self.open()
        if self._capture is None:
            raise CameraError("The camera was not opened.")
        # Burn a few frames so auto-exposure and white-balance settle.
        for _ in range(self.warmup):
            self._capture.read()
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraError("The camera did not return a frame.")
        return frame

    def snapshot(self, path: "str | Path") -> Path:
        """Save a one-shot frame to disk. Convenience wrapper around ``read()``."""
        return save_image(path, self.read())

    # -- live stream (GUI preview) ---------------------------------------
    def start_stream(self) -> "Camera":
        """Prepare the camera for a continuous preview.

        Call ``stream_frame()`` repeatedly (for example from a Tkinter
        ``after()`` timer) and call ``stop_stream()`` once the preview is no
        longer needed. A stream keeps the device open and reads frames without
        the warm-up loop that one-shot ``read()`` uses.
        """
        self.open()
        self._streaming = True
        # Drain any stale frames so the first preview frame is current.
        for _ in range(self.warmup):
            self._capture.read()
        return self

    def stream_frame(self) -> Optional[np.ndarray]:
        """Return the latest frame during a live stream, or ``None`` on failure.

        When no frame is available for this tick, returns ``None`` and keeps
        the stream alive so the next call can retry.
        """
        if not self._streaming or self._capture is None:
            return None
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        return frame

    def stop_stream(self) -> None:
        """End the live preview. The device stays open for later one-shot reads."""
        self._streaming = False

    # -- release / context manager ---------------------------------------
    def release(self) -> None:
        """Close the device and drop the capture handle."""
        if self._capture is not None:
            # If streaming is still on, turn it off first.
            self._streaming = False
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.release()
