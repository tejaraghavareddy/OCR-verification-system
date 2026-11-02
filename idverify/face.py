"""Offline face detection, embedding and 1:1 matching (YuNet + SFace).

Both models are plain ONNX files vendored in ``./models``; OpenCV's DNN module
never contacts the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence, Union

import cv2
import numpy as np

from . import config
from .util import load_image

ImageSource = Union[str, "Path", np.ndarray]


class FaceError(RuntimeError):
    """Base class for face-pipeline failures."""


class ModelMissingError(FaceError):
    """A vendored ONNX model is not present on disk."""


class NoFaceError(FaceError):
    """No face could be detected in the supplied image."""


class MultipleFacesError(FaceError):
    """More than one face where exactly one is required."""


class FaceQualityError(FaceError):
    """A face was detected but the capture quality is too low to trust."""


@dataclass(frozen=True)
class DetectedFace:
    """One face box in image coordinates."""

    x: int
    y: int
    w: int
    h: int
    score: float
    row: np.ndarray = field(repr=False, compare=False, default=None)

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


@dataclass
class FaceEmbedding:
    """A 128-D SFace descriptor plus the face it came from."""

    vector: np.ndarray
    face: DetectedFace
    face_count: int = 1

    @property
    def detection_score(self) -> float:
        return self.face.score


def _row_to_face(row: np.ndarray) -> DetectedFace:
    x, y, w, h = (int(round(v)) for v in row[:4])
    return DetectedFace(x=x, y=y, w=w, h=h, score=float(row[-1]), row=np.asarray(row))


class FaceEngine:
    """Detection + recognition with explicit quality gates."""

    def __init__(
        self,
        detector_path: "str | Path | None" = None,
        recognizer_path: "str | Path | None" = None,
        min_detection_score: float = config.MIN_DETECTION_SCORE,
        min_face_px: int = config.MIN_FACE_PX,
    ) -> None:
        models = config.models_dir()
        self.detector_path = Path(detector_path or models / config.DETECTOR_FILE)
        self.recognizer_path = Path(recognizer_path or models / config.RECOGNIZER_FILE)
        for path in (self.detector_path, self.recognizer_path):
            if not path.is_file():
                raise ModelMissingError(
                    f"Missing face model: {path}. Run tools/fetch_models.py once on a "
                    f"networked machine, or set {config.ENV_MODELS} to the directory "
                    f"holding {config.DETECTOR_FILE} and {config.RECOGNIZER_FILE}."
                )
        self.min_detection_score = min_detection_score
        self.min_face_px = min_face_px
        # (320, 320) is only the initial input size; setInputSize is called per image.
        self._detector = cv2.FaceDetectorYN.create(
            str(self.detector_path), "", (320, 320), 0.6, 0.3, 5000
        )
        self._recognizer = cv2.FaceRecognizerSF.create(str(self.recognizer_path), "")

    # -- detection ---------------------------------------------------------
    def detect(self, image: ImageSource) -> list[DetectedFace]:
        """All faces in the image, highest detection score first."""
        img = load_image(image)
        height, width = img.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(img)
        if faces is None:
            return []
        return sorted((_row_to_face(row) for row in faces), key=lambda f: (-f.score, -f.area))

    # -- embedding ---------------------------------------------------------
    def embed(
        self,
        image: ImageSource,
        *,
        single: bool = False,
        min_face_px: int | None = None,
        min_detection_score: float | None = None,
    ) -> FaceEmbedding:
        """Detect the primary face and return its descriptor.

        ``single=True`` fails when more than one face is in frame - used for the
        live capture, where a second person in shot must not be accepted.
        """
        img = load_image(image)
        faces = self.detect(img)
        if not faces:
            raise NoFaceError("No face detected in the image.")
        if single and len(faces) > 1:
            raise MultipleFacesError(
                f"{len(faces)} faces detected; exactly one is required for verification."
            )

        min_px = self.min_face_px if min_face_px is None else min_face_px
        min_score = (
            self.min_detection_score if min_detection_score is None else min_detection_score
        )
        face = faces[0]
        if face.score < min_score:
            raise FaceQualityError(
                f"Face detection confidence {face.score:.2f} is below the required "
                f"{min_score:.2f}; move closer or improve the lighting."
            )
        if min(face.w, face.h) < min_px:
            raise FaceQualityError(
                f"Face is {min(face.w, face.h)}px, below the required {min_px}px."
            )
        return FaceEmbedding(
            vector=self._feature(img, face), face=face, face_count=len(faces)
        )

    def _feature(self, img: np.ndarray, face: DetectedFace) -> np.ndarray:
        if face.row is None:
            raise FaceError("Face row missing; cannot align the crop.")
        aligned = self._recognizer.alignCrop(img, face.row)
        return np.asarray(self._recognizer.feature(aligned), dtype=np.float32).reshape(-1)

    # -- matching ----------------------------------------------------------
    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity in [-1, 1]; higher means more alike."""
        va = np.asarray(a, dtype=np.float32).ravel()
        vb = np.asarray(b, dtype=np.float32).ravel()
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom == 0.0:
            return 0.0
        return float(np.dot(va, vb) / denom)

    def compare(self, a: np.ndarray, b: np.ndarray) -> str:
        """Map a similarity onto ACCEPT / REVIEW / REJECT using config thresholds."""
        score = self.similarity(a, b)
        if score >= config.FACE_ACCEPT_COSINE:
            return "accept"
        if score < config.FACE_REJECT_COSINE:
            return "reject"
        return "review"

    def live_liveness(self, frames: list[np.ndarray], min_score_delta: float = 0.10) -> None:
        """Simple liveness check: two consecutive frames must show different
        face detection scores; a static photo can't produce changing scores.

        Raises FaceQualityError if the confidence delta is below the threshold,
        suggesting the face is from a printed photo or screen.
        """
        if len(frames) < 2:
            return
        # detect faces in both frames
        detections = [self.detect(f) for f in frames]
        # if no face in either frame, that's a failure
        if not detections[0] or not detections[1]:
            raise FaceQualityError(
                "Liveness check failed: no face detected in one or both frames."
            )
        # compare the detection confidence scores (higher = more likely a real face)
        score0 = detections[0][0].score
        score1 = detections[1][0].score
        delta = abs(score0 - score1)
        if delta < min_score_delta:
            raise FaceQualityError(
                f"Liveness check failed: face confidence delta {delta:.3f} below "
                f"{min_score_delta:.2f}; the face may be from a static photo or screen."
            )

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def crop(image: ImageSource, face: DetectedFace, pad: float = 0.12) -> np.ndarray:
        """Crop a face box out of the image with a little breathing room."""
        img = load_image(image)
        height, width = img.shape[:2]
        dx, dy = int(face.w * pad), int(face.h * pad)
        x0 = max(0, face.x - dx)
        y0 = max(0, face.y - dy)
        x1 = min(width, face.x + face.w + dx)
        y1 = min(height, face.y + face.h + dy)
        if x1 <= x0 or y1 <= y0:
            return img
        return np.ascontiguousarray(img[y0:y1, x0:x1])

    @staticmethod
    def cosine_all(reference: np.ndarray, others: Iterable[np.ndarray]) -> list[float]:
        return [FaceEngine.similarity(reference, other) for other in others]

    @staticmethod
    def best_match(
        reference: np.ndarray, candidates: Sequence[np.ndarray]
    ) -> tuple[int, float]:
        """Index and score of the closest candidate, or ``(-1, -1.0)`` when empty."""
        if not candidates:
            return -1, -1.0
        scores = [FaceEngine.similarity(reference, other) for other in candidates]
        index = max(range(len(scores)), key=scores.__getitem__)
        return index, scores[index]