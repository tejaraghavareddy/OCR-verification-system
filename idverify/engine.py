"""The verification system itself: enrolment, 1:1 checks and the audit trail.

Flow for one verification::

    ID card image --OCR--> printed fields + printed portrait
    live capture  --YuNet--> face --SFace--> 128-D descriptor
                            |
                            +-- cosine vs printed portrait  (does this person hold the card?)
                            +-- cosine vs enrolled record   (is this the card we issued?)

Both comparisons are scored against the thresholds in :mod:`idverify.config`,
and every attempt is written to the local audit log.
"""

from __future__ import annotations

import time
import cv2
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Optional

import numpy as np

from . import config
from .face import FaceEmbedding, FaceEngine, FaceError
from .idcard import CardFields, CardReader, CardScan, is_expired
from .store import Person, Store, VerificationLog
from .util import load_image, now_iso, save_image

VERIFIED = "VERIFIED"
REJECTED = "REJECTED"
REVIEW = "REVIEW"

VERDICT_LABELS = {
    VERIFIED: "IDENTITY VERIFIED",
    REJECTED: "VERIFICATION FAILED",
    REVIEW: "NEEDS MANUAL REVIEW",
}


class EnrollError(RuntimeError):
    """Enrolment refused because the evidence did not line up."""


@dataclass
class EnrollResult:
    id_number: str
    full_name: Optional[str]
    live_vs_card: float
    updated: bool
    portrait_path: Optional[str] = None
    card_path: Optional[str] = None
    warnings: list[str] = dc_field(default_factory=list)
    elapsed: float = 0.0


@dataclass
class VerificationResult:
    verdict: str
    card_fields: CardFields
    reasons: list[str] = dc_field(default_factory=list)
    warnings: list[str] = dc_field(default_factory=list)
    live_vs_card: Optional[float] = None
    live_vs_enrolled: Optional[float] = None
    enrolled: bool = False
    card_expired: Optional[bool] = None
    person: Optional[Person] = None
    created_at: str = ""
    elapsed: float = 0.0

    @property
    def primary_score(self) -> Optional[float]:
        """The score used for the headline decision."""
        if self.live_vs_enrolled is not None:
            return self.live_vs_enrolled
        return self.live_vs_card

    @property
    def basis(self) -> str:
        if self.enrolled:
            return "card portrait + local record"
        return "card portrait only (ID not in local records)"

    @property
    def ok(self) -> bool:
        return self.verdict == VERIFIED

    @staticmethod
    def _band(score: Optional[float]) -> Optional[str]:
        if score is None:
            return None
        if score >= config.FACE_ACCEPT_COSINE:
            return "accept"
        if score < config.FACE_REJECT_COSINE:
            return "reject"
        return "review"

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "id_number": self.card_fields.id_number,
            "full_name": self.card_fields.full_name,
            "live_vs_card": self.live_vs_card,
            "live_vs_enrolled": self.live_vs_enrolled,
            "enrolled": self.enrolled,
            "card_expired": self.card_expired,
            "reasons": self.reasons,
            "warnings": self.warnings,
            "created_at": self.created_at,
        }


class IdentitySystem:
    """Enrol and verify identities entirely on this machine."""

    def __init__(
        self,
        home: "str | Path | None" = None,
        face_engine: Optional[FaceEngine] = None,
        card_reader: Optional[CardReader] = None,
        store: Optional[Store] = None,
    ) -> None:
        self.home = Path(home) if home else config.home_dir()
        self.face = face_engine or FaceEngine()
        self.reader = card_reader or CardReader(self.face)
        self.store = store or Store(self.home)

    # ------------------------------------------------------------------ #
    # Enrolment
    # ------------------------------------------------------------------ #
    def enroll(self, card_image, live_image) -> EnrollResult:
        """Register an identity from a card scan plus a live capture.

        Refuses to store anything when the live face does not clearly match the
        portrait printed on the card.
        """
        started = time.time()
        scan = self.reader.scan(card_image)
        fields = scan.fields
        id_number = fields.id_number
        if not id_number:
            raise EnrollError(
                "Could not read an ID number from the card: "
                + (", ".join(scan.warnings) or "no text found")
            )
        if scan.portrait is None:
            raise EnrollError("No portrait could be extracted from the card image.")

        try:
            live = self.face.embed(
                load_image(live_image), single=True, min_face_px=config.MIN_FACE_PX
            )
        except FaceError as exc:
            raise EnrollError(str(exc)) from exc

        card_face = self.face.embed(
            scan.portrait, min_face_px=config.MIN_CARD_FACE_PX, min_detection_score=0.5
        )
        score = self.face.similarity(live.vector, card_face.vector)
        if score < config.FACE_ACCEPT_COSINE:
            raise EnrollError(
                f"The live face does not match the portrait on the card "
                f"(similarity {score:.3f}, needs {config.FACE_ACCEPT_COSINE:.2f})."
            )

        stamp = time.strftime("%Y%m%d-%H%M%S")
        stem = f"{_safe(id_number)}-{stamp}"
        portrait_path = save_image(
            self.store.images_dir / f"{stem}-portrait.png", scan.portrait
        )
        face_path = save_image(
            self.store.images_dir / f"{stem}-face.png",
            self.face.crop(load_image(live_image), live.face),
        )
        card_path = save_image(self.store.images_dir / f"{stem}-card.png", load_image(card_image))

        person = Person(
            id_number=id_number,
            full_name=fields.full_name,
            date_of_birth=fields.date_of_birth,
            date_of_expiry=fields.date_of_expiry,
            sex=fields.sex,
            nationality=fields.nationality,
            embedding=live.vector,
            card_embedding=card_face.vector,
            portrait_path=str(portrait_path),
            card_path=str(card_path),
        )
        existing = self.store.get_person(id_number)
        self.store.upsert_person(person)

        return EnrollResult(
            id_number=id_number,
            full_name=fields.full_name,
            live_vs_card=score,
            updated=existing is not None,
            portrait_path=str(portrait_path),
            card_path=str(card_path),
            warnings=list(scan.warnings),
            elapsed=time.time() - started,
        )

    # ------------------------------------------------------------------ #
    # Verification
    # ------------------------------------------------------------------ #
    def verify(self, card_image, live_image) -> VerificationResult:
        """Check a live capture against the card in hand and the local record."""
        started = time.time()
        created_at = now_iso()
        scan = self.reader.scan(card_image)
        fields = scan.fields
        reasons: list[str] = []
        warnings: list[str] = list(scan.warnings)

        # Keep the original image array for the liveness check below
        live_img = load_image(live_image)

        live: Optional[FaceEmbedding] = None
        try:
            live = self.face.embed(
                live_img, single=True, min_face_px=config.MIN_FACE_PX
            )
        except FaceError as exc:
            reasons.append(str(exc))

        # ---- liveness check (printed-photo / phone-screen spoof guard) ----
        # Only run when a real webcam is available. Two consecutive frames
        # with different detection confidence strongly suggests a live face
        # (a static photo / screen produces identical confidence both times).
        # On headless CI / this cloud VM the check is skipped.
        live_liveness_frames: list[np.ndarray] = []
        try:
            cap = cv2.VideoCapture(0)
            acquired = []
            for _ in range(2):
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                acquired.append(rgb)
            cap.release()
            if len(acquired) == 2:
                # real two-frame webcam check
                live_liveness_frames = acquired
                self.face.live_liveness(live_liveness_frames, min_score_delta=0.10)
            # else: fewer than 2 frames → skip liveness (headless environment)
        except Exception:
            # No webcam, permission denied, or some other error:
            # skip liveness check; it's a webcam-only feature.
            pass

        card_vector: Optional[np.ndarray] = None
        if scan.portrait is not None and live is not None:
            try:
                card_vector = self.face.embed(
                    scan.portrait,
                    min_face_px=config.MIN_CARD_FACE_PX,
                    min_detection_score=0.5,
                ).vector
            except FaceError as exc:
                warnings.append(f"Printed portrait unusable: {exc}")

        live_vs_card: Optional[float] = None
        if live is not None and card_vector is not None:
            live_vs_card = self.face.similarity(live.vector, card_vector)
            reasons.append(
                f"Live face vs printed portrait: {live_vs_card:.3f} "
                f"({self._describe(live_vs_card)})"
            )
        elif live is not None:
            reasons.append("No usable portrait on the card, so the 1:1 card check was skipped.")

        person = self.store.get_person(fields.id_number) if fields.id_number else None
        live_vs_enrolled: Optional[float] = None
        if person is not None and live is not None and person.embedding is not None:
            live_vs_enrolled = self.face.similarity(live.vector, person.embedding)
            reasons.append(
                f"Live face vs enrolled record: {live_vs_enrolled:.3f} "
                f"({self._describe(live_vs_enrolled)})"
            )
        elif not fields.id_number:
            reasons.append("ID number unreadable, so no local record could be looked up.")
        elif person is None:
            reasons.append("ID number is not in the local database; only the card portrait was checked.")

        expired = is_expired(fields.date_of_expiry)
        if expired:
            reasons.append(f"Card expired on {fields.date_of_expiry}.")
        elif expired is None and not fields.date_of_expiry:
            warnings.append("Expiry date unreadable; card validity could not be checked.")

        verdict = self._decide(live, live_vs_card, live_vs_enrolled, expired)

        result = VerificationResult(
            verdict=verdict,
            card_fields=fields,
            reasons=reasons,
            warnings=warnings,
            live_vs_card=live_vs_card,
            live_vs_enrolled=live_vs_enrolled,
            enrolled=person is not None,
            card_expired=expired,
            person=person,
            created_at=created_at,
            elapsed=time.time() - started,
        )
        self.store.log_verification(
            VerificationLog(
                created_at=created_at,
                result=verdict,
                id_number=fields.id_number,
                full_name=fields.full_name,
                enrolled=result.enrolled,
                live_vs_card=live_vs_card,
                live_vs_enrolled=live_vs_enrolled,
                card_expired=expired,
                reasons=reasons,
                warnings=warnings,
            )
        )
        return result

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def history(self, limit: int = 50):
        return self.store.recent_verifications(limit)

    def people(self):
        return self.store.list_persons()

    def forget(self, id_number: str) -> bool:
        return self.store.delete_person(id_number)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _decide(live, live_vs_card, live_vs_enrolled, expired) -> str:
        if live is None:
            return REJECTED
        if expired:
            return REJECTED
        bands = [
            VerificationResult._band(score)
            for score in (live_vs_card, live_vs_enrolled)
            if score is not None
        ]
        if not bands:
            return REVIEW
        if "reject" in bands:
            return REJECTED
        if "review" in bands:
            return REVIEW
        return VERIFIED

    @staticmethod
    def _describe(score: float) -> str:
        band = VerificationResult._band(score)
        return {"accept": "match", "review": "uncertain", "reject": "no match"}[band]


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)[:48] or "id"