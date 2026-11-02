"""ID card reading: OCR of the printed fields plus extraction of the portrait.

The OCR engine is RapidOCR (ONNX Runtime). Its three models are packaged inside
the wheel, so ``RapidOCR()`` never reaches out to the network.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field as dc_field
from typing import Optional

import numpy as np

from . import config
from .face import DetectedFace, FaceEngine, FaceError
from .util import load_image

# --------------------------------------------------------------------------- #
# OCR engine (lazy, shared - it is stateless and costs ~0.5s to construct)
# --------------------------------------------------------------------------- #
_ocr_engine = None


def get_ocr():
    """Return the shared RapidOCR instance, constructing it on first use."""
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr import RapidOCR

        _ocr_engine = RapidOCR()
    return _ocr_engine


# --------------------------------------------------------------------------- #
# Field labels
# --------------------------------------------------------------------------- #
# Most-specific rules first: "DATE OF BIRTH" must win over the generic "NAME"
# family, and "ID NUMBER" must win over a bare "ID".
# Alternatives are written longest-first and the matcher appends a
# "not followed by a letter or digit" guard, so `NO` inside `NUMBER` backtracks
# to the full word instead of silently leaving `BER:` behind as the value.
LABEL_RULES: list[tuple[str, list[str]]] = [
    ("date_of_birth", [r"DATE\s*OF\s*BIRTH", r"BIRTH\s*DATE", r"\bDOB\b", r"BORN\s*ON"]),
    ("date_of_expiry", [
        r"DATE\s*OF\s*EXPIR[A-Z]*", r"EXPIRY\s*DATE", r"EXPIR[A-Z]*",
        r"VALID\s*(THRU|THROUGH|UNTIL)", r"REPLACE\s*BY", r"\bEXP\b",
    ]),
    ("id_number", [
        r"ID\s*(CARD\s*)?(NUMBER|NUMB|NUM|NO|N|#)",
        r"IDENTITY\s*(NUMBER|NO)[A-Z]*",
        r"NATIONAL\s*ID(\s*(NUMBER|NO))?", r"\bNID\b",
        r"CITIZEN[A-Z]*\s*(NUMBER|NO|#)", r"PASSPORT\s*(NUMBER|NO|#)",
        r"DOCUMENT\s*(NUMBER|NO)", r"\bSSN\b", r"\bDL\s*NO\b", r"\bID\b",
    ]),
    ("full_name", [r"FULL\s*NAME", r"NAME", r"SURNAME", r"GIVEN\s*NAME", r"HOLDER", r"BEARER"]),
    ("sex", [r"\bSEX\b", r"GENDER"]),
    ("nationality", [r"NATIONALITY", r"COUNTRY\s*OF\s*CITIZENSHIP"]),
]

FIELD_LABELS = {
    "full_name": "Name",
    "id_number": "ID number",
    "date_of_birth": "Date of birth",
    "date_of_expiry": "Date of expiry",
    "sex": "Sex",
    "nationality": "Nationality",
}

MONTHS = {
    "JAN": 1, "JANUARY": 1, "FEB": 2, "FEBRUARY": 2, "MAR": 3, "MARCH": 3,
    "APR": 4, "APRIL": 4, "MAY": 5, "JUN": 6, "JUNE": 6, "JUL": 7, "JULY": 7,
    "AUG": 8, "AUGUST": 8, "SEP": 9, "SEPT": 9, "SEPTEMBER": 9, "OCT": 10,
    "OCTOBER": 10, "NOV": 11, "NOVEMBER": 11, "DEC": 12, "DECEMBER": 12,
}

# OCR engines routinely swap glyphs inside numeric fields.
_DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "B": "8", "G": "6", "Z": "2"})


def _fix_digits(token: str) -> str:
    return token.translate(_DIGIT_FIX)


def parse_date(text: str) -> Optional[str]:
    """Parse a printed date into ``YYYY-MM-DD``; ``None`` when unparseable.

    Day-first is assumed for all-numeric dates (the common convention on
    national ID cards); an unambiguous day/month pair overrides that.
    """
    if not text:
        return None
    raw = re.sub(r"\s+", " ", text).strip().upper()

    # 1961-08-04 / 1961.08.04
    m = re.search(r"\b(19\d{2}|20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b", raw)
    if m:
        return _iso(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    # Digits are matched loosely so an OCR glyph swap ("O4" for "04") still
    # parses; month names stay strict so "AUG" is never mangled.
    day = r"[0-9OIlSBGZol]{1,2}"

    # 04 AUG 1961  |  04-AUG-61
    m = re.search(
        r"\b(" + day + r")[-\s/](" + "|".join(MONTHS) + r")\.?[-\s/,]+(\d{2,4})\b",
        raw,
    )
    if m:
        return _iso(_year(m.group(3)), MONTHS[m.group(2)], int(_fix_digits(m.group(1))))

    # AUG 04, 1961
    m = re.search(
        r"\b(" + "|".join(MONTHS) + r")\.?[-\s/]+(" + day + r"),?[-\s/]+(\d{2,4})\b",
        raw,
    )
    if m:
        return _iso(_year(m.group(3)), MONTHS[m.group(1)], int(_fix_digits(m.group(2))))

    # 04/08/1961 or 04-08-61
    m = re.search(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})\b", raw)
    if m:
        first = int(_fix_digits(m.group(1)))
        second = int(_fix_digits(m.group(2)))
        if first > 12 >= second:
            day, month = first, second
        elif second > 12 >= first:
            month, day = first, second
        else:
            day, month = first, second  # day-first
        return _iso(_year(m.group(3)), month, day)

    return None


def _year(token: str) -> int:
    token = _fix_digits(token)
    if len(token) == 2:
        value = int(token)
        return 1900 + value if value >= 26 else 2000 + value
    return int(token)


def _iso(year: int, month: int, day: int) -> Optional[str]:
    if not (1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2999):
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def is_expired(date_iso: Optional[str], today_iso: Optional[str] = None) -> Optional[bool]:
    """``True`` when the card has expired, ``None`` when it cannot be judged."""
    if not date_iso:
        return None
    today = today_iso or time.strftime("%Y-%m-%d")
    return date_iso < today


# --------------------------------------------------------------------------- #
# Structured results
# --------------------------------------------------------------------------- #
@dataclass
class CardFields:
    """The identity data read off the card."""

    full_name: Optional[str] = None
    id_number: Optional[str] = None
    date_of_birth: Optional[str] = None
    date_of_expiry: Optional[str] = None
    sex: Optional[str] = None
    nationality: Optional[str] = None
    raw_lines: list[str] = dc_field(default_factory=list)

    def as_dict(self) -> dict:
        return {key: getattr(self, key) for key in FIELD_LABELS}

    def missing(self) -> list[str]:
        return [
            FIELD_LABELS[key]
            for key in ("full_name", "id_number")
            if not getattr(self, key)
        ]

    def label(self, key: str) -> str:
        value = getattr(self, key)
        return value or "-"


@dataclass
class CardScan:
    """Everything one pass over a card image produced."""

    fields: CardFields
    portrait: Optional[np.ndarray] = None
    portrait_face: Optional[DetectedFace] = None
    faces: list[DetectedFace] = dc_field(default_factory=list)
    warnings: list[str] = dc_field(default_factory=list)
    elapsed: float = 0.0

    @property
    def found_portrait(self) -> bool:
        return self.portrait is not None


# --------------------------------------------------------------------------- #
# Reading order
# --------------------------------------------------------------------------- #
def _reading_order(entries: list[tuple[str, float, tuple]]) -> list[str]:
    """Sort OCR boxes into human reading order (rows top-down, then left-right)."""
    prepared = []
    for text, score, box in entries:
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        prepared.append(
            (sum(ys) / len(ys), sum(xs) / len(xs), max(ys) - min(ys), text, score)
        )
    prepared.sort(key=lambda item: (item[0], item[1]))

    rows: list[list[tuple]] = []
    for item in prepared:
        centre_y, _, height, _, _ = item
        tolerance = max(height, 8.0) * 0.6
        if rows and abs(centre_y - rows[-1][0][0]) <= tolerance:
            rows[-1].append(item)
        else:
            rows.append([item])

    ordered: list[str] = []
    for row in rows:
        for item in sorted(row, key=lambda entry: entry[1]):
            text = re.sub(r"\s+", " ", item[3]).strip()
            if text:
                ordered.append(text)
    return ordered


def _match_label(line: str) -> tuple[Optional[str], Optional[str]]:
    """Return ``(field, inline_value)`` for a label line; ``(None, None)`` if it is not one.

    ``inline_value`` is ``''`` when the value sits on the following line.
    """
    cleaned = re.sub(r"^\s*\d{1,2}[\.\)]\s*", "", line).strip()
    for field_name, patterns in LABEL_RULES:
        for pattern in patterns:
            match = re.match(
                r"\s*(?:" + pattern + r")(?![A-Za-z0-9])\s*", cleaned, re.IGNORECASE
            )
            if not match:
                continue
            tail = cleaned[match.end():]
            tail = re.sub(r"^[\s:;,\-–—#()\[\]]+\s*", "", tail).strip()
            return field_name, tail
    return None, None


def _looks_like_value(text: str) -> bool:
    """A candidate field value must contain something beyond punctuation."""
    return bool(re.search(r"[A-Za-z0-9]", text))


_ID_FALLBACK = re.compile(r"\b(?=[A-Z0-9\-]{6,}\b)[A-Z]{0,2}\d[\d\-]{4,17}\d\b")
_ALPHA_NAME = re.compile(r"^[A-Za-z][A-Za-z'’.\-]+(?:\s+[A-Za-z][A-Za-z'’.\-]+){1,}$")


class CardReader:
    """OCR + portrait extraction for a single ID card image."""

    def __init__(self, face_engine: Optional[FaceEngine] = None, ocr=None) -> None:
        self.face_engine = face_engine or FaceEngine()
        self._ocr = ocr

    def _run_ocr(self, image: np.ndarray):
        engine = self._ocr or get_ocr()
        return engine(image)

    # ------------------------------------------------------------------ #
    def scan(self, image) -> CardScan:
        started = time.time()
        img = load_image(image)
        warnings: list[str] = []

        result = self._run_ocr(img)
        entries: list[tuple[str, float, tuple]] = []
        if result is not None:
            boxes = result.boxes if result.boxes is not None else []
            texts = result.txts or []
            scores = result.scores or []
            for index, text in enumerate(texts):
                score = float(scores[index]) if index < len(scores) else 1.0
                if score < config.OCR_MIN_LINE_SCORE or not str(text).strip():
                    continue
                box = boxes[index] if index < len(boxes) else None
                if box is None:
                    entries.append((str(text), score, ((0, index * 10), (1, index * 10), (1, index * 10 + 8), (0, index * 10 + 8))))
                else:
                    entries.append((str(text), score, box))

        lines = _reading_order(entries)
        fields = self._extract_fields(lines)
        fields.raw_lines = lines
        if not lines:
            warnings.append("No text could be read from the card.")
        if not fields.id_number:
            warnings.append("ID number could not be read.")
        if not fields.full_name:
            warnings.append("Name could not be read.")
        low_confidence = [score for _, score, _ in entries if score < config.OCR_MIN_SCORE]
        if entries and len(low_confidence) > len(entries) / 2:
            warnings.append("OCR confidence is low; check the captured image.")

        portrait, portrait_face, faces = self._extract_portrait(img, warnings)

        return CardScan(
            fields=fields,
            portrait=portrait,
            portrait_face=portrait_face,
            faces=faces,
            warnings=warnings,
            elapsed=time.time() - started,
        )

    # ------------------------------------------------------------------ #
    def _extract_fields(self, lines: list[str]) -> CardFields:
        fields = CardFields()
        values: dict[str, str] = {}
        pending: Optional[str] = None
        consumed: set[str] = set()

        for line in lines:
            label, inline = _match_label(line)
            if label is not None:
                if inline and _looks_like_value(inline):
                    if label not in values:
                        values[label] = inline
                        consumed.add(line)
                    pending = None
                else:
                    pending = label
                continue
            if pending and _looks_like_value(line):
                values.setdefault(pending, line)
                consumed.add(line)
                pending = None

        for key in ("date_of_birth", "date_of_expiry", "id_number", "full_name", "sex", "nationality"):
            if key not in values:
                fallback = self._fallback_for(key, lines, consumed)
                if fallback:
                    values[key] = fallback
                    consumed.add(fallback)

        fields.full_name = _clean_name(values.get("full_name"))
        fields.id_number = _clean_id(values.get("id_number"))
        fields.date_of_birth = parse_date(values.get("date_of_birth", ""))
        if not fields.date_of_birth and values.get("date_of_birth"):
            fields.date_of_birth = None
        fields.date_of_expiry = parse_date(values.get("date_of_expiry", ""))
        sex = values.get("sex")
        fields.sex = sex[:12] if sex else None
        fields.nationality = values.get("nationality")
        return fields

    @staticmethod
    def _fallback_for(key: str, lines: list[str], consumed: set[str]) -> Optional[str]:
        candidates = [line for line in lines if line not in consumed]
        if key == "id_number":
            for line in candidates:
                if _match_label(line)[0] is not None:
                    continue
                match = _ID_FALLBACK.search(line)
                if match and any(ch.isdigit() for ch in match.group(0)):
                    return match.group(0)
            return None
        if key in ("date_of_birth", "date_of_expiry"):
            for line in candidates:
                if _match_label(line)[0] is not None:
                    continue
                if parse_date(line):
                    return line
            return None
        if key == "full_name":
            for line in candidates:
                if _match_label(line)[0] is not None:
                    continue
                if _ALPHA_NAME.match(line) and not parse_date(line):
                    return line
            return None
        return None

    # ------------------------------------------------------------------ #
    def _extract_portrait(
        self, img: np.ndarray, warnings: list[str]
    ) -> tuple[Optional[np.ndarray], Optional[DetectedFace], list[DetectedFace]]:
        try:
            faces = self.face_engine.detect(img)
        except FaceError:
            faces = []
        if not faces:
            warnings.append("No portrait found on the card image.")
            return None, None, []

        usable = [f for f in faces if min(f.w, f.h) >= config.MIN_CARD_FACE_PX]
        pool = usable or faces
        # Highest detection score wins (the printed portrait is the clear,
        # frontal face); area breaks ties.
        portrait_face = max(pool, key=lambda f: (f.score, f.area))
        portrait = self.face_engine.crop(img, portrait_face)
        if len(faces) > 1:
            warnings.append(
                f"{len(faces)} faces visible in the card image; the clearest one was "
                f"used as the printed portrait. Confirm it in the preview."
            )
        return portrait, portrait_face, faces


def _clean_name(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    text = re.sub(r"\s+", " ", value).strip(" .,:;-")
    text = re.sub(r"^[^A-Za-z]+", "", text)
    return text or None


def _clean_id(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    text = re.sub(r"\s+", "", value).strip(" .,:;-").upper()
    return text or None