"""IDVerify - an offline, system-level identity verification desk.

Reads a government ID card with OCR, extracts the printed portrait, and
verifies the person standing in front of the machine against that portrait
(1:1 face verification) plus the locally enrolled record.

Nothing in this package performs a network call at runtime: OCR models ship
inside the ``rapidocr`` wheel, and the face models are vendored under
``./models``.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]