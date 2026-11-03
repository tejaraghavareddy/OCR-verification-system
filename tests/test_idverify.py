"""End-to-end tests: OCR field extraction, face matching, enrolment, audit log."""

from __future__ import annotations

import pytest

from idverify import config
from idverify.engine import REVIEW, REJECTED, VERIFIED, EnrollError
from idverify.face import FaceError, NoFaceError, FaceQualityError
from idverify.idcard import (
    CardReader,
    _match_label,
    is_expired,
    parse_date,
)
from idverify.store import Person, Store
from tests.conftest import BIDEN, OBAMA, build_card


# --------------------------------------------------------------------------- #
# Field parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,expected",
    [
        ("04 AUG 1961", "1961-08-04"),
        ("12 MAY 2032", "2032-05-12"),
        ("1961-08-04", "1961-08-04"),
        ("31/12/1999", "1999-12-31"),
        ("13/05/1970", "1970-05-13"),  # day > 12 disambiguates
        ("SEP 9, 2001", "2001-09-09"),
        ("O4 AUG 1961", "1961-08-04"),  # OCR glyph swap
        ("not a date", None),
        ("", None),
    ],
)
def test_parse_date(text, expected):
    assert parse_date(text) == expected


@pytest.mark.parametrize(
    "line,field",
    [
        ("NAME:", "full_name"),
        ("FULL NAME:", "full_name"),
        ("ID NUMBER:", "id_number"),
        ("ID NO:", "id_number"),
        ("PASSPORT NO:", "id_number"),
        ("DATE OF BIRTH:", "date_of_birth"),
        ("DOB:", "date_of_birth"),
        ("DATE OF EXPIRY:", "date_of_expiry"),
        ("EXPIRY:", "date_of_expiry"),
        ("SEX:", "sex"),
        ("NATIONALITY:", "nationality"),
    ],
)
def test_match_label(line, field):
    assert _match_label(line)[0] == field


@pytest.mark.parametrize(
    "line", ["REPUBLIC IDENTITY CARD", "IDENTITY CARD", "BARACK H. OBAMA", "4471-9938-2210"]
)
def test_match_label_ignores_non_labels(line):
    assert _match_label(line) == (None, None)


def test_match_label_splits_inline_value():
    assert _match_label("NAME: JOHN DOE") == ("full_name", "JOHN DOE")


def test_is_expired():
    assert is_expired("2020-01-01") is True
    assert is_expired("2999-12-31") is False
    assert is_expired(None) is None


# --------------------------------------------------------------------------- #
# Card reading
# --------------------------------------------------------------------------- #
def test_scan_reads_every_field(card_image, face_engine):
    scan = CardReader(face_engine).scan(card_image)
    fields = scan.fields
    assert fields.full_name == "Sample Student"
    assert fields.id_number == "TEST-001-ENROLL"
    assert fields.date_of_birth == "2000-01-01"
    assert fields.date_of_expiry == "2030-12-31"
    assert fields.sex == "F"
    assert fields.nationality == "LOCAL"
    assert fields.missing() == []
    assert scan.found_portrait
    assert not scan.warnings


def test_scan_extracts_the_printed_portrait(card_image, face_engine):
    scan = CardReader(face_engine).scan(card_image)
    assert scan.portrait is not None
    # The crop is a real face the detector can find again. A printed portrait is
    # small, so it is embedded with the card-side gates the production path uses
    # rather than the stricter live-capture ones.
    embedded = face_engine.embed(
        scan.portrait,
        min_face_px=config.MIN_CARD_FACE_PX,
        min_detection_score=0.5,
    )
    assert embedded.face.score > 0.6


def test_scan_reports_missing_text(tmp_path, face_engine):
    import numpy as np

    blank = tmp_path / "blank.png"
    image = np.full((400, 600, 3), 245, dtype="uint8")
    import cv2

    cv2.imwrite(str(blank), image)
    scan = CardReader(face_engine).scan(blank)
    assert scan.fields.id_number is None
    assert scan.warnings


# --------------------------------------------------------------------------- #
# Face engine
# --------------------------------------------------------------------------- #
def test_similarity_separates_same_and_different(face_engine):
    same_a = face_engine.embed(OBAMA)
    same_b = face_engine.embed(OBAMA)
    other = face_engine.embed(BIDEN)
    assert face_engine.similarity(same_a.vector, same_b.vector) > config.FACE_ACCEPT_COSINE
    assert face_engine.similarity(same_a.vector, other.vector) < config.FACE_REJECT_COSINE


def test_embed_rejects_images_without_faces(tmp_path):
    import cv2
    import numpy as np

    from idverify.face import FaceEngine

    path = tmp_path / "empty.png"
    cv2.imwrite(str(path), np.full((300, 300, 3), 200, dtype="uint8"))
    engine = FaceEngine()
    with pytest.raises(NoFaceError):
        engine.embed(path)


def test_embed_requires_single_face_for_live_capture(face_engine, tmp_path):
    import cv2
    import numpy as np

    left = cv2.imread(str(OBAMA))
    right = cv2.imread(str(BIDEN))
    height = max(left.shape[0], right.shape[0])
    canvas = np.zeros((height, left.shape[1] + right.shape[1], 3), dtype="uint8")
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1]:] = right
    both = tmp_path / "two_faces.png"
    cv2.imwrite(str(both), canvas)

    assert len(face_engine.detect(both)) >= 2
    with pytest.raises(FaceError):
        face_engine.embed(both, single=True)


def test_embed_enforces_quality_gates(face_engine, tmp_path):
    import cv2
    import numpy as np

    tiny = tmp_path / "tiny.png"
    crop = cv2.resize(cv2.imread(str(OBAMA)), (40, 40))
    cv2.imwrite(str(tiny), crop)
    with pytest.raises((FaceQualityError, NoFaceError)):
        face_engine.embed(tiny, min_face_px=80)


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
def test_store_round_trip(tmp_path):
    store = Store(tmp_path / "home")
    person = Person(id_number="TEST-X1", full_name="Ada", embedding=[0.1, 0.2, 0.3])
    store.upsert_person(person)
    loaded = store.get_person("TEST-X1")
    assert loaded.full_name == "Ada"
    assert loaded.embedding.shape == (3,)

    person.full_name = "Ada L."
    store.upsert_person(person)
    assert store.count_persons() == 1
    assert store.get_person("TEST-X1").full_name == "Ada L."

    assert store.delete_person("TEST-X1") is True
    assert store.get_person("TEST-X1") is None  # deleted


# --------------------------------------------------------------------------- #
# Enrolment + verification
# --------------------------------------------------------------------------- #
def test_enroll_then_verify_same_person(system, card_image):
    result = system.enroll(card_image, OBAMA)
    assert result.id_number == "TEST-001-ENROLL"
    assert result.live_vs_card > config.FACE_ACCEPT_COSINE
    assert result.updated is False

    check = system.verify(card_image, OBAMA)
    assert check.verdict == VERIFIED
    assert check.enrolled is True
    assert check.live_vs_enrolled > config.FACE_ACCEPT_COSINE
    assert check.basis == "card portrait + local record"


def test_enroll_refuses_a_face_that_does_not_match(system, make_card):
    card = make_card("mismatch.png")
    with pytest.raises(EnrollError):
        system.enroll(card, BIDEN)
    assert system.count_persons() == 0  # nothing was stored


def test_verify_rejects_an_impostor(system, card_image):
    system.enroll(card_image, OBAMA)
    check = system.verify(card_image, BIDEN)
    assert check.verdict == REJECTED
    assert check.live_vs_card < config.FACE_REJECT_COSINE
    assert any("printed portrait" in reason for reason in check.reasons)


def test_verify_without_a_record_checks_the_card_only(system, card_image):
    check = system.verify(card_image, OBAMA)
    assert check.enrolled is False
    assert check.live_vs_card > config.FACE_ACCEPT_COSINE
    assert check.verdict == VERIFIED
    assert any("not in the local database" in reason for reason in check.reasons)


def test_verify_rejects_an_expired_card(system, make_card):
    card = make_card("expired.png", date_of_expiry="01 JAN 2020")
    check = system.verify(card, OBAMA)
    assert check.card_expired is True
    assert check.verdict == REJECTED
    assert any("expired" in reason for reason in check.reasons)


def test_every_verification_is_logged(system, card_image):
    system.verify(card_image, OBAMA)
    system.verify(card_image, BIDEN)
    entries = system.history()
    assert len(entries) == 2
    assert {entry.result for entry in entries} == {VERIFIED, REJECTED}
    assert entries[0].created_at and entries[0].id_number == "TEST-001-ENROLL"  # same as card_image fixture


def test_forget_removes_the_record(system, card_image):
    system.enroll(card_image, OBAMA)
    assert system.forget("TEST-001-ENROLL") is True
    assert system.people() == []


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_verify_exit_codes(tmp_path, card_image, capsys):
    from idverify.cli import main

    home = str(tmp_path / "cli-home")
    assert main(["--home", home, "enroll", "--card", str(card_image), "--face", str(OBAMA)]) == 0
    assert main(["--home", home, "verify", "--card", str(card_image), "--face", str(OBAMA)]) == 0
    assert main(["--home", home, "verify", "--card", str(card_image), "--face", str(BIDEN)]) == 1
    assert main(["--home", home, "scan", "--card", str(card_image)]) == 0
    assert main(["--home", home, "list"]) == 0
    assert main(["--home", home, "history"]) == 0
    out = capsys.readouterr().out
    assert "TEST-001-ENROLL" in out


def test_cli_reports_missing_images(tmp_path, card_image):
    from idverify.cli import main

    home = str(tmp_path / "cli-missing")
    assert main(["--home", home, "verify", "--card", str(card_image), "--face", "nope.jpg"]) == 3