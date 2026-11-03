"""Tests for the live attendance capture flow and the attendance register."""

from __future__ import annotations

import tempfile
from pathlib import Path
from datetime import date

import pytest

from idverify.store import AttendanceLog, Store
from idverify.attend import _score


def _today_iso() -> str:
    return date.today().isoformat()


def test_attendance_log_round_trip(tmp_path):
    store = Store(tmp_path / "home")

    entry = AttendanceLog(
        created_at=f"{_today_iso()}T10:00:00Z",
        result="VERIFIED",
        id_number="TEST-ID-001",
        full_name="Jane Doe",
        enrolled=True,
        live_vs_card=0.85,
        live_vs_enrolled=0.85,
        card_expired=False,
        reasons=["Live face vs printed portrait: 0.85 (match)"],
        warnings=[],
        card_path=str(tmp_path / "card.png"),
        face_path=str(tmp_path / "face.png"),
    )
    row_id = store.log_attendance(entry)
    assert row_id > 0
    assert entry.row_id == row_id

    rows = store.today_attendances()
    assert len(rows) == 1
    r = rows[0]
    assert r.result == "VERIFIED"
    assert r.id_number == "TEST-ID-001"
    assert r.live_vs_card == pytest.approx(0.85)
    assert r.card_path == str(tmp_path / "card.png")
    assert r.face_path == str(tmp_path / "face.png")

    store.close()


def test_today_attendances_only_returns_today(tmp_path):
    store = Store(tmp_path / "home")

    # row dated today
    today_entry = AttendanceLog(
        created_at=f"{_today_iso()}T10:00:00Z",
        result="VERIFIED",
        id_number="STD-101",
        full_name="Alice Smith",
        enrolled=False,
        live_vs_card=None,
        live_vs_enrolled=None,
        card_expired=None,
        reasons=[],
        warnings=[],
    )
    store.log_attendance(today_entry)

    # row dated yesterday — should not appear in today's register
    yesterday_entry = AttendanceLog(
        created_at=f"{date.fromordinal(date.today().toordinal() - 1).isoformat()}T23:59:59Z",
        result="REJECTED",
        id_number="STD-102",
        full_name="Bob Jones",
        enrolled=False,
        live_vs_card=None,
        live_vs_enrolled=None,
        card_expired=None,
        reasons=[],
        warnings=[],
    )
    store.log_attendance(yesterday_entry)

    rows = store.today_attendances()
    assert len(rows) == 1
    assert rows[0].id_number == "STD-101"

    # clear today
    cleared = store.clear_today_attendances()
    assert cleared == 1
    assert store.today_attendances() == []

    # yesterday row still there for other queries (not exposed by today_*)
    store.close()


def test_score_formatting():
    assert _score(None) == "n/a"
    assert _score(0.935) == "0.935"
    assert _score(0.0) == "0.000"
