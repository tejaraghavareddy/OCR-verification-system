"""Local persistence: enrolled persons and the verification audit log.

A single SQLite file plus an ``images/`` folder inside the app home directory.
No cloud, no sync - this is the whole database.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Optional

import numpy as np

from . import config
from .util import ensure_dir, now_iso

_SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    id_number        TEXT    NOT NULL UNIQUE,
    full_name        TEXT,
    date_of_birth    TEXT,
    date_of_expiry   TEXT,
    sex              TEXT,
    nationality      TEXT,
    embedding        BLOB    NOT NULL,
    card_embedding   BLOB,
    portrait_path    TEXT,
    card_path        TEXT,
    created_at       TEXT    NOT NULL,
    updated_at       TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS verifications (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at        TEXT    NOT NULL,
    result            TEXT    NOT NULL,
    id_number         TEXT,
    full_name         TEXT,
    enrolled          INTEGER NOT NULL DEFAULT 0,
    live_vs_card      REAL,
    live_vs_enrolled  REAL,
    card_expired      INTEGER,
    reasons           TEXT,
    warnings          TEXT,
    person_id         INTEGER REFERENCES persons(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_persons_id_number ON persons(id_number);
CREATE INDEX IF NOT EXISTS idx_verifications_at  ON verifications(created_at DESC);
"""


@dataclass
class Person:
    """One enrolled identity."""

    id_number: str
    full_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    date_of_expiry: Optional[str] = None
    sex: Optional[str] = None
    nationality: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    card_embedding: Optional[np.ndarray] = None
    portrait_path: Optional[str] = None
    card_path: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""
    row_id: Optional[int] = None

    @property
    def display_name(self) -> str:
        return self.full_name or "(name not read)"


@dataclass
class VerificationLog:
    """One row of the audit trail."""

    created_at: str
    result: str
    id_number: Optional[str]
    full_name: Optional[str]
    enrolled: bool
    live_vs_card: Optional[float]
    live_vs_enrolled: Optional[float]
    card_expired: Optional[bool]
    reasons: list[str] = dc_field(default_factory=list)
    warnings: list[str] = dc_field(default_factory=list)
    row_id: Optional[int] = None


def _pack(vector: Optional[np.ndarray]) -> Optional[bytes]:
    if vector is None:
        return None
    return np.asarray(vector, dtype=np.float32).ravel().tobytes()


def _unpack(blob: Optional[bytes]) -> Optional[np.ndarray]:
    if not blob:
        return None
    return np.frombuffer(blob, dtype=np.float32)


class Store:
    """SQLite-backed store, safe to share across the GUI's worker thread."""

    def __init__(self, home: Optional[Path] = None) -> None:
        self.home = ensure_dir(home or config.home_dir())
        self.images_dir = ensure_dir(self.home / "images")
        self.db_path = self.home / "idverify.db"
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # -- persons --------------------------------------------------------
    def get_person(self, id_number: str) -> Optional[Person]:
        if not id_number:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM persons WHERE id_number = ?", (id_number,)
            ).fetchone()
        return self._row_to_person(row) if row else None

    def list_persons(self) -> list[Person]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM persons ORDER BY updated_at DESC"
            ).fetchall()
        return [self._row_to_person(row) for row in rows]

    def count_persons(self) -> int:
        with self._lock:
            return int(
                self._conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
            )

    def upsert_person(self, person: Person) -> Person:
        """Insert or update by ``id_number``; returns the stored record."""
        stamp = now_iso()
        with self._lock:
            existing = self.get_person(person.id_number)
            if existing is None:
                cursor = self._conn.execute(
                    """
                    INSERT INTO persons (id_number, full_name, date_of_birth, date_of_expiry,
                                         sex, nationality, embedding, card_embedding,
                                         portrait_path, card_path, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        person.id_number, person.full_name, person.date_of_birth,
                        person.date_of_expiry, person.sex, person.nationality,
                        _pack(person.embedding), _pack(person.card_embedding),
                        person.portrait_path, person.card_path, stamp, stamp,
                    ),
                )
                person.row_id = cursor.lastrowid
                person.created_at = stamp
            else:
                self._conn.execute(
                    """
                    UPDATE persons SET full_name = ?, date_of_birth = ?, date_of_expiry = ?,
                                       sex = ?, nationality = ?, embedding = ?, card_embedding = ?,
                                       portrait_path = ?, card_path = ?, updated_at = ?
                    WHERE id_number = ?
                    """,
                    (
                        person.full_name, person.date_of_birth, person.date_of_expiry,
                        person.sex, person.nationality, _pack(person.embedding),
                        _pack(person.card_embedding), person.portrait_path,
                        person.card_path, stamp, person.id_number,
                    ),
                )
                person.row_id = existing.row_id
                person.created_at = existing.created_at
            person.updated_at = stamp
            self._conn.commit()
        return person

    def delete_person(self, id_number: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM persons WHERE id_number = ?", (id_number,)
            )
            self._conn.commit()
        return cursor.rowcount > 0

    # -- audit log ------------------------------------------------------
    def log_verification(self, entry: VerificationLog) -> int:
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO verifications (created_at, result, id_number, full_name, enrolled,
                                           live_vs_card, live_vs_enrolled, card_expired,
                                           reasons, warnings, person_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    entry.created_at, entry.result, entry.id_number, entry.full_name,
                    int(entry.enrolled),
                    None if entry.live_vs_card is None else round(float(entry.live_vs_card), 6),
                    None if entry.live_vs_enrolled is None else round(float(entry.live_vs_enrolled), 6),
                    None if entry.card_expired is None else int(entry.card_expired),
                    json.dumps(entry.reasons), json.dumps(entry.warnings),
                ),
            )
            self._conn.commit()
        entry.row_id = cursor.lastrowid
        return entry.row_id

    def recent_verifications(self, limit: int = 50) -> list[VerificationLog]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM verifications ORDER BY id DESC LIMIT ?", (int(limit),)
            ).fetchall()
        out: list[VerificationLog] = []
        for row in rows:
            out.append(
                VerificationLog(
                    row_id=row["id"],
                    created_at=row["created_at"],
                    result=row["result"],
                    id_number=row["id_number"],
                    full_name=row["full_name"],
                    enrolled=bool(row["enrolled"]),
                    live_vs_card=row["live_vs_card"],
                    live_vs_enrolled=row["live_vs_enrolled"],
                    card_expired=bool(row["card_expired"]) if row["card_expired"] is not None else None,
                    reasons=_loads(row["reasons"]),
                    warnings=_loads(row["warnings"]),
                )
            )
        return out

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- helpers --------------------------------------------------------
    @staticmethod
    def _row_to_person(row: sqlite3.Row) -> Person:
        return Person(
            row_id=row["id"],
            id_number=row["id_number"],
            full_name=row["full_name"],
            date_of_birth=row["date_of_birth"],
            date_of_expiry=row["date_of_expiry"],
            sex=row["sex"],
            nationality=row["nationality"],
            embedding=_unpack(row["embedding"]),
            card_embedding=_unpack(row["card_embedding"]),
            portrait_path=row["portrait_path"],
            card_path=row["card_path"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


def _loads(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return value if isinstance(value, list) else []