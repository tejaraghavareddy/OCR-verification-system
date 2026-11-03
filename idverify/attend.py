"""Attendance and enrollment capture from a live camera.

Usage:
    python main.py attend                 # open camera, capture card + face, record attendance
    python main.py attend --enroll       # same flow but also enrolls the identity into the DB
    python main.py attend --list         # show today's attendance register
    python main.py attend --camera 1     # use a different camera index

The camera is opened on startup and the user captures:
  1. an ID card image  (the camera points at the card)
  2. a live face image (the camera points at the person)

Both images are captured through the same camera so no extra hardware is needed.
Student details are persisted in the local SQLite database (the same store the
CLI and GUI use) plus an ``attendances`` table for the attendance register.

Exit codes mirror the CLI:
    0  verified / attendance recorded
    1  rejected
    2  review
    3  error (camera unavailable, OCR failure, etc.)
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from typing import Optional, Sequence

import cv2
import numpy as np

from idverify import __version__, config
from idverify.camera import Camera, CameraError
from idverify.engine import (
    EnrollError,
    IdentitySystem,
    REVIEW,
    REJECTED,
    VERIFIED,
    VerificationResult,
)
from idverify.face import FaceError, ModelMissingError
from idverify.store import Store, AttendanceLog


today_iso = date.today().isoformat()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="idverify-attend",
        description=(
            "Live camera attendance / enrollment capture. Opens the webcam, "
            "captures an ID card and a live face, then records attendance "
            "(and optionally enrolls the identity) in the local database."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"idverify-attend {__version__}"
    )
    parser.add_argument(
        "--home",
        metavar="DIR",
        default=None,
        help=f"data directory (default: {config.home_dir()})",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        metavar="INDEX",
        help="camera index to use (default: 0)",
    )
    parser.add_argument(
        "--enroll",
        action="store_true",
        help="also enroll the identity into the database when verification succeeds",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print today's attendance register instead of capturing",
    )
    parser.add_argument(
        "--export",
        metavar="FILE",
        default=None,
        help="export today's register as CSV to FILE",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="clear today's attendance register before the next capture session",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list or args.export:
        return _cmd_list(args)

    if args.clear:
        _cmd_clear(args)
        return 0

    return _cmd_capture(args)


# --------------------------------------------------------------------------- #
# Today's register
# --------------------------------------------------------------------------- #


def _cmd_list(args) -> int:
    try:
        store = Store(args.home and config.home_dir() / args.home)
    except ModelMissingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    entries = store.today_attendances()
    if not entries:
        print("No attendance recorded today.")
        store.close()
        return 0

    print(f"Attendance register — {today_iso}  ({len(entries)} records)")
    print("-" * 78)
    print(
        f"{'TIME':<9}{'ID NUMBER':<16}{'NAME':<24}{'VERDICT':<10}"
        f"{'CARD':>6}{'RECORD':>7}"
    )
    print("-" * 78)
    for e in entries:
        card = "-" if e.live_vs_card is None else f"{e.live_vs_card:.2f}"
        record = "-" if e.live_vs_enrolled is None else f"{e.live_vs_enrolled:.2f}"
        print(
            f"{e.created_at[11:19]:<9}"
            f"{(e.id_number or '-'):<16}"
            f"{(e.full_name or '-'):<24}"
            f"{e.result:<10}"
            f"{card:>6}"
            f"{record:>7}",
        )
    print("-" * 78)
    store.close()
    return 0


def _cmd_export(args) -> int:
    if not args.export:
        return 0
    try:
        store = Store(args.home and config.home_dir() / args.home)
    except ModelMissingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    entries = store.today_attendances()
    with open(args.export, "w") as f:
        f.write("time,id_number,name,result,live_vs_card,live_vs_enrolled,card_expired,reasons\n")
        for e in entries:
            reasons = "; ".join(e.reasons)
            f.write(
                f"{e.created_at[11:19]},{e.id_number or ''},{e.full_name or ''},"
                f"{e.result},{e.live_vs_card or ''},{e.live_vs_enrolled or ''},"
                f"{e.card_expired or ''},{reasons}\n"
            )
    print(f"Exported {len(entries)} records to {args.export}")
    store.close()
    return 0


def _cmd_list_and_export(args) -> int:
    _cmd_list(args)
    if args.export:
        _cmd_export(args)
    return 0


def _cmd_clear(args) -> int:
    try:
        store = Store(args.home and config.home_dir() / args.home)
    except ModelMissingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    cleared = store.clear_today_attendances()
    print(f"Cleared {cleared} attendance record(s) for today.")
    store.close()
    return 0


# --------------------------------------------------------------------------- #
# Live capture flow
# --------------------------------------------------------------------------- #


def _cmd_capture(args) -> int:
    try:
        system = IdentitySystem(home=args.home)
    except ModelMissingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    print(f"Starting camera {args.camera} ...")
    try:
        with Camera(args.camera) as cam:
            _live_session(system, cam, enroll=args.enroll)
    except CameraError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except (FaceError, EnrollError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    return 0


def _live_session(system: IdentitySystem, cam: Camera, enroll: bool) -> int:
    """Interactive capture loop: capture card + live face, verify / enroll.

    The camera is shared — when capturing the card the operator points it at
    the card, then at the person for the live face. Each capture is saved to
    the images folder so the operator can replay / re-capture if needed.
    """
    store = system.store
    print()
    print("=" * 70)
    print("  IDVERIFY ATTENDANCE CAPTURE")
    print("=" * 70)
    print()
    print("  Point the camera at the ID card and press ENTER to capture it.")
    print("  Then point the camera at the person and press ENTER for the live face.")
    print("  Type 'q' and ENTER at any prompt to quit.")
    print()

    while True:
        card = _capture_card(cam)
        if card is None:
            print("Quitting.")
            return 0

        face = _capture_face(cam)
        if face is None:
            print("Quitting.")
            return 0

        stamp = time.strftime("%Y%m%d-%H%M%S")
        card_path = store.images_dir / f"attend-{stamp}-card.png"
        face_path = store.images_dir / f"attend-{stamp}-face.png"
        cv2.imwrite(str(card_path), card)
        cv2.imwrite(str(face_path), face)

        print()
        print("Verifying... ", end="", flush=True)
        result = system.verify(card, face)

        _print_result(result)

        # Record the attendance attempt whichever way it went
        store.log_attendance(
            AttendanceLog(
                created_at=result.created_at,
                result=result.verdict,
                id_number=result.card_fields.id_number,
                full_name=result.card_fields.full_name,
                enrolled=result.enrolled,
                live_vs_card=result.live_vs_card,
                live_vs_enrolled=result.live_vs_enrolled,
                card_expired=result.card_expired,
                reasons=result.reasons,
                warnings=result.warnings,
                card_path=str(card_path),
                face_path=str(face_path),
            )
        )

        if enroll and result.verdict == VERIFIED and not result.enrolled:
            print()
            print("Enrolling this identity into the database... ", end="", flush=True)
            try:
                enroll_result = system.enroll(card, face)
                print(f"OK ({enroll_result.elapsed:.1f}s)")
                print(f"  Enrolled: {enroll_result.id_number}  {enroll_result.full_name or '-'}")
            except EnrollError as exc:
                print(f"refused: {exc}")

        if result.verdict == VERIFIED:
            return 0
        if result.verdict == REVIEW:
            return 2
        return 1


def _capture_card(cam: Camera) -> Optional[np.ndarray]:
    print("-" * 70)
    print("  STEP 1 — ID CARD")
    print("-" * 70)
    while True:
        reply = input("Point the camera at the ID card and press ENTER  [q to quit]: ").strip()
        if reply.lower() == "q":
            return None
        if reply == "":
            break
    print("Capturing card image... ", end="", flush=True)
    frame = cam.read()
    print(f"OK ({frame.shape[1]}x{frame.shape[0]})")
    _preview_small(frame)
    return frame


def _capture_face(cam: Camera) -> Optional[np.ndarray]:
    print()
    print("-" * 70)
    print("  STEP 2 — LIVE FACE")
    print("-" * 70)
    while True:
        reply = input("Point the camera at the person and press ENTER  [q to quit]: ").strip()
        if reply.lower() == "q":
            return None
        if reply == "":
            break
    print("Capturing live face... ", end="", flush=True)
    frame = cam.read()
    print(f"OK ({frame.shape[1]}x{frame.shape[0]})")
    _preview_small(frame)
    return frame


def _preview_small(frame: np.ndarray) -> None:
    """Print a tiny ASCII preview so the operator can sanity-check the frame."""
    small = cv2.resize(frame, (48, 18))
    small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    chars = " .:-=+*#%@"
    print()
    for row in small:
        line = "".join(chars[min(len(chars) - 1, v * len(chars) // 256)] for v in row)
        print("  " + line)
    print()


def _print_result(result: VerificationResult) -> None:
    print()
    print("=" * 70)
    print(f"  RESULT: {result.verdict}")
    print("-" * 70)
    print(f"  ID number   : {result.card_fields.label('id_number')}")
    print(f"  Name        : {result.card_fields.label('full_name')}")
    print(f"  DOB / Expiry: {result.card_fields.label('date_of_birth')} / "
          f"{result.card_fields.label('date_of_expiry')}"
          + ("  (EXPIRED)" if result.card_expired else ""))
    print(f"  Basis       : {result.basis}")
    print(f"  Card check  : {_score(result.live_vs_card)}")
    print(f"  Record check: {_score(result.live_vs_enrolled)}")
    print()
    for reason in result.reasons:
        print(f"  - {reason}")
    for warning in result.warnings:
        print(f"  ! {warning}")
    print("=" * 70)


def _score(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"
