"""Command line front end: scan, enroll, verify, history - all offline.

Exit codes for ``verify``: 0 = verified, 1 = rejected, 2 = needs review.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from . import __version__, config
from .engine import (
    REVIEW,
    VERIFIED,
    EnrollError,
    IdentitySystem,
    VerificationResult,
)
from .face import FaceError, ModelMissingError
from .idcard import FIELD_LABELS, CardScan

RULE = "-" * 66


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="idverify",
        description="Offline ID-card identity verification (OCR + face match, no network).",
    )
    parser.add_argument("--version", action="version", version=f"idverify {__version__}")
    parser.add_argument(
        "--home",
        metavar="DIR",
        default=None,
        help=f"data directory for the local database and images (default: {config.home_dir()})",
    )
    sub = parser.add_subparsers(dest="command")

    scan = sub.add_parser("scan", help="read a card image and print its fields")
    scan.add_argument("--card", required=True, metavar="IMAGE", help="photo/scan of the ID card")

    enroll = sub.add_parser("enroll", help="register an identity from a card and a live face")
    enroll.add_argument("--card", required=True, metavar="IMAGE")
    enroll.add_argument("--face", required=True, metavar="IMAGE", help="live capture of the holder")

    verify = sub.add_parser("verify", help="verify a live face against a card and the local record")
    verify.add_argument("--card", required=True, metavar="IMAGE")
    verify.add_argument("--face", required=True, metavar="IMAGE")
    verify.add_argument("--camera", type=int, metavar="INDEX", default=None,
                        help="read the live face from this camera index instead of a file")

    history = sub.add_parser("history", help="show the local verification audit log")
    history.add_argument("--limit", type=int, default=20)

    sub.add_parser("list", help="list enrolled identities")

    forget = sub.add_parser("forget", help="remove an enrolled identity")
    forget.add_argument("--id", required=True, dest="id_number", metavar="ID_NUMBER")

    attend = sub.add_parser("attend", help="live camera attendance / enrollment capture")
    attend.add_argument("--camera", type=int, default=0, metavar="INDEX",
                       help="camera index (default: 0)")
    attend.add_argument("--enroll", action="store_true",
                       help="also enroll the identity when verification succeeds")
    attend.add_argument("--list", action="store_true",
                       help="print today's attendance register")
    attend.add_argument("--export", metavar="FILE", default=None,
                       help="export today's register as CSV")
    attend.add_argument("--clear", action="store_true",
                       help="clear today's attendance register")
    sub.add_parser("gui", help="launch the desktop interface (default)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in (None, "gui"):
        return _run_gui()

    try:
        system = IdentitySystem(home=args.home)
    except ModelMissingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    try:
        if args.command == "scan":
            return _cmd_scan(system, args)
        if args.command == "enroll":
            return _cmd_enroll(system, args)
        if args.command == "verify":
            return _cmd_verify(system, args)
        if args.command == "attend":
            return _cmd_attend(args)
        if args.command == "history":
            return _cmd_history(system, args)
        if args.command == "list":
            return _cmd_list(system)
        if args.command == "forget":
            return _cmd_forget(system, args)
    except (FaceError, EnrollError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    parser.error(f"unknown command {args.command!r}")
    return 3


# --------------------------------------------------------------------------- #
def _run_gui() -> int:
    try:
        from .gui import run
    except ImportError as exc:  # tkinter missing (common on minimal Linux)
        print(
            "error: the desktop interface needs tkinter "
            f"({exc}). Install it (Debian/Ubuntu: sudo apt install python3-tk) "
            "or use the commands: idverify scan | enroll | verify | history | list",
            file=sys.stderr,
        )
        return 3
    run()
    return 0


def _cmd_scan(system: IdentitySystem, args) -> int:
    scan = system.reader.scan(args.card)
    _print_scan(scan)
    return 0


def _cmd_enroll(system: IdentitySystem, args) -> int:
    print(f"Reading card... ", end="", flush=True)
    result = system.enroll(args.card, args.face)
    verb = "updated" if result.updated else "enrolled"
    print(f"OK ({result.elapsed:.1f}s)")
    print(RULE)
    print(f"  {verb.upper():<18} {result.id_number}")
    print(f"  {'Name':<18} {result.full_name or '-'}")
    print(f"  {'Face vs portrait':<18} {result.live_vs_card:.3f}")
    print(f"  {'Saved':<18} {result.portrait_path}")
    for warning in result.warnings:
        print(f"  warning: {warning}")
    print(RULE)
    return 0


def _cmd_verify(system: IdentitySystem, args) -> int:
    face_source = args.face
    if args.camera is not None:
        from .camera import Camera

        with Camera(args.camera) as camera:
            face_source = camera.read()
    print("Verifying... ", end="", flush=True)
    result = system.verify(args.card, face_source)
    print(f"OK ({result.elapsed:.1f}s)")
    _print_verification(result)
    if result.verdict == VERIFIED:
        return 0
    return 2 if result.verdict == REVIEW else 1


def _cmd_history(system: IdentitySystem, args) -> int:
    entries = system.history(args.limit)
    if not entries:
        print("No verifications recorded yet.")
        return 0
    print(RULE)
    print(f"{'WHEN (UTC)':<26}{'RESULT':<10}{'ID NUMBER':<18}{'CARD':>7}{'RECORD':>8}")
    print(RULE)
    for entry in entries:
        card = "-" if entry.live_vs_card is None else f"{entry.live_vs_card:.2f}"
        record = "-" if entry.live_vs_enrolled is None else f"{entry.live_vs_enrolled:.2f}"
        print(
            f"{entry.created_at:<26}{entry.result:<10}"
            f"{(entry.id_number or '-'):<18}{card:>7}{record:>8}"
        )
    print(RULE)
    print(f"{len(entries)} entr(ies) from {system.store.db_path}")
    return 0


def _cmd_list(system: IdentitySystem) -> int:
    people = system.people()
    if not people:
        print("No identities enrolled yet.")
        return 0
    print(RULE)
    print(f"{'ID NUMBER':<18}{'NAME':<26}{'DOB':<13}{'EXPIRY':<13}UPDATED")
    print(RULE)
    for person in people:
        print(
            f"{person.id_number:<18}{person.display_name[:25]:<26}"
            f"{(person.date_of_birth or '-'):<13}"
            f"{(person.date_of_expiry or '-'):<13}{person.updated_at}"
        )
    print(RULE)
    print(f"{len(people)} enrolled identit(ies) in {system.store.home}")
    return 0


def _cmd_forget(system: IdentitySystem, args) -> int:
    if system.forget(args.id_number):
        print(f"Removed {args.id_number}.")
        return 0
    print(f"No enrolled identity with ID number {args.id_number}.", file=sys.stderr)
    return 1


# --------------------------------------------------------------------------- #
def _print_scan(scan: CardScan) -> None:
    fields = scan.fields
    print(RULE)
    for key, label in FIELD_LABELS.items():
        print(f"  {label:<18} {fields.label(key)}")
    print(RULE)
    print(f"  {'Portrait':<18} {'extracted' if scan.found_portrait else 'NOT FOUND'}")
    print(f"  {'Faces on card':<18} {len(scan.faces)}")
    print(f"  {'OCR lines':<18} {len(fields.raw_lines)}   ({scan.elapsed:.1f}s)")
    for warning in scan.warnings:
        print(f"  warning: {warning}")
    print(RULE)


def _print_verification(result: VerificationResult) -> None:
    fields = result.card_fields
    print(RULE)
    print(f"  RESULT          {result.verdict}")
    print(f"  {'ID NUMBER':<16}  {fields.label('id_number')}")
    print(f"  {'NAME':<16}  {fields.label('full_name')}")
    print(f"  {'BIRTH / EXPIRY':<16}  {fields.label('date_of_birth')} / {fields.label('date_of_expiry')}")
    print(f"  {'BASIS':<16}  {result.basis}")
    if result.person is not None:
        print(f"  {'RECORD NAME':<16}  {result.person.display_name}")
    print(f"  {'CARD CHECK':<16}  {_score(result.live_vs_card)}")
    print(f"  {'RECORD CHECK':<16}  {_score(result.live_vs_enrolled)}")
    print(RULE)
    for reason in result.reasons:
        print(f"  - {reason}")
    for warning in result.warnings:
        print(f"  ! {warning}")
    print(RULE)


def _score(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"