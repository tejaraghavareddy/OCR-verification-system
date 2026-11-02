"""Shared fixtures: sample faces and a synthetic ID card image."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

ASSETS = Path(__file__).resolve().parent / "assets"
OBAMA = ASSETS / "obama.jpg"  # the face printed on the synthetic card
BIDEN = ASSETS / "biden.jpg"  # a different person - the impostor


def find_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]
    candidates += glob.glob(
        "/opt/*/lib/python*/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf"
    )
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def build_card(
    target: Path,
    portrait: Path = OBAMA,
    id_number: str = "4471-9938-2210",
    name: str = "BARACK H. OBAMA",
    date_of_birth: str = "04 AUG 1961",
    date_of_expiry: str = "12 MAY 2032",
) -> Path:
    """Render a plausible ID card with a real face as the printed portrait."""
    width, height = 1000, 630
    card = Image.new("RGB", (width, height), (238, 242, 248))
    draw = ImageDraw.Draw(card)
    draw.rectangle([0, 0, width, 90], fill=(24, 56, 110))
    draw.text((34, 26), "REPUBLIC IDENTITY CARD", font=find_font(34), fill=(255, 255, 255))

    photo = Image.open(portrait).convert("RGB").resize((300, 380))
    card.paste(photo, (40, 140))
    draw.rectangle([39, 139, 341, 521], outline=(24, 56, 110), width=3)

    fields = [
        ("NAME", name),
        ("ID NUMBER", id_number),
        ("DATE OF BIRTH", date_of_birth),
        ("DATE OF EXPIRY", date_of_expiry),
        ("SEX", "M"),
        ("NATIONALITY", "AMERICAN"),
    ]
    y = 150
    for label, value in fields:
        draw.text((380, y), label + ":", font=find_font(24), fill=(90, 90, 90))
        draw.text((380, y + 32), value, font=find_font(30), fill=(15, 15, 15))
        y += 62

    target.parent.mkdir(parents=True, exist_ok=True)
    card.save(target)
    return target


@pytest.fixture(scope="session")
def assets_ready() -> bool:
    for path in (OBAMA, BIDEN):
        assert path.is_file() and path.stat().st_size > 5000, f"missing asset {path}"
    return True


@pytest.fixture(scope="session")
def face_engine(assets_ready):
    from idverify.face import FaceEngine

    return FaceEngine()


@pytest.fixture(scope="session")
def card_image(tmp_path_factory, assets_ready) -> Path:
    return build_card(tmp_path_factory.mktemp("cards") / "card.png")


@pytest.fixture()
def make_card(tmp_path):
    def _make(name: str = "card.png", **kwargs) -> Path:
        return build_card(tmp_path / name, **kwargs)

    return _make


@pytest.fixture()
def system(tmp_path, face_engine):
    from idverify.engine import IdentitySystem

    return IdentitySystem(home=tmp_path / "home", face_engine=face_engine)