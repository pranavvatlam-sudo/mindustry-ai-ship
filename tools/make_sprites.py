"""Generates the AI Ship sprites and mod icon.

Everything is drawn at 4x and downsampled, so the result stays crisp at game
scale without hand-pixeling. Run from the repo root:

    python tools/make_sprites.py
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

SS = 4  # supersampling factor

OUTLINE = (11, 15, 24, 255)
HULL_DARK = (36, 45, 70, 255)
HULL_MID = (52, 67, 101, 255)
HULL_LIGHT = (78, 99, 146, 255)
ACCENT = (96, 232, 255, 255)
ACCENT_DIM = (46, 122, 158, 255)
ENGINE = (140, 246, 255, 255)

ROOT = Path(__file__).resolve().parent.parent


def canvas(size: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (size * SS, size * SS), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def pt(points, scale=SS):
    return [(x * scale, y * scale) for x, y in points]


def mirror(points):
    return [(64 - x, y) for x, y in points]


def draw_ship(size: int = 64) -> Image.Image:
    img, d = canvas(size)

    hull = [
        (32, 2),
        (37, 14),
        (39, 30),
        (38, 44),
        (42, 56),
        (35, 60),
        (32, 50),
        (29, 60),
        (22, 56),
        (26, 44),
        (25, 30),
        (27, 14),
    ]
    wing_l = [(26, 18), (9, 36), (5, 46), (14, 43), (24, 32)]
    wing_r = mirror(wing_l)
    fin_l = [(25, 42), (16, 57), (23, 56), (28, 47)]
    fin_r = mirror(fin_l)
    spine = [(30, 8), (34, 8), (33, 46), (31, 46)]

    # engine glow behind the hull
    glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for x in (25.5, 38.5):
        gd.ellipse(
            [(x - 6) * SS, (54 - 6) * SS, (x + 6) * SS, (54 + 8) * SS],
            fill=(80, 230, 255, 130),
        )
    glow = glow.filter(ImageFilter.GaussianBlur(9 * SS))
    img.alpha_composite(glow)

    # dark silhouette first, slightly expanded
    for shape in (wing_l, wing_r, fin_l, fin_r, hull):
        d.polygon(pt(shape), fill=OUTLINE, outline=OUTLINE, width=int(1.6 * SS))

    for shape in (wing_l, wing_r):
        d.polygon(pt(shape), fill=HULL_MID)
    for shape in (fin_l, fin_r):
        d.polygon(pt(shape), fill=HULL_DARK)

    d.polygon(pt(hull), fill=HULL_MID)
    d.polygon(pt(spine), fill=HULL_LIGHT)

    # leading-edge highlights
    d.line(pt([(7, 38), (24, 20)]), fill=HULL_LIGHT, width=int(1.2 * SS))
    d.line(pt([(57, 38), (40, 20)]), fill=HULL_LIGHT, width=int(1.2 * SS))

    # cockpit lens
    d.ellipse(
        [27.5 * SS, 16 * SS, 36.5 * SS, 33 * SS],
        fill=ACCENT_DIM,
        outline=OUTLINE,
        width=int(1.1 * SS),
    )
    d.ellipse([29.2 * SS, 18.5 * SS, 34.8 * SS, 28.5 * SS], fill=ACCENT)
    d.ellipse([30.5 * SS, 20 * SS, 32.4 * SS, 23 * SS], fill=(235, 255, 255, 255))

    # panel lines
    d.line(pt([(32, 34), (32, 46)]), fill=OUTLINE, width=int(0.7 * SS))
    d.line(pt([(24, 40), (30, 44)]), fill=OUTLINE, width=int(0.6 * SS))
    d.line(pt([(40, 40), (34, 44)]), fill=OUTLINE, width=int(0.6 * SS))

    # engine cores
    for x in (25.5, 38.5):
        d.ellipse(
            [(x - 2.4) * SS, 52.4 * SS, (x + 2.4) * SS, 57.4 * SS],
            fill=ENGINE,
        )

    # faint rim light along the nose
    d.line(pt([(32, 4), (36, 15)]), fill=(190, 250, 255, 220), width=int(0.9 * SS))

    return img.resize((size, size), Image.LANCZOS)


def draw_weapon(size: int = 32) -> Image.Image:
    img, d = canvas(size)

    # weapon sprites point up; centered on the mount
    body = [(16, 3), (20, 9), (20, 26), (12, 26), (12, 9)]
    d.polygon(pt(body), fill=OUTLINE)
    d.polygon(pt([(16.6, 5), (19, 10), (19, 24), (13, 24), (13, 10)]), fill=HULL_MID)

    d.rectangle([13 * SS, 21 * SS, 19 * SS, 26 * SS], fill=HULL_DARK)
    d.rectangle([14.5 * SS, 23 * SS, 17.5 * SS, 26 * SS], fill=ACCENT_DIM)

    # muzzle
    d.rectangle([14.7 * SS, 2.5 * SS, 17.3 * SS, 8 * SS], fill=HULL_LIGHT)
    d.ellipse([14.6 * SS, 1.6 * SS, 17.4 * SS, 4.6 * SS], fill=ACCENT)

    # heat vents
    for y in (12, 15, 18):
        d.line(pt([(13.4, y), (18.6, y)]), fill=OUTLINE, width=int(0.5 * SS))

    return img.resize((size, size), Image.LANCZOS)


def draw_icon(size: int = 128) -> Image.Image:
    img = Image.new("RGBA", (size * SS, size * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # rounded dark plate with a cyan rim
    d.rounded_rectangle(
        [2 * SS, 2 * SS, 126 * SS, 126 * SS],
        radius=26 * SS,
        fill=(13, 18, 30, 255),
        outline=(38, 52, 78, 255),
        width=int(2 * SS),
    )

    # glow
    glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse([18 * SS, 18 * SS, 110 * SS, 110 * SS], fill=(60, 200, 255, 60))
    glow = glow.filter(ImageFilter.GaussianBlur(10 * SS))
    img.alpha_composite(glow)

    # orbit ring
    d.arc(
        [14 * SS, 14 * SS, 114 * SS, 114 * SS],
        start=200,
        end=340,
        fill=(80, 210, 240, 170),
        width=int(1.4 * SS),
    )
    d.arc(
        [14 * SS, 14 * SS, 114 * SS, 114 * SS],
        start=20,
        end=160,
        fill=(80, 210, 240, 90),
        width=int(1.4 * SS),
    )

    ship = draw_ship(96).resize((96 * SS, 96 * SS), Image.LANCZOS)
    img.alpha_composite(ship, (16 * SS, 14 * SS))

    return img.resize((size, size), Image.LANCZOS)


def main():
    sprites = ROOT / "sprites"
    sprites.mkdir(exist_ok=True)

    draw_ship(64).save(sprites / "ai-ship.png")
    draw_weapon(32).save(sprites / "ai-ship-weapon.png")
    draw_icon(128).save(ROOT / "icon.png")

    print("wrote:")
    for path in (sprites / "ai-ship.png", sprites / "ai-ship-weapon.png", ROOT / "icon.png"):
        print("  ", path.relative_to(ROOT), path.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
