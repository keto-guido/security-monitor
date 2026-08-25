"""Anti-aliased HUD drawing (TrueType text, soft bars, stable layout)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = (
    # Windows
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "C:/Windows/Fonts/arial.ttf",
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
)


def _windir_fonts() -> list[Path]:
    windir = os.environ.get("WINDIR")
    if not windir:
        return []
    root = Path(windir) / "Fonts"
    return [root / name for name in ("segoeui.ttf", "calibri.ttf", "tahoma.ttf", "arial.ttf")]


@lru_cache(maxsize=1)
def _font_path() -> str | None:
    for path in [*_windir_fonts(), *[Path(p) for p in _FONT_CANDIDATES]]:
        if path.is_file():
            return str(path)
    return None


@lru_cache(maxsize=16)
def _font(size: int) -> ImageFont.ImageFont:
    path = _font_path()
    if path:
        return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _rgb(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    return int(bgr[2]), int(bgr[1]), int(bgr[0])


def blit_rgba(image: np.ndarray, rgba: np.ndarray, x: int, y: int) -> None:
    ih, iw = image.shape[:2]
    h, w = rgba.shape[:2]
    if w <= 0 or h <= 0:
        return
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(iw, x + w), min(ih, y + h)
    if x0 >= x1 or y0 >= y1:
        return
    sx0, sy0 = x0 - x, y0 - y
    patch = rgba[sy0 : sy0 + (y1 - y0), sx0 : sx0 + (x1 - x0)]
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
    if float(alpha.max()) <= 0:
        return
    rgb = patch[:, :, :3][:, :, ::-1].astype(np.float32)
    roi = image[y0:y1, x0:x1].astype(np.float32)
    image[y0:y1, x0:x1] = (roi * (1.0 - alpha) + rgb * alpha).astype(np.uint8)


def measure_text(text: str, size: int, stroke: int = 1) -> tuple[int, int]:
    font = _font(size)
    dummy = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    bbox = dummy.textbbox((0, 0), text, font=font, stroke_width=stroke)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    size: int = 15,
    color: tuple[int, int, int] = (236, 236, 236),
    align: str = "left",
    valign: str = "bottom",
    stroke: int = 1,
) -> None:
    """Draw anti-aliased TrueType text. Origin uses box anchors, not font baselines."""
    if not text:
        return
    font = _font(size)
    dummy = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    bbox = dummy.textbbox((0, 0), text, font=font, stroke_width=stroke)
    left, top, right, bottom = bbox
    pad = stroke + 1
    w, h = right - left + pad * 2, bottom - top + pad * 2
    canvas = Image.new("RGBA", (max(1, w), max(1, h)), (0, 0, 0, 0))
    ImageDraw.Draw(canvas).text(
        (pad - left, pad - top),
        text,
        font=font,
        fill=(*_rgb(color), 255),
        stroke_width=stroke,
        stroke_fill=(10, 10, 10, 230),
    )
    rgba = np.array(canvas)
    x, y = origin
    if align == "right":
        x -= rgba.shape[1]
    elif align == "center":
        x -= rgba.shape[1] // 2
    if valign == "bottom":
        y -= rgba.shape[0]
    elif valign == "center":
        y -= rgba.shape[0] // 2
    blit_rgba(image, rgba, int(x), int(y))


def draw_dot(
    image: np.ndarray,
    center: tuple[int, int],
    radius: float,
    color: tuple[int, int, int],
) -> None:
    d = int(round(radius * 2)) + 4
    im = Image.new("RGBA", (d, d), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    inset = 1.5
    rgb = _rgb(color)
    draw.ellipse((inset, inset, d - 1 - inset, d - 1 - inset), fill=(*rgb, 255))
    ring = tuple(max(0, c - 50) for c in rgb)
    draw.ellipse((inset, inset, d - 1 - inset, d - 1 - inset), outline=(*ring, 180), width=1)
    cx, cy = center
    blit_rgba(image, np.array(im), int(cx - d / 2), int(cy - d / 2))


def shade_bottom_bar(
    image: np.ndarray,
    height: int,
    *,
    color: tuple[int, int, int] = (10, 10, 12),
    alpha: float = 0.72,
    fade: int = 10,
) -> None:
    """Soft-edged status strip so the HUD does not crawl against the video."""
    h, w = image.shape[:2]
    height = min(height, h)
    fade = min(fade, height)
    weights = np.ones((height, 1, 1), dtype=np.float32) * alpha
    if fade > 1:
        weights[:fade, 0, 0] = np.linspace(0.0, alpha, fade, dtype=np.float32)
    roi = image[h - height : h, :].astype(np.float32)
    fill = np.array(color, dtype=np.float32)
    image[h - height : h, :] = (roi * (1.0 - weights) + fill * weights).astype(np.uint8)


def shade_round_rect(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    color: tuple[int, int, int] = (8, 8, 10),
    alpha: float = 0.78,
    radius: int = 10,
) -> None:
    x0, y0, x1, y1 = box
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle(
        (0, 0, w - 1, h - 1),
        radius=radius,
        fill=(*_rgb(color), int(alpha * 255)),
    )
    blit_rgba(image, np.array(im), x0, y0)
