"""High-quality HUD drawing: supersampled TrueType, soft shadows, smooth shapes."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SS = 2  # supersample factor for glyphs and shapes

_FONT_CANDIDATES = (
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/segoeuisl.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/arial.ttf",
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
    return [root / name for name in ("segoeui.ttf", "segoeuisl.ttf", "calibri.ttf", "arial.ttf")]


@lru_cache(maxsize=1)
def _font_path() -> str | None:
    for path in [*_windir_fonts(), *[Path(p) for p in _FONT_CANDIDATES]]:
        if path.is_file():
            return str(path)
    return None


@lru_cache(maxsize=24)
def _font(size: int) -> ImageFont.ImageFont:
    path = _font_path()
    if path:
        return ImageFont.truetype(path, size=max(6, size))
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


def _downscale_rgba(im: Image.Image, out_w: int, out_h: int) -> np.ndarray:
    if im.size != (out_w, out_h):
        im = im.resize((max(1, out_w), max(1, out_h)), Image.Resampling.LANCZOS)
    return np.array(im)


@lru_cache(maxsize=256)
def _raster_text(text: str, size: int, color: tuple[int, int, int]) -> tuple[int, int, bytes]:
    font = _font(size * SS)
    dummy = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    bbox = dummy.textbbox((0, 0), text, font=font)
    left, top, right, bottom = bbox
    blur = max(1, SS)
    pad = blur * 3
    w = max(1, right - left + pad * 2)
    h = max(1, bottom - top + pad * 2)
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).text(
        (pad - left + SS, pad - top + SS),
        text,
        font=font,
        fill=(0, 0, 0, 150),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=blur * 0.7))
    fg = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(fg).text(
        (pad - left, pad - top),
        text,
        font=font,
        fill=(*color, 255),
    )
    composed = Image.alpha_composite(shadow, fg)
    out_w = max(1, int(round(w / SS)))
    out_h = max(1, int(round(h / SS)))
    arr = _downscale_rgba(composed, out_w, out_h)
    return int(arr.shape[0]), int(arr.shape[1]), arr.tobytes()


def measure_text(text: str, size: int, stroke: int = 0) -> tuple[int, int]:
    h, w, _payload = _raster_text(text, size, (236, 236, 236))
    return w, h


def draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    size: int = 15,
    color: tuple[int, int, int] = (236, 236, 236),
    align: str = "left",
    valign: str = "bottom",
    stroke: int = 0,
) -> None:
    """Draw supersampled TrueType text with a soft shadow (stroke is ignored)."""
    if not text:
        return
    height, width, payload = _raster_text(text, size, _rgb(color))
    rgba = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 4).copy()
    x, y = origin
    if align == "right":
        x -= width
    elif align == "center":
        x -= width // 2
    if valign == "bottom":
        y -= height
    elif valign == "center":
        y -= height // 2
    blit_rgba(image, rgba, int(x), int(y))


def draw_dot(
    image: np.ndarray,
    center: tuple[int, int],
    radius: float,
    color: tuple[int, int, int],
) -> None:
    ss = 4
    out_d = max(8, int(round(radius * 2)) + 6)
    d = out_d * ss
    im = Image.new("RGBA", (d, d), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    rgb = _rgb(color)
    inset = ss * 1.2
    draw.ellipse((inset, inset, d - 1 - inset, d - 1 - inset), fill=(*rgb, 255))
    ring = tuple(max(0, c - 40) for c in rgb)
    draw.ellipse(
        (inset, inset, d - 1 - inset, d - 1 - inset),
        outline=(*ring, 120),
        width=max(1, ss // 2),
    )
    arr = _downscale_rgba(im, out_d, out_d)
    cx, cy = center
    blit_rgba(image, arr, int(cx - out_d / 2), int(cy - out_d / 2))


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
    fade = max(2, min(fade, height))
    weights = np.ones((height, 1, 1), dtype=np.float32) * alpha
    # Smoothstep fade instead of a linear ramp (less banding).
    t = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    weights[:fade, 0, 0] = t * t * (3.0 - 2.0 * t) * alpha
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
    ss = 2
    im = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle(
        (0, 0, w * ss - 1, h * ss - 1),
        radius=max(1, radius * ss),
        fill=(*_rgb(color), int(alpha * 255)),
    )
    arr = _downscale_rgba(im, w, h)
    blit_rgba(image, arr, x0, y0)
