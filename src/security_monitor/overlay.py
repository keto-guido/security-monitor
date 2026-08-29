"""High-quality HUD drawing: supersampled TrueType, soft shadows, smooth shapes.

Every primitive here is blitted on top of live video, so the per-frame cost is
paid once per tile per rendered frame. Two things keep that affordable on older
machines:

* **Rasters are cached premultiplied.** Glyph runs, status dots and rounded
  cards are rendered by Pillow once, then stored as ``(premul, inv)`` uint16
  pairs. Compositing is then a single integer multiply-add per pixel with no
  float conversion and no per-call copy.
* **HUD quality is switchable.** ``set_hud_quality("fast")`` drops the
  supersample factor and the Gaussian shadow blur, which is what a weak CPU
  notices when cache-missing text (the FPS readout changes twice a second).
  ``"high"`` keeps the original look for capable machines.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SS = 2  # supersample factor for glyphs and shapes (high quality)
SS_FAST = 1  # no supersample when the HUD is in fast mode

HUD_QUALITY_CHOICES = ("auto", "high", "fast")

# Machines at or below this core count default to the fast HUD under "auto".
_AUTO_FAST_MAX_CORES = 4

_hud_fast = False

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

# A premultiplied raster: (height, width, rgb*alpha as uint16, 255-alpha as uint16).
Raster = tuple[int, int, np.ndarray, np.ndarray]


def hud_quality_label(mode: str) -> str:
    key = (mode or "auto").strip().lower()
    if key == "high":
        return "High — supersampled text and soft shadows"
    if key == "fast":
        return "Fast — cheaper HUD for older CPUs"
    return f"Auto ({'fast' if _auto_prefers_fast() else 'high'} on this machine)"


def _auto_prefers_fast() -> bool:
    return (os.cpu_count() or 1) <= _AUTO_FAST_MAX_CORES


def resolve_hud_quality(mode: str) -> str:
    """Map ``auto`` to the concrete quality this machine should use."""
    key = (mode or "auto").strip().lower()
    if key in {"high", "fast"}:
        return key
    return "fast" if _auto_prefers_fast() else "high"


def set_hud_quality(mode: str) -> str:
    """Switch HUD rendering quality. Returns the resolved mode (high|fast)."""
    global _hud_fast
    resolved = resolve_hud_quality(mode)
    fast = resolved == "fast"
    if fast != _hud_fast:
        _hud_fast = fast
        clear_raster_caches()
    return resolved


def hud_quality() -> str:
    return "fast" if _hud_fast else "high"


def clear_raster_caches() -> None:
    """Drop cached rasters — call after anything that changes how they look."""
    _raster_text.cache_clear()
    _raster_dot.cache_clear()
    _raster_round_rect.cache_clear()
    _bar_weights.cache_clear()


def _ss() -> int:
    return SS_FAST if _hud_fast else SS


def _resample():
    # LANCZOS is noticeably sharper but several times the cost of BILINEAR.
    return Image.Resampling.BILINEAR if _hud_fast else Image.Resampling.LANCZOS


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


def _premultiply(rgba: np.ndarray) -> Raster:
    """Split an RGBA raster into the two uint16 planes ``blit_premul`` wants."""
    alpha = rgba[:, :, 3].astype(np.uint16)[:, :, None]
    bgr = rgba[:, :, :3][:, :, ::-1].astype(np.uint16)
    premul = bgr * alpha  # <= 255*255, fits uint16
    inv = (np.uint16(255) - alpha).astype(np.uint16)
    return int(rgba.shape[0]), int(rgba.shape[1]), premul, inv


def blit_premul(image: np.ndarray, raster: Raster, x: int, y: int) -> None:
    """
    Alpha-composite a cached premultiplied raster with integer math only.

    ``out = (dst * (255 - a) + src * a) / 255`` in uint16 — half the memory
    traffic of the float32 path and no temporary copy of the raster.
    """
    h, w, premul, inv = raster
    if w <= 0 or h <= 0:
        return
    ih, iw = image.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(iw, x + w), min(ih, y + h)
    if x0 >= x1 or y0 >= y1:
        return
    sx0, sy0 = x0 - x, y0 - y
    sx1, sy1 = sx0 + (x1 - x0), sy0 + (y1 - y0)
    src = premul[sy0:sy1, sx0:sx1]
    a_inv = inv[sy0:sy1, sx0:sx1]
    roi = image[y0:y1, x0:x1]
    blended = roi.astype(np.uint16)
    blended *= a_inv
    blended += src
    blended //= 255
    image[y0:y1, x0:x1] = blended.astype(np.uint8)


def blit_rgba(image: np.ndarray, rgba: np.ndarray, x: int, y: int) -> None:
    """Composite a plain RGBA raster. Prefer ``blit_premul`` on hot paths."""
    if rgba.size == 0:
        return
    blit_premul(image, _premultiply(rgba), x, y)


def _downscale_rgba(im: Image.Image, out_w: int, out_h: int) -> np.ndarray:
    if im.size != (out_w, out_h):
        im = im.resize((max(1, out_w), max(1, out_h)), _resample())
    return np.array(im)


@lru_cache(maxsize=384)
def _raster_text(text: str, size: int, color: tuple[int, int, int]) -> Raster:
    ss = _ss()
    font = _font(size * ss)
    dummy = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    bbox = dummy.textbbox((0, 0), text, font=font)
    left, top, right, bottom = bbox
    blur = max(1, ss)
    pad = blur * 3
    w = max(1, right - left + pad * 2)
    h = max(1, bottom - top + pad * 2)
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).text(
        (pad - left + ss, pad - top + ss),
        text,
        font=font,
        fill=(0, 0, 0, 150),
    )
    if not _hud_fast:
        # The blur is the single most expensive step; fast mode keeps the
        # offset drop-shadow but skips softening it.
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=blur * 0.7))
    fg = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(fg).text(
        (pad - left, pad - top),
        text,
        font=font,
        fill=(*color, 255),
    )
    composed = Image.alpha_composite(shadow, fg)
    out_w = max(1, int(round(w / ss)))
    out_h = max(1, int(round(h / ss)))
    return _premultiply(_downscale_rgba(composed, out_w, out_h))


def measure_text(text: str, size: int, stroke: int = 0) -> tuple[int, int]:
    h, w, _premul, _inv = _raster_text(text, size, (236, 236, 236))
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
    raster = _raster_text(text, size, _rgb(color))
    height, width = raster[0], raster[1]
    x, y = origin
    if align == "right":
        x -= width
    elif align == "center":
        x -= width // 2
    if valign == "bottom":
        y -= height
    elif valign == "center":
        y -= height // 2
    blit_premul(image, raster, int(x), int(y))


@lru_cache(maxsize=64)
def _raster_dot(out_d: int, color: tuple[int, int, int]) -> Raster:
    ss = 2 if _hud_fast else 4
    d = out_d * ss
    im = Image.new("RGBA", (d, d), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    inset = ss * 1.2
    draw.ellipse((inset, inset, d - 1 - inset, d - 1 - inset), fill=(*color, 255))
    ring = tuple(max(0, c - 40) for c in color)
    draw.ellipse(
        (inset, inset, d - 1 - inset, d - 1 - inset),
        outline=(*ring, 120),
        width=max(1, ss // 2),
    )
    return _premultiply(_downscale_rgba(im, out_d, out_d))


def draw_dot(
    image: np.ndarray,
    center: tuple[int, int],
    radius: float,
    color: tuple[int, int, int],
) -> None:
    out_d = max(8, int(round(radius * 2)) + 6)
    raster = _raster_dot(out_d, _rgb(color))
    cx, cy = center
    blit_premul(image, raster, int(cx - out_d / 2), int(cy - out_d / 2))


@lru_cache(maxsize=64)
def _bar_weights(height: int, fade: int, alpha_q: int) -> tuple[np.ndarray, np.ndarray]:
    """Cached (weight, 255-weight) ramps for the tile status strip, as uint16."""
    alpha = alpha_q / 255.0
    ramp = np.full((height, 1, 1), alpha, dtype=np.float32)
    # Smoothstep fade instead of a linear ramp (less banding).
    t = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    ramp[:fade, 0, 0] = t * t * (3.0 - 2.0 * t) * alpha
    weight = np.clip(np.rint(ramp * 255.0), 0, 255).astype(np.uint16)
    return weight, (np.uint16(255) - weight).astype(np.uint16)


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
    height = min(int(height), h)
    if height <= 0:
        return
    fade = max(2, min(int(fade), height))
    alpha_q = int(round(max(0.0, min(1.0, float(alpha))) * 255))
    weight, inv = _bar_weights(height, fade, alpha_q)
    fill = np.array(color, dtype=np.uint16) * weight
    roi = image[h - height : h, :]
    blended = roi.astype(np.uint16)
    blended *= inv
    blended += fill
    blended //= 255
    image[h - height : h, :] = blended.astype(np.uint8)


@lru_cache(maxsize=48)
def _raster_round_rect(
    w: int,
    h: int,
    color: tuple[int, int, int],
    alpha_q: int,
    radius: int,
) -> Raster:
    ss = 1 if _hud_fast else 2
    im = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle(
        (0, 0, w * ss - 1, h * ss - 1),
        radius=max(1, radius * ss),
        fill=(*color, alpha_q),
    )
    return _premultiply(_downscale_rgba(im, w, h))


def shade_round_rect(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    color: tuple[int, int, int] = (8, 8, 10),
    alpha: float = 0.78,
    radius: int = 10,
) -> None:
    x0, y0, x1, y1 = box
    w, h = max(1, int(x1 - x0)), max(1, int(y1 - y0))
    alpha_q = int(round(max(0.0, min(1.0, float(alpha))) * 255))
    raster = _raster_round_rect(w, h, _rgb(color), alpha_q, max(1, int(radius)))
    blit_premul(image, raster, int(x0), int(y0))
