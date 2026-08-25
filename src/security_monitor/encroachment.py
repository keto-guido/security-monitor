"""Per-camera encroachment / tripwire helpers.

A directed line divides the frame into two half-planes. People whose feet
(bottom-center of the person box) land on the configured side are "in the
zone". Coordinates are normalized 0–1 in the post-rotate source frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from security_monitor.detection import Box

VALID_ENCROACH_SIDES = ("positive", "negative")

# Preset tripwires (x1, y1, x2, y2) in normalized frame space.
LINE_PRESETS: tuple[tuple[str, tuple[float, float, float, float]], ...] = (
    ("Horizontal mid", (0.0, 0.5, 1.0, 0.5)),
    ("Horizontal low", (0.0, 0.66, 1.0, 0.66)),
    ("Horizontal high", (0.0, 0.33, 1.0, 0.33)),
    ("Vertical mid", (0.5, 0.0, 0.5, 1.0)),
    ("Vertical left", (0.33, 0.0, 0.33, 1.0)),
    ("Vertical right", (0.66, 0.0, 0.66, 1.0)),
)

DEFAULT_ENCROACH_LINE: tuple[float, float, float, float] = (0.0, 0.5, 1.0, 0.5)


@dataclass(frozen=True)
class EncroachLine:
    """Directed tripwire in normalized 0–1 frame coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    side: str = "positive"  # positive | negative

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def with_side(self, side: str) -> EncroachLine:
        return EncroachLine(self.x1, self.y1, self.x2, self.y2, side=side)

    def flip_side(self) -> EncroachLine:
        nxt = "negative" if self.side == "positive" else "positive"
        return self.with_side(nxt)

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        return (
            int(round(self.x1 * (width - 1))),
            int(round(self.y1 * (height - 1))),
            int(round(self.x2 * (width - 1))),
            int(round(self.y2 * (height - 1))),
        )


def parse_encroach_line(raw: object) -> tuple[float, float, float, float] | None:
    """Parse YAML list/tuple of four numbers into a clamped line, or None."""
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValueError("encroach_line must be [x1, y1, x2, y2]")
    try:
        vals = [float(v) for v in raw]
    except (TypeError, ValueError) as exc:
        raise ValueError("encroach_line values must be numbers") from exc
    clamped = tuple(max(0.0, min(1.0, v)) for v in vals)
    x1, y1, x2, y2 = clamped
    if abs(x2 - x1) < 1e-6 and abs(y2 - y1) < 1e-6:
        raise ValueError("encroach_line endpoints must differ")
    return (x1, y1, x2, y2)


def next_line_preset(
    current: tuple[float, float, float, float] | None, step: int = 1
) -> tuple[float, float, float, float]:
    presets = [coords for _, coords in LINE_PRESETS]
    if current is None:
        return presets[0]
    try:
        index = next(
            i
            for i, coords in enumerate(presets)
            if all(abs(a - b) < 1e-4 for a, b in zip(coords, current))
        )
    except StopIteration:
        index = -1 if step > 0 else 0
    return presets[(index + int(step)) % len(presets)]


def line_preset_label(line: tuple[float, float, float, float] | None) -> str:
    if line is None:
        return "Horizontal mid (default)"
    for name, coords in LINE_PRESETS:
        if all(abs(a - b) < 1e-4 for a, b in zip(coords, line)):
            return name
    return "Custom"


def signed_side(
    line: EncroachLine, px: float, py: float, *, normalized: bool = True
) -> float:
    """
    Cross-product sign of point vs directed line.

    Positive = left of the directed segment (image y grows downward).
    """
    if normalized:
        x1, y1, x2, y2 = line.as_tuple()
    else:
        # Caller already in pixel space — treat line tuple as pixels.
        x1, y1, x2, y2 = line.as_tuple()
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def point_in_zone(line: EncroachLine, px: float, py: float) -> bool:
    value = signed_side(line, px, py, normalized=True)
    if line.side == "negative":
        return value < 0
    return value > 0


def person_anchor(box: Box, width: int, height: int) -> tuple[float, float]:
    """Bottom-center of person box in normalized frame coords (feet)."""
    mid_x = (box.x1 + box.x2) * 0.5
    foot_y = float(box.y2)
    w = max(1, width - 1)
    h = max(1, height - 1)
    return mid_x / w, foot_y / h


def any_person_in_zone(
    line: EncroachLine,
    people: list[Box],
    width: int,
    height: int,
) -> bool:
    for box in people:
        if box.label != "person":
            continue
        px, py = person_anchor(box, width, height)
        if point_in_zone(line, px, py):
            return True
    return False


def resolve_line(
    coords: tuple[float, float, float, float] | None,
    side: str = "positive",
) -> EncroachLine:
    x1, y1, x2, y2 = coords or DEFAULT_ENCROACH_LINE
    side_norm = (side or "positive").lower()
    if side_norm not in VALID_ENCROACH_SIDES:
        side_norm = "positive"
    return EncroachLine(x1, y1, x2, y2, side=side_norm)


def draw_encroach_line(
    frame: np.ndarray,
    line: EncroachLine,
    *,
    active: bool = False,
) -> np.ndarray:
    """Draw the tripwire + zone hint onto a BGR frame (in place)."""
    if frame is None or frame.size == 0:
        return frame
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = line.to_pixels(w, h)
    color = (40, 80, 255) if active else (60, 160, 255)
    thickness = 3 if active else 2
    cv2.line(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    # Small arrow / tick indicating the "inside" side near the midpoint.
    mx = (x1 + x2) // 2
    my = (y1 + y2) // 2
    dx = x2 - x1
    dy = y2 - y1
    length = max(1.0, (dx * dx + dy * dy) ** 0.5)
    # Perpendicular (left of direction): (-dy, dx)
    nx, ny = -dy / length, dx / length
    if line.side == "negative":
        nx, ny = -nx, -ny
    tip = (int(mx + nx * 18), int(my + ny * 18))
    cv2.arrowedLine(frame, (mx, my), tip, color, 2, tipLength=0.4)
    label = "ZONE" if active else "line"
    cv2.putText(
        frame,
        label,
        (mx + 8, my - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )
    return frame


def draw_encroach_highlight(tile: np.ndarray, *, active: bool = True) -> np.ndarray:
    """Draw a thick alert border on a mosaic tile (in place)."""
    if tile is None or tile.size == 0 or not active:
        return tile
    h, w = tile.shape[:2]
    color = (40, 70, 255)  # strong red-orange in BGR
    thickness = max(4, min(w, h) // 40)
    cv2.rectangle(tile, (0, 0), (w - 1, h - 1), color, thickness)
    # Soft inner wash so the alert reads even on busy scenes.
    inset = thickness + 2
    if w > inset * 2 and h > inset * 2:
        overlay = tile.copy()
        cv2.rectangle(overlay, (0, 0), (w - 1, max(inset * 3, 28)), color, -1)
        cv2.addWeighted(overlay, 0.28, tile, 0.72, 0, tile)
    return tile


def map_tile_click_to_frame_norm(
    px: int,
    py: int,
    *,
    tile_w: int,
    tile_h: int,
    frame_w: int,
    frame_h: int,
    mode: str = "fit",
) -> tuple[float, float] | None:
    """
    Map a click inside a scaled tile back to normalized frame coordinates.

    Returns None when the click lands in letterbox padding (fit mode).
    """
    if tile_w <= 0 or tile_h <= 0 or frame_w <= 0 or frame_h <= 0:
        return None
    if mode == "stretch":
        return (
            min(max(px / tile_w, 0.0), 1.0),
            min(max(py / tile_h, 0.0), 1.0),
        )
    if mode == "fill":
        scale = max(tile_w / frame_w, tile_h / frame_h)
        new_w, new_h = max(1, int(frame_w * scale)), max(1, int(frame_h * scale))
        x0 = max(0, (new_w - tile_w) // 2)
        y0 = max(0, (new_h - tile_h) // 2)
        fx = (px + x0) / new_w
        fy = (py + y0) / new_h
        return min(max(fx, 0.0), 1.0), min(max(fy, 0.0), 1.0)
    # fit (letterbox)
    scale = min(tile_w / frame_w, tile_h / frame_h)
    new_w, new_h = max(1, int(frame_w * scale)), max(1, int(frame_h * scale))
    x0 = (tile_w - new_w) // 2
    y0 = (tile_h - new_h) // 2
    if px < x0 or py < y0 or px >= x0 + new_w or py >= y0 + new_h:
        return None
    fx = (px - x0) / new_w
    fy = (py - y0) / new_h
    return min(max(fx, 0.0), 1.0), min(max(fy, 0.0), 1.0)
