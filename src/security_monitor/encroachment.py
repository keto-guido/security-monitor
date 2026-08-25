"""Per-camera encroachment zones (tripwire lines + polygon ROIs).

People whose feet (bottom-center of the person box) land inside any
configured zone trigger an alert. Coordinates are normalized 0–1 in the
post-rotate source frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from security_monitor.detection import Box

VALID_ENCROACH_SIDES = ("positive", "negative")
VALID_ZONE_KINDS = ("line", "polygon")

# Preset tripwires (x1, y1, x2, y2) in normalized frame space.
LINE_PRESETS: tuple[tuple[str, tuple[float, float, float, float]], ...] = (
    ("Horizontal mid", (0.0, 0.5, 1.0, 0.5)),
    ("Horizontal low", (0.0, 0.66, 1.0, 0.66)),
    ("Horizontal high", (0.0, 0.33, 1.0, 0.33)),
    ("Vertical mid", (0.5, 0.0, 0.5, 1.0)),
    ("Vertical left", (0.33, 0.0, 0.33, 1.0)),
    ("Vertical right", (0.66, 0.0, 0.66, 1.0)),
)

# Preset polygon ROIs (name, points...).
POLYGON_PRESETS: tuple[tuple[str, tuple[tuple[float, float], ...]], ...] = (
    (
        "Bottom half",
        ((0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)),
    ),
    (
        "Bottom third",
        ((0.0, 0.66), (1.0, 0.66), (1.0, 1.0), (0.0, 1.0)),
    ),
    (
        "Center box",
        ((0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)),
    ),
    (
        "Left half",
        ((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)),
    ),
    (
        "Right half",
        ((0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)),
    ),
    (
        "Porch strip",
        ((0.05, 0.55), (0.95, 0.55), (0.95, 0.98), (0.05, 0.98)),
    ),
)

DEFAULT_ENCROACH_LINE: tuple[float, float, float, float] = (0.0, 0.5, 1.0, 0.5)

_ZONE_COLORS = (
    (60, 160, 255),
    (80, 200, 120),
    (220, 160, 60),
    (200, 100, 220),
    (40, 200, 220),
)


@dataclass(frozen=True)
class EncroachZone:
    """One ROI: a directed tripwire (2 points) or a polygon (≥3 points)."""

    name: str
    points: tuple[tuple[float, float], ...]
    side: str = "positive"  # tripwire half-plane only

    @property
    def kind(self) -> str:
        return "line" if len(self.points) == 2 else "polygon"

    @property
    def is_line(self) -> bool:
        return len(self.points) == 2

    @property
    def is_polygon(self) -> bool:
        return len(self.points) >= 3

    def as_line_tuple(self) -> tuple[float, float, float, float] | None:
        if not self.is_line:
            return None
        (x1, y1), (x2, y2) = self.points
        return (x1, y1, x2, y2)


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

    def to_zone(self, name: str = "Tripwire") -> EncroachZone:
        return EncroachZone(
            name=name,
            points=((self.x1, self.y1), (self.x2, self.y2)),
            side=self.side,
        )

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


def parse_zone_points(raw: object, *, minimum: int = 2) -> tuple[tuple[float, float], ...]:
    if not isinstance(raw, (list, tuple)) or len(raw) < minimum:
        raise ValueError(f"zone points need at least {minimum} [x, y] pairs")
    points: list[tuple[float, float]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("each zone point must be [x, y]")
        try:
            x, y = float(item[0]), float(item[1])
        except (TypeError, ValueError) as exc:
            raise ValueError("zone point values must be numbers") from exc
        points.append((max(0.0, min(1.0, x)), max(0.0, min(1.0, y))))
    if minimum >= 3 and len(points) < 3:
        raise ValueError("polygon zones need at least 3 points")
    if len(points) == 2:
        (x1, y1), (x2, y2) = points
        if abs(x2 - x1) < 1e-6 and abs(y2 - y1) < 1e-6:
            raise ValueError("line endpoints must differ")
    return tuple(points)


def parse_encroach_zone(raw: object, index: int = 0) -> EncroachZone:
    if not isinstance(raw, dict):
        raise ValueError("each encroach zone must be a mapping")
    name = str(raw.get("name") or f"Zone {index + 1}").strip() or f"Zone {index + 1}"
    kind = str(raw.get("kind") or "").strip().lower()
    side = str(raw.get("side") or "positive").strip().lower()
    if side not in VALID_ENCROACH_SIDES:
        raise ValueError(f"zone side must be one of {VALID_ENCROACH_SIDES}")
    if "points" in raw and raw.get("points") is not None:
        minimum = 3 if kind == "polygon" else 2
        points = parse_zone_points(raw.get("points"), minimum=minimum)
    elif "line" in raw and raw.get("line") is not None:
        line = parse_encroach_line(raw.get("line"))
        if line is None:
            raise ValueError("zone line is empty")
        points = ((line[0], line[1]), (line[2], line[3]))
    else:
        raise ValueError("zone needs points: [[x,y], ...] or line: [x1,y1,x2,y2]")
    if kind == "line" and len(points) != 2:
        raise ValueError("kind=line requires exactly 2 points")
    if kind == "polygon" and len(points) < 3:
        raise ValueError("kind=polygon requires at least 3 points")
    if kind and kind not in VALID_ZONE_KINDS and kind != "auto":
        raise ValueError(f"zone kind must be one of {VALID_ZONE_KINDS}")
    return EncroachZone(name=name, points=points, side=side)


def parse_encroach_zones(raw: object) -> list[EncroachZone]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("encroach_zones must be a list")
    return [parse_encroach_zone(item, i) for i, item in enumerate(raw)]


def zone_to_dict(zone: EncroachZone) -> dict:
    item: dict = {
        "name": zone.name,
        "kind": zone.kind,
        "points": [[float(x), float(y)] for x, y in zone.points],
    }
    if zone.is_line and zone.side != "positive":
        item["side"] = zone.side
    elif zone.is_line:
        item["side"] = zone.side
    return item


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


def next_polygon_preset(
    current: tuple[tuple[float, float], ...] | None, step: int = 1
) -> tuple[str, tuple[tuple[float, float], ...]]:
    if current is None:
        index = -1 if step > 0 else 0
    else:
        try:
            index = next(
                i
                for i, (_, pts) in enumerate(POLYGON_PRESETS)
                if len(pts) == len(current)
                and all(
                    abs(a[0] - b[0]) < 1e-4 and abs(a[1] - b[1]) < 1e-4
                    for a, b in zip(pts, current)
                )
            )
        except StopIteration:
            index = -1 if step > 0 else 0
    return POLYGON_PRESETS[(index + int(step)) % len(POLYGON_PRESETS)]


def line_preset_label(line: tuple[float, float, float, float] | None) -> str:
    if line is None:
        return "Horizontal mid (default)"
    for name, coords in LINE_PRESETS:
        if all(abs(a - b) < 1e-4 for a, b in zip(coords, line)):
            return name
    return "Custom"


def polygon_preset_label(points: tuple[tuple[float, float], ...] | None) -> str:
    if points is None:
        return "Bottom half"
    for name, pts in POLYGON_PRESETS:
        if len(pts) == len(points) and all(
            abs(a[0] - b[0]) < 1e-4 and abs(a[1] - b[1]) < 1e-4
            for a, b in zip(pts, points)
        ):
            return name
    return "Custom"


def signed_side(line: EncroachLine, px: float, py: float) -> float:
    """Cross-product sign of point vs directed line (image y grows downward)."""
    x1, y1, x2, y2 = line.as_tuple()
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def point_in_halfplane(line: EncroachLine, px: float, py: float) -> bool:
    value = signed_side(line, px, py)
    if line.side == "negative":
        return value < 0
    return value > 0


# Back-compat alias used by older tests / callers.
point_in_zone = point_in_halfplane


def point_in_polygon(points: Sequence[tuple[float, float]], px: float, py: float) -> bool:
    """Ray-casting point-in-polygon test in normalized coordinates."""
    if len(points) < 3:
        return False
    inside = False
    j = len(points) - 1
    for i, (xi, yi) in enumerate(points):
        xj, yj = points[j]
        intersects = ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def person_anchor(box: Box, width: int, height: int) -> tuple[float, float]:
    """Bottom-center of person box in normalized frame coords (feet)."""
    mid_x = (box.x1 + box.x2) * 0.5
    foot_y = float(box.y2)
    w = max(1, width - 1)
    h = max(1, height - 1)
    return mid_x / w, foot_y / h


def zone_contains(zone: EncroachZone, px: float, py: float) -> bool:
    if zone.is_line:
        (x1, y1), (x2, y2) = zone.points
        return point_in_halfplane(EncroachLine(x1, y1, x2, y2, side=zone.side), px, py)
    return point_in_polygon(zone.points, px, py)


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
        if point_in_halfplane(line, px, py):
            return True
    return False


def evaluate_zones(
    zones: Sequence[EncroachZone],
    people: list[Box],
    width: int,
    height: int,
) -> tuple[bool, tuple[str, ...]]:
    """Return (any_hit, names of zones containing a person)."""
    if not zones:
        return False, ()
    hits: list[str] = []
    for zone in zones:
        for box in people:
            if box.label != "person":
                continue
            px, py = person_anchor(box, width, height)
            if zone_contains(zone, px, py):
                hits.append(zone.name)
                break
    return bool(hits), tuple(hits)


def resolve_line(
    coords: tuple[float, float, float, float] | None,
    side: str = "positive",
) -> EncroachLine:
    x1, y1, x2, y2 = coords or DEFAULT_ENCROACH_LINE
    side_norm = (side or "positive").lower()
    if side_norm not in VALID_ENCROACH_SIDES:
        side_norm = "positive"
    return EncroachLine(x1, y1, x2, y2, side=side_norm)


def default_zones_from_line(
    coords: tuple[float, float, float, float] | None = None,
    side: str = "positive",
) -> list[EncroachZone]:
    line = resolve_line(coords, side)
    return [line.to_zone("Tripwire")]


def effective_zones(
    zones: Sequence[EncroachZone] | None,
    *,
    legacy_line: tuple[float, float, float, float] | None = None,
    legacy_side: str = "positive",
) -> list[EncroachZone]:
    """Prefer explicit zones; fall back to legacy single tripwire."""
    if zones:
        return list(zones)
    return default_zones_from_line(legacy_line, legacy_side)


def unique_zone_name(zones: Sequence[EncroachZone], base: str) -> str:
    base = (base or "Zone").strip() or "Zone"
    taken = {z.name.strip().lower() for z in zones}
    if base.lower() not in taken:
        return base
    for n in range(2, 10000):
        candidate = f"{base} {n}"
        if candidate.lower() not in taken:
            return candidate
    return f"{base} copy"


def draw_encroach_line(
    frame: np.ndarray,
    line: EncroachLine,
    *,
    active: bool = False,
    label: str = "line",
    color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Draw the tripwire + zone hint onto a BGR frame (in place)."""
    if frame is None or frame.size == 0:
        return frame
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = line.to_pixels(w, h)
    color = color or ((40, 80, 255) if active else (60, 160, 255))
    thickness = 3 if active else 2
    cv2.line(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    mx = (x1 + x2) // 2
    my = (y1 + y2) // 2
    dx = x2 - x1
    dy = y2 - y1
    length = max(1.0, (dx * dx + dy * dy) ** 0.5)
    nx, ny = -dy / length, dx / length
    if line.side == "negative":
        nx, ny = -nx, -ny
    tip = (int(mx + nx * 18), int(my + ny * 18))
    cv2.arrowedLine(frame, (mx, my), tip, color, 2, tipLength=0.4)
    text = "ZONE" if active else label
    cv2.putText(
        frame,
        text,
        (mx + 8, my - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )
    return frame


def draw_encroach_polygon(
    frame: np.ndarray,
    points: Sequence[tuple[float, float]],
    *,
    active: bool = False,
    label: str = "ROI",
    color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    if frame is None or frame.size == 0 or len(points) < 3:
        return frame
    h, w = frame.shape[:2]
    color = color or ((40, 80, 255) if active else (60, 160, 255))
    pts = np.array(
        [
            [int(round(x * (w - 1))), int(round(y * (h - 1)))]
            for x, y in points
        ],
        dtype=np.int32,
    )
    overlay = frame.copy()
    fill_alpha = 0.35 if active else 0.18
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, fill_alpha, frame, 1.0 - fill_alpha, 0, frame)
    cv2.polylines(frame, [pts], True, color, 3 if active else 2, cv2.LINE_AA)
    cx = int(pts[:, 0].mean())
    cy = int(pts[:, 1].mean())
    text = f"ZONE {label}" if active else label
    cv2.putText(
        frame,
        text,
        (cx - 20, cy),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )
    return frame


def draw_zones(
    frame: np.ndarray,
    zones: Sequence[EncroachZone],
    *,
    active_names: Sequence[str] | None = None,
    draft_points: Sequence[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Draw all zones (and optional in-progress polygon) onto a frame."""
    if frame is None or frame.size == 0:
        return frame
    active = set(active_names or ())
    for index, zone in enumerate(zones):
        color = _ZONE_COLORS[index % len(_ZONE_COLORS)]
        hit = zone.name in active
        if zone.is_line:
            (x1, y1), (x2, y2) = zone.points
            draw_encroach_line(
                frame,
                EncroachLine(x1, y1, x2, y2, side=zone.side),
                active=hit,
                label=zone.name,
                color=color if not hit else (40, 80, 255),
            )
        else:
            draw_encroach_polygon(
                frame,
                zone.points,
                active=hit,
                label=zone.name,
                color=color if not hit else (40, 80, 255),
            )
    if draft_points:
        h, w = frame.shape[:2]
        pts = [
            (int(round(x * (w - 1))), int(round(y * (h - 1))))
            for x, y in draft_points
        ]
        for i, (x, y) in enumerate(pts):
            cv2.circle(frame, (x, y), 5, (60, 200, 255), -1, cv2.LINE_AA)
            if i > 0:
                cv2.line(frame, pts[i - 1], (x, y), (60, 200, 255), 2, cv2.LINE_AA)
        if len(pts) >= 3:
            cv2.line(frame, pts[-1], pts[0], (60, 200, 255), 1, cv2.LINE_AA)
    return frame


def draw_encroach_highlight(
    tile: np.ndarray,
    *,
    active: bool = True,
    pulse: float = 1.0,
    strong: bool = False,
) -> np.ndarray:
    """Draw a thick alert border on a mosaic tile (in place)."""
    if tile is None or tile.size == 0 or not active:
        return tile
    h, w = tile.shape[:2]
    pulse = max(0.35, min(1.0, float(pulse)))
    color = (40, 70, 255)
    thickness = max(4, min(w, h) // (28 if strong else 40))
    if strong:
        thickness = max(thickness, 6)
    cv2.rectangle(tile, (0, 0), (w - 1, h - 1), color, thickness)
    inset = thickness + 2
    if w > inset * 2 and h > inset * 2:
        overlay = tile.copy()
        wash_h = max(inset * 3, int(h * (0.18 if strong else 0.12)))
        cv2.rectangle(overlay, (0, 0), (w - 1, wash_h), color, -1)
        if strong:
            cv2.rectangle(overlay, (0, h - wash_h), (w - 1, h - 1), color, -1)
        alpha = (0.42 if strong else 0.28) * pulse
        cv2.addWeighted(overlay, alpha, tile, 1.0 - alpha, 0, tile)
    return tile


def draw_alarm_banner(
    canvas: np.ndarray,
    labels: Sequence[str],
    *,
    pulse: float = 1.0,
) -> np.ndarray:
    """Full-mosaic encroachment alarm: pulsing vignette + banner."""
    if canvas is None or canvas.size == 0 or not labels:
        return canvas
    h, w = canvas.shape[:2]
    pulse = max(0.4, min(1.0, float(pulse)))
    color = (30, 50, 255)
    overlay = canvas.copy()
    edge = max(18, min(w, h) // 16)
    cv2.rectangle(overlay, (0, 0), (w - 1, edge), color, -1)
    cv2.rectangle(overlay, (0, h - edge), (w - 1, h - 1), color, -1)
    cv2.rectangle(overlay, (0, 0), (edge, h - 1), color, -1)
    cv2.rectangle(overlay, (w - edge, 0), (w - 1, h - 1), color, -1)
    cv2.addWeighted(overlay, 0.55 * pulse, canvas, 1.0 - 0.55 * pulse, 0, canvas)
    thickness = max(4, edge // 3)
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), color, thickness)
    names = ", ".join(labels[:4])
    if len(labels) > 4:
        names += f" +{len(labels) - 4}"
    text = f"ENCROACHMENT  —  {names}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.7, min(1.6, w / 700))
    (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
    tx = max(12, (w - tw) // 2)
    ty = max(th + 16, edge + th + 8)
    cv2.rectangle(
        canvas,
        (tx - 12, ty - th - 12),
        (tx + tw + 12, ty + 12),
        (10, 10, 30),
        -1,
    )
    cv2.putText(canvas, text, (tx, ty), font, scale, (220, 230, 255), 2, cv2.LINE_AA)
    return canvas


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
