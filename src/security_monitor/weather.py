"""Local weather HUD: Open-Meteo fetch, layout reservation, and widget drawing."""

from __future__ import annotations

import json
import math
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from security_monitor.overlay import draw_text, shade_round_rect

VALID_WEATHER_SLOTS = (
    "bottom_left",
    "bottom_right",
    "between_h",
    "between_v",
    "custom",
)
VALID_WEATHER_UNITS = ("f", "c")

# WMO weather interpretation codes (Open-Meteo) that imply thunder / lightning risk.
_THUNDER_CODES = frozenset({95, 96, 97, 98, 99})
_STORMY_CODES = frozenset({65, 67, 75, 82, 86, *_THUNDER_CODES})

_WMO_LABELS = {
    0: "Clear",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Rime fog",
    51: "Light drizzle",
    53: "Drizzle",
    55: "Heavy drizzle",
    61: "Light rain",
    63: "Rain",
    65: "Heavy rain",
    66: "Freezing rain",
    67: "Heavy frz rain",
    71: "Light snow",
    73: "Snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Rain showers",
    81: "Rain showers",
    82: "Violent showers",
    85: "Snow showers",
    86: "Heavy snow shwr",
    95: "Thunderstorm",
    96: "Thunder + hail",
    97: "Heavy thunder",
    98: "Thunder + hail",
    99: "Severe thunder",
}


@dataclass(frozen=True)
class WeatherSnapshot:
    temperature_c: float | None = None
    weather_code: int | None = None
    wind_kmh: float | None = None
    precip_mm: float | None = None
    condition: str = ""
    storm_warning: str = ""
    lightning_risk: str = ""  # None / Low / Elevated / High
    lightning_detail: str = ""
    latitude: float | None = None
    longitude: float | None = None
    place: str = ""
    updated_at: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and self.temperature_c is not None

    def temp_label(self, units: str = "f") -> str:
        if self.temperature_c is None:
            return "--"
        if (units or "f").lower().startswith("c"):
            return f"{self.temperature_c:.0f}°C"
        f = self.temperature_c * 9.0 / 5.0 + 32.0
        return f"{f:.0f}°F"


@dataclass
class WeatherRect:
    """Pixel rectangle on the mosaic canvas (x, y, w, h)."""

    x: int
    y: int
    w: int
    h: int

    @property
    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def clamp(self, width: int, height: int) -> WeatherRect:
        w = max(40, min(self.w, width))
        h = max(36, min(self.h, height))
        x = max(0, min(self.x, width - w))
        y = max(0, min(self.y, height - h))
        return WeatherRect(x, y, w, h)


def next_weather_slot(current: str, step: int = 1) -> str:
    values = list(VALID_WEATHER_SLOTS)
    try:
        index = values.index((current or "bottom_right").lower())
    except ValueError:
        index = 0
    return values[(index + int(step)) % len(values)]


def slot_label(slot: str) -> str:
    return {
        "bottom_left": "Bottom left",
        "bottom_right": "Bottom right",
        "between_h": "Between columns",
        "between_v": "Between rows",
        "custom": "Custom",
    }.get((slot or "").lower(), slot or "Custom")


def wmo_label(code: int | None) -> str:
    if code is None:
        return "—"
    return _WMO_LABELS.get(int(code), f"Code {code}")


def _http_json(url: str, timeout: float = 8.0) -> dict:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "security-monitor/0.2 (local weather hud)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def detect_location() -> tuple[float, float, str]:
    """Best-effort lat/lon via IP geolocation."""
    try:
        data = _http_json("http://ip-api.com/json/?fields=status,lat,lon,city,regionName", timeout=5)
        if data.get("status") == "success":
            city = str(data.get("city") or "")
            region = str(data.get("regionName") or "")
            place = ", ".join(p for p in (city, region) if p) or "Local"
            return float(data["lat"]), float(data["lon"]), place
    except (OSError, urllib.error.URLError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        pass
    # Fallback: continental US centroid (user should set lat/lon).
    return 39.8283, -98.5795, "Set location"


def _lightning_from_code(code: int | None, precip_mm: float | None) -> tuple[str, str]:
    if code is None:
        return "Unknown", ""
    code = int(code)
    if code in _THUNDER_CODES:
        detail = wmo_label(code)
        return "High", detail
    if code in {80, 81, 82} and (precip_mm or 0) >= 2.0:
        return "Elevated", "Heavy showers — watch for lightning"
    if code in _STORMY_CODES:
        return "Elevated", wmo_label(code)
    if code >= 61:
        return "Low", wmo_label(code)
    return "None", ""


def _fetch_warnings(lat: float, lon: float) -> str:
    """Open-Meteo warnings API (regional coverage varies)."""
    try:
        qs = urllib.parse.urlencode(
            {
                "latitude": f"{lat:.4f}",
                "longitude": f"{lon:.4f}",
            }
        )
        data = _http_json(f"https://api.open-meteo.com/v1/warnings?{qs}", timeout=6)
    except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError):
        return ""
    warnings = data.get("warnings") or data.get("warning") or []
    if isinstance(warnings, dict):
        warnings = [warnings]
    if not isinstance(warnings, list) or not warnings:
        return ""
    parts: list[str] = []
    for item in warnings[:3]:
        if not isinstance(item, dict):
            continue
        event = str(item.get("event") or item.get("headline") or item.get("type") or "").strip()
        severity = str(item.get("severity") or item.get("level") or "").strip()
        if event and severity:
            parts.append(f"{severity}: {event}")
        elif event:
            parts.append(event)
    return " · ".join(parts)[:160]


def fetch_weather(
    latitude: float | None,
    longitude: float | None,
    *,
    place: str = "",
) -> WeatherSnapshot:
    """Fetch current conditions (+ warnings) from Open-Meteo."""
    loc_place = place
    try:
        if latitude is None or longitude is None:
            latitude, longitude, loc_place = detect_location()
        qs = urllib.parse.urlencode(
            {
                "latitude": f"{float(latitude):.4f}",
                "longitude": f"{float(longitude):.4f}",
                "current": "temperature_2m,weather_code,precipitation,wind_speed_10m",
                "wind_speed_unit": "kmh",
                "timezone": "auto",
            }
        )
        data = _http_json(f"https://api.open-meteo.com/v1/forecast?{qs}")
        current = data.get("current") or {}
        temp = current.get("temperature_2m")
        code = current.get("weather_code")
        wind = current.get("wind_speed_10m")
        precip = current.get("precipitation")
        temp_f = float(temp) if temp is not None else None
        code_i = int(code) if code is not None else None
        precip_f = float(precip) if precip is not None else None
        wind_f = float(wind) if wind is not None else None
        risk, detail = _lightning_from_code(code_i, precip_f)
        warning = _fetch_warnings(float(latitude), float(longitude))
        if not warning and code_i in _THUNDER_CODES:
            warning = f"Thunderstorm conditions ({wmo_label(code_i)})"
        elif not warning and code_i in _STORMY_CODES:
            warning = f"Stormy weather ({wmo_label(code_i)})"
        return WeatherSnapshot(
            temperature_c=temp_f,
            weather_code=code_i,
            wind_kmh=wind_f,
            precip_mm=precip_f,
            condition=wmo_label(code_i),
            storm_warning=warning,
            lightning_risk=risk,
            lightning_detail=detail,
            latitude=float(latitude),
            longitude=float(longitude),
            place=loc_place or place or "Local",
            updated_at=time.time(),
        )
    except Exception as exc:  # noqa: BLE001
        return WeatherSnapshot(error=str(exc)[:120], updated_at=time.time(), place=loc_place or place)


class WeatherService:
    """Background refresher for the weather HUD."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = WeatherSnapshot(error="Waiting for first update…")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._interval = 300.0
        self._lat: float | None = None
        self._lon: float | None = None
        self._place = ""
        self._enabled = False

    @property
    def snapshot(self) -> WeatherSnapshot:
        with self._lock:
            return self._snapshot

    def configure(
        self,
        *,
        enabled: bool,
        latitude: float | None,
        longitude: float | None,
        place: str = "",
        refresh_seconds: float = 300.0,
    ) -> None:
        self._enabled = bool(enabled)
        self._lat = latitude
        self._lon = longitude
        self._place = place or ""
        self._interval = max(60.0, float(refresh_seconds))
        if enabled and (self._thread is None or not self._thread.is_alive()):
            self.start()
        if enabled:
            # Kick an immediate refresh when toggled on / location changes.
            threading.Thread(target=self._refresh_once, name="weather-kick", daemon=True).start()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="weather-service", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _refresh_once(self) -> None:
        if not self._enabled:
            return
        snap = fetch_weather(self._lat, self._lon, place=self._place)
        with self._lock:
            self._snapshot = snap

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._enabled:
                self._refresh_once()
            self._stop.wait(self._interval)


def resolve_weather_rect(
    *,
    slot: str,
    norm_x: float,
    norm_y: float,
    norm_w: float,
    norm_h: float,
    grid_x: int,
    grid_y: int,
    grid_w: int,
    grid_h: int,
    columns: int,
    rows: int,
    canvas_w: int,
    canvas_h: int,
) -> WeatherRect:
    """
    Resolve widget placement inside the camera grid.

    norm_x/y are:
      - custom: top-left of widget in grid-normalized 0–1 space
      - presets: fine-tune offsets added to the preset anchor (−0.5…0.5 typical)
    """
    slot = (slot or "bottom_right").lower()
    nw = max(0.12, min(0.55, float(norm_w) or 0.22))
    nh = max(0.10, min(0.45, float(norm_h) or 0.18))
    ox = float(norm_x)
    oy = float(norm_y)

    if slot == "custom":
        left = ox
        top = oy
    elif slot == "bottom_left":
        left = 0.02 + ox
        top = 1.0 - nh - 0.02 + oy
    elif slot == "bottom_right":
        left = 1.0 - nw - 0.02 + ox
        top = 1.0 - nh - 0.02 + oy
    elif slot == "between_h":
        # Straddle the vertical gutter between the first two columns (or center).
        col_split = 0.5 if columns <= 1 else (1.0 / columns)
        left = col_split - nw * 0.5 + ox
        top = 0.5 - nh * 0.5 + oy
    elif slot == "between_v":
        row_split = 0.5 if rows <= 1 else (1.0 / rows)
        left = 0.5 - nw * 0.5 + ox
        top = row_split - nh * 0.5 + oy
    else:
        left = 1.0 - nw - 0.02 + ox
        top = 1.0 - nh - 0.02 + oy

    left = max(0.0, min(1.0 - nw, left))
    top = max(0.0, min(1.0 - nh, top))
    x = int(round(grid_x + left * grid_w))
    y = int(round(grid_y + top * grid_h))
    w = max(48, int(round(nw * grid_w)))
    h = max(40, int(round(nh * grid_h)))
    return WeatherRect(x, y, w, h).clamp(canvas_w, canvas_h)


def shrink_tile_from_reserved(
    tile: tuple[int, int, int, int],
    reserved: WeatherRect,
) -> tuple[int, int, int, int]:
    """
    Shrink a camera tile so it no longer overlaps the weather widget.

    When the widget sits on a shared edge, each abutting tile loses only its
    overlapping strip — space is taken from both cameras rather than one.
    """
    x, y, w, h = tile
    if w <= 1 or h <= 1:
        return tile
    rx, ry, rw, rh = reserved.as_tuple
    ix1 = max(x, rx)
    iy1 = max(y, ry)
    ix2 = min(x + w, rx + rw)
    iy2 = min(y + h, ry + rh)
    if ix1 >= ix2 or iy1 >= iy2:
        return tile

    tcx, tcy = x + w * 0.5, y + h * 0.5
    rcx, rcy = rx + rw * 0.5, ry + rh * 0.5
    ow, oh = ix2 - ix1, iy2 - iy1

    # Prefer cutting along the thinner overlap axis so both neighbors share loss.
    if ow <= oh:
        if rcx >= tcx:
            new_w = max(1, ix1 - x)
            return (x, y, new_w, h)
        new_x = ix2
        new_w = max(1, x + w - ix2)
        return (new_x, y, new_w, h)
    if rcy >= tcy:
        new_h = max(1, iy1 - y)
        return (x, y, w, new_h)
    new_y = iy2
    new_h = max(1, y + h - iy2)
    return (x, new_y, w, new_h)


def compute_tile_rects(
    *,
    columns: int,
    rows: int,
    cell_w: int,
    cell_h: int,
    grid_x: int,
    grid_y: int,
    reserved: WeatherRect | None,
) -> list[tuple[int, int, int, int]]:
    """Per-tile destination rects, optionally carved around the weather widget."""
    rects: list[tuple[int, int, int, int]] = []
    for index in range(columns * rows):
        row, col = divmod(index, columns)
        rect = (grid_x + col * cell_w, grid_y + row * cell_h, cell_w, cell_h)
        if reserved is not None:
            rect = shrink_tile_from_reserved(rect, reserved)
        rects.append(rect)
    return rects


def draw_weather_widget(
    canvas: np.ndarray,
    rect: WeatherRect,
    snap: WeatherSnapshot,
    *,
    units: str = "f",
    show_temp: bool = True,
    show_conditions: bool = True,
    show_storm: bool = True,
    show_lightning: bool = True,
    editing: bool = False,
) -> None:
    """Paint the weather HUD into ``rect`` on the mosaic canvas."""
    if canvas is None or canvas.size == 0:
        return
    x, y, w, h = rect.as_tuple
    if w < 20 or h < 20:
        return
    shade_round_rect(
        canvas,
        (x, y, x + w, y + h),
        color=(22, 24, 32),
        alpha=0.88,
        radius=12,
    )
    # Accent bar — redder when storm/lightning elevated.
    accent = (70, 160, 90)
    if snap.lightning_risk in {"High", "Elevated"} or snap.storm_warning:
        accent = (40, 80, 255) if snap.lightning_risk == "High" else (40, 150, 230)
    if editing:
        accent = (60, 180, 255)
        cv2.rectangle(canvas, (x, y), (x + w - 1, y + h - 1), accent, 2)

    bar_h = max(4, h // 18)
    cv2.rectangle(canvas, (x + 4, y + 4), (x + w - 4, y + 4 + bar_h), accent, -1)

    pad = max(8, w // 24)
    cursor_y = y + bar_h + pad + 2
    title = snap.place or "Local weather"
    if editing:
        title = "Weather widget — drag to place"
    draw_text(
        canvas,
        title[:28],
        (x + pad, cursor_y),
        size=max(12, min(16, h // 10)),
        color=(200, 205, 215),
        valign="top",
    )
    cursor_y += max(18, h // 8)

    lines: list[tuple[str, tuple[int, int, int]]] = []
    if show_temp:
        temp = snap.temp_label(units) if snap.ok else "--"
        lines.append((temp, (235, 235, 240)))
    if show_conditions and snap.condition:
        lines.append((snap.condition, (180, 190, 200)))
    elif show_conditions and snap.error:
        lines.append((snap.error[:36], (90, 90, 220)))
    if show_storm:
        if snap.storm_warning:
            lines.append((f"⚠ {snap.storm_warning[:42]}", (70, 120, 255)))
        else:
            lines.append(("Storm: none", (120, 140, 130)))
    if show_lightning:
        risk = snap.lightning_risk or "—"
        color = (180, 190, 200)
        if risk == "High":
            color = (60, 80, 255)
        elif risk == "Elevated":
            color = (60, 170, 240)
        detail = f" — {snap.lightning_detail}" if snap.lightning_detail else ""
        lines.append((f"Lightning: {risk}{detail}"[:44], color))

    for text, color in lines:
        if cursor_y > y + h - 16:
            break
        size = max(13, min(28 if show_temp and text == lines[0][0] else 15, h // 7))
        draw_text(canvas, text, (x + pad, cursor_y), size=size, color=color, valign="top")
        cursor_y += size + max(4, h // 28)

    if snap.ok and snap.updated_at:
        age = max(0, int(time.time() - snap.updated_at))
        age_txt = f"{age}s ago" if age < 120 else f"{age // 60}m ago"
        draw_text(
            canvas,
            age_txt,
            (x + w - pad, y + h - 8),
            size=11,
            color=(120, 125, 135),
            align="right",
            valign="bottom",
        )


def nudge_norm(value: float, step: float, lo: float = -0.45, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value) + float(step)))
