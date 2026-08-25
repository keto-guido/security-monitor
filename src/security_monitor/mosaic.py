"""OpenCV mosaic window: grid of camera tiles, zoom, overlays."""

from __future__ import annotations

import sys
import time

import cv2
import numpy as np

from security_monitor.config import AppConfig
from security_monitor.overlay import draw_dot, draw_text, shade_bottom_bar, shade_round_rect
from security_monitor.stream import Snapshot, build_sources

HELP_LINES = (
    "q / ESC  quit",
    "f        fullscreen",
    "g / 0    grid view",
    "1-9      focus camera",
    "wheel    zoom in/out",
    "+ / -    zoom in/out",
    "arrows   pan / strafe",
    "Home     reset zoom",
    "r        reconnect all",
    "click    focus tile",
)

ZOOM_MIN = 1.0
ZOOM_MAX = 12.0
ZOOM_FACTOR = 1.2
PAN_VIEW_FRACTION = 0.18

# waitKeyEx codes differ by GUI backend (Win32 / GTK / Qt).
KEY_LEFT = frozenset({2424832, 65361, 81, 2, 0xFF51, 16777234})
KEY_UP = frozenset({2490368, 65362, 82, 0xFF52, 16777235})
KEY_RIGHT = frozenset({2555904, 65363, 83, 3, 0xFF53, 16777236})
KEY_DOWN = frozenset({2621440, 65364, 84, 1, 0xFF54, 16777237})
KEY_HOME = frozenset({2359296, 65360, 16777232, 0xFF50})
KEY_PLUS = frozenset({ord("+"), ord("=")})
KEY_MINUS = frozenset({ord("-"), ord("_")})

STATUS_COLOR = {
    "live": (70, 190, 90),
    "demo": (70, 190, 90),
    "reconnecting": (40, 180, 220),
    "disconnected": (50, 50, 200),
    "error": (40, 40, 220),
}


class MosaicApp:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.display = config.display
        self.cameras = config.visible_cameras()
        self.sources = build_sources(self.cameras, self.display)
        self.zoom_index: int | None = None
        self.fullscreen = self.display.fullscreen
        self.window = self.display.window_title
        self._running = True
        self._ui_fps = 0.0
        self._show_help = False
        self._fps_shown: dict[str, tuple[float, int]] = {}
        self.view_zoom = 1.0
        self.pan_x = 0.5
        self.pan_y = 0.5

    def run(self) -> int:
        if not self.sources:
            print("No enabled cameras in the current grid. Edit config.yaml.", file=sys.stderr)
            return 1
        for source in self.sources:
            source.start()
            print(f"Started {source.name}")

        try:
            cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        except cv2.error:
            print(
                "OpenCV could not create a window. Install the GUI build:\n"
                "  pip install opencv-python\n"
                "(not opencv-python-headless)",
                file=sys.stderr,
            )
            self._shutdown()
            return 1

        width, height = self.display.canvas_size
        cv2.resizeWindow(self.window, width, height)
        cv2.setMouseCallback(self.window, self._on_mouse)
        self._apply_fullscreen()
        print(
            "Controls: q quit | f fullscreen | 1-9 focus | g grid | "
            "wheel/+/- zoom | arrows pan | Home reset zoom | h help"
        )

        delay = max(1, int(1000 / self.display.fps))
        stamps: list[float] = []
        try:
            while self._running:
                canvas = self._compose()
                cv2.imshow(self.window, canvas)
                now = time.monotonic()
                stamps.append(now)
                stamps = [t for t in stamps if now - t <= 1.5]
                if len(stamps) >= 2:
                    self._ui_fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])
                key = cv2.waitKeyEx(delay)
                if key >= 0:
                    self._handle_key(key)
                try:
                    if cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            self._shutdown()
        return 0

    def _shutdown(self) -> None:
        for source in self.sources:
            source.stop()
        cv2.destroyAllWindows()

    def _compose(self) -> np.ndarray:
        d = self.display
        if self.zoom_index is not None and 0 <= self.zoom_index < len(self.sources):
            snap = self.sources[self.zoom_index].snapshot()
            name = self.sources[self.zoom_index].name
            width, height = d.columns * d.cell_width, d.rows * d.cell_height
            cell = self._render_cell(snap, name, width, height, overlay=False)
            cell = magnify(cell, self.view_zoom, self.pan_x, self.pan_y)
            self._draw_cell_overlay(cell, snap, name)
            self._draw_zoom_badge(cell)
            return self._draw_help(cell)

        canvas = np.zeros((d.rows * d.cell_height, d.columns * d.cell_width, 3), dtype=np.uint8)
        canvas[:] = (12, 12, 14)
        zoomed = self.view_zoom > 1.001
        for index in range(d.tile_count):
            row, col = divmod(index, d.columns)
            y, x = row * d.cell_height, col * d.cell_width
            if index < len(self.sources):
                snap = self.sources[index].snapshot()
                tile = self._render_cell(
                    snap,
                    self.sources[index].name,
                    d.cell_width,
                    d.cell_height,
                    overlay=not zoomed,
                )
            else:
                tile = placeholder(d.cell_width, d.cell_height, "Empty", "No camera assigned")
            canvas[y : y + d.cell_height, x : x + d.cell_width] = tile
        if not zoomed:
            self._draw_grid_lines(canvas)
        canvas = magnify(canvas, self.view_zoom, self.pan_x, self.pan_y)
        self._draw_zoom_badge(canvas)
        return self._draw_help(canvas)

    def _draw_grid_lines(self, canvas: np.ndarray) -> None:
        d = self.display
        tint = np.array([18, 18, 20], dtype=np.float32)
        alpha = 0.45
        for col in range(1, d.columns):
            x = col * d.cell_width
            col_pixels = canvas[:, x].astype(np.float32)
            canvas[:, x] = (col_pixels * (1.0 - alpha) + tint * alpha).astype(np.uint8)
        for row in range(1, d.rows):
            y = row * d.cell_height
            row_pixels = canvas[y, :].astype(np.float32)
            canvas[y, :] = (row_pixels * (1.0 - alpha) + tint * alpha).astype(np.uint8)

    def _render_cell(self, snap: Snapshot, name: str, width: int, height: int) -> np.ndarray:
        if snap.frame is None:
            message = snap.detail or snap.status.replace("_", " ")
            tile = placeholder(width, height, name, message.upper() or "NO SIGNAL")
        else:
            tile = scale_frame(snap.frame, width, height, self.display.scale_mode)
        if self.display.show_labels:
            fps_text = None
            if self.display.show_fps:
                shown = self._stable_fps(name, snap.fps)
                fps_text = f"{shown:2d} fps" if shown else "-- fps"
            draw_status_bar(tile, name, snap, fps_text)
        return tile

    def _stable_fps(self, key: str, fps: float) -> int:
        now = time.monotonic()
        value = int(round(fps)) if fps > 0.5 else 0
        prev = self._fps_shown.get(key)
        if prev is None or now - prev[0] >= 0.5 or abs(value - prev[1]) >= 4:
            self._fps_shown[key] = (now, value)
            return value
        return prev[1]

    def _draw_help(self, canvas: np.ndarray) -> np.ndarray:
        if not self._show_help:
            return canvas
        ui = self._stable_fps("_ui", self._ui_fps)
        lines = (f"Controls  (h to hide)   UI {ui} fps", *HELP_LINES)
        box_h = 20 + 22 * len(lines)
        shade_round_rect(canvas, (16, 16, 352, 16 + box_h), alpha=0.82, radius=12)
        y = 28
        for i, line in enumerate(lines):
            draw_text(
                canvas,
                line,
                (32, y),
                size=14 if i == 0 else 13,
                color=(232, 232, 232) if i == 0 else (198, 198, 198),
                valign="top",
            )
            y += 22
        return canvas

    def _handle_key(self, key: int) -> None:
        if key in (ord("q"), ord("Q"), 27):
            self._running = False
        elif key in (ord("f"), ord("F")):
            self.fullscreen = not self.fullscreen
            self._apply_fullscreen()
        elif key in (ord("g"), ord("G"), ord("0")):
            self.zoom_index = None
        elif key in (ord("r"), ord("R")):
            print("Reconnecting all cameras...")
            for source in self.sources:
                source.reconnect()
        elif key in (ord("h"), ord("H"), ord("?")):
            self._show_help = not self._show_help
        elif ord("1") <= key <= ord("9"):
            index = key - ord("1")
            if index < len(self.sources):
                self.zoom_index = index

    def _apply_fullscreen(self) -> None:
        prop = getattr(cv2, "WND_PROP_FULLSCREEN", None)
        mode_full = getattr(cv2, "WINDOW_FULLSCREEN", 1)
        mode_normal = getattr(cv2, "WINDOW_NORMAL", 0)
        if prop is None:
            return
        cv2.setWindowProperty(self.window, prop, mode_full if self.fullscreen else mode_normal)

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        if event != cv2.EVENT_LBUTTONUP:
            return
        if self.zoom_index is not None:
            self.zoom_index = None
            return
        col = min(self.display.columns - 1, max(0, x // self.display.cell_width))
        row = min(self.display.rows - 1, max(0, y // self.display.cell_height))
        index = row * self.display.columns + col
        if index < len(self.sources):
            self.zoom_index = index


def scale_frame(frame: np.ndarray, cell_w: int, cell_h: int, mode: str) -> np.ndarray:
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    src_h, src_w = frame.shape[:2]
    if src_w == 0 or src_h == 0:
        return placeholder(cell_w, cell_h, "", "BAD FRAME")
    if mode == "stretch":
        return cv2.resize(frame, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
    if mode == "fill":
        scale = max(cell_w / src_w, cell_h / src_h)
        new_w, new_h = max(1, int(src_w * scale)), max(1, int(src_h * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        x = max(0, (new_w - cell_w) // 2)
        y = max(0, (new_h - cell_h) // 2)
        cropped = resized[y : y + cell_h, x : x + cell_w]
        if cropped.shape[0] != cell_h or cropped.shape[1] != cell_w:
            return cv2.resize(cropped, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        return cropped
    scale = min(cell_w / src_w, cell_h / src_h)
    new_w, new_h = max(1, int(src_w * scale)), max(1, int(src_h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    canvas[:] = (8, 8, 10)
    x = (cell_w - new_w) // 2
    y = (cell_h - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def placeholder(width: int, height: int, title: str, message: str) -> np.ndarray:
    tile = np.zeros((height, width, 3), dtype=np.uint8)
    tile[:] = (22, 22, 26)
    inset = 18
    shade_round_rect(
        tile,
        (inset, inset, width - inset, height - inset),
        color=(48, 48, 54),
        alpha=0.28,
        radius=10,
    )
    cx, cy = width // 2, height // 2
    draw_text(
        tile,
        title or "Camera",
        (cx, cy - 6),
        size=18,
        color=(210, 210, 210),
        align="center",
        valign="bottom",
    )
    draw_text(
        tile,
        message or "NO SIGNAL",
        (cx, cy + 8),
        size=14,
        color=(88, 88, 200),
        align="center",
        valign="top",
    )
    return tile


def draw_status_bar(
    tile: np.ndarray,
    name: str,
    snap: Snapshot,
    fps_text: str | None,
) -> None:
    h, w = tile.shape[:2]
    bar_h = 36
    shade_bottom_bar(tile, bar_h, alpha=0.70, fade=12)
    color = STATUS_COLOR.get(snap.status, (180, 180, 180))
    cy = h - 15
    draw_dot(tile, (16, cy), 5.5, color)
    draw_text(tile, name, (28, h - 8), size=15, valign="bottom")
    status = snap.status.upper()
    if snap.detail and snap.status not in ("live", "demo"):
        status = f"{status}  {snap.detail}"
    if fps_text:
        status = f"{status}   {fps_text}"
    draw_text(
        tile,
        status,
        (w - 12, h - 8),
        size=13,
        color=color,
        align="right",
        valign="bottom",
    )


def run_monitor(config: AppConfig) -> int:
    return MosaicApp(config).run()
