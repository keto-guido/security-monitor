"""OpenCV mosaic window: grid of camera tiles, zoom, overlays."""

from __future__ import annotations

import sys
import time

import cv2
import numpy as np

from security_monitor.config import AppConfig
from security_monitor.stream import Snapshot, build_sources

HELP_LINES = (
    "q / ESC  quit",
    "f        fullscreen",
    "g / 0    grid view",
    "1-9      zoom camera",
    "r        reconnect all",
    "click    zoom tile",
)

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
        print("Controls: q quit | f fullscreen | 1-9 zoom | g grid | r reconnect | h help | click tile")

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
                key = cv2.waitKey(delay) & 0xFF
                if key != 255:
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
            cell = self._render_cell(
                snap,
                self.sources[self.zoom_index].name,
                d.columns * d.cell_width,
                d.rows * d.cell_height,
            )
            return self._draw_help(cell)

        canvas = np.zeros((d.rows * d.cell_height, d.columns * d.cell_width, 3), dtype=np.uint8)
        canvas[:] = (12, 12, 14)
        for index in range(d.tile_count):
            row, col = divmod(index, d.columns)
            y, x = row * d.cell_height, col * d.cell_width
            if index < len(self.sources):
                snap = self.sources[index].snapshot()
                tile = self._render_cell(
                    snap, self.sources[index].name, d.cell_width, d.cell_height
                )
            else:
                tile = placeholder(d.cell_width, d.cell_height, "Empty", "No camera assigned")
            canvas[y : y + d.cell_height, x : x + d.cell_width] = tile
        self._draw_grid_lines(canvas)
        return self._draw_help(canvas)

    def _draw_grid_lines(self, canvas: np.ndarray) -> None:
        d = self.display
        color = (32, 32, 36)
        for col in range(1, d.columns):
            x = col * d.cell_width
            cv2.line(canvas, (x, 0), (x, canvas.shape[0]), color, 1)
        for row in range(1, d.rows):
            y = row * d.cell_height
            cv2.line(canvas, (0, y), (canvas.shape[1], y), color, 1)

    def _render_cell(self, snap: Snapshot, name: str, width: int, height: int) -> np.ndarray:
        if snap.frame is None:
            message = snap.detail or snap.status.replace("_", " ")
            tile = placeholder(width, height, name, message.upper() or "NO SIGNAL")
        else:
            tile = scale_frame(snap.frame, width, height, self.display.scale_mode)
        if self.display.show_labels:
            draw_status_bar(tile, name, snap, self.display.show_fps)
        color = STATUS_COLOR.get(snap.status, (80, 80, 80))
        cv2.rectangle(tile, (0, 0), (width - 1, height - 1), color, 2)
        return tile

    def _draw_help(self, canvas: np.ndarray) -> np.ndarray:
        if not self._show_help:
            return canvas
        overlay = canvas.copy()
        box_h = 28 + 22 * len(HELP_LINES)
        cv2.rectangle(overlay, (12, 12), (320, box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
        cv2.putText(
            canvas,
            f"Controls  (h to hide)   UI {self._ui_fps:.0f} fps",
            (24, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        for i, line in enumerate(HELP_LINES):
            cv2.putText(
                canvas,
                line,
                (24, 62 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
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
    cv2.rectangle(tile, (12, 12), (width - 13, height - 13), (50, 50, 58), 1)
    _center_text(tile, title or "Camera", height // 2 - 18, 0.7, (200, 200, 200))
    _center_text(tile, message or "NO SIGNAL", height // 2 + 18, 0.55, (90, 90, 210))
    return tile


def draw_status_bar(
    tile: np.ndarray,
    name: str,
    snap: Snapshot,
    show_fps: bool,
) -> None:
    h, w = tile.shape[:2]
    bar_h = 28
    overlay = tile.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, tile, 0.45, 0, tile)
    color = STATUS_COLOR.get(snap.status, (180, 180, 180))
    cv2.circle(tile, (14, h - bar_h // 2), 5, color, -1)
    label = name
    if show_fps:
        cam_fps = f"{snap.fps:.0f}" if snap.fps else "--"
        label = f"{name}   {cam_fps} fps"
    cv2.putText(
        tile,
        label,
        (26, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    status = snap.status.upper()
    if snap.detail and snap.status != "live":
        status = f"{status}  {snap.detail}"
    size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    cv2.putText(
        tile,
        status,
        (w - size[0] - 10, h - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def _center_text(tile: np.ndarray, text: str, y: int, scale: float, color: tuple[int, int, int]) -> None:
    width = tile.shape[1]
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0]
    x = max(8, (width - size[0]) // 2)
    cv2.putText(tile, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def run_monitor(config: AppConfig) -> int:
    return MosaicApp(config).run()
