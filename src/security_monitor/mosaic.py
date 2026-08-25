"""OpenCV mosaic window: grid of camera tiles, zoom, overlays."""

from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np


def _prefer_x11_on_wayland() -> None:
    """OpenCV window backends are more reliable on XWayland than native Wayland."""
    if not sys.platform.startswith("linux"):
        return
    if os.environ.get("WAYLAND_DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"

from security_monitor.config import AppConfig
from security_monitor.overlay import draw_dot, draw_text, shade_bottom_bar, shade_round_rect
from security_monitor.reboot import RebootJob, reboot_targets
from security_monitor.stream import Snapshot, build_sources

HELP_LINES = (
    "Esc      back / options",
    "q        quit",
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
KEY_LEFT = frozenset({2424832, 65361, 16777234})
KEY_UP = frozenset({2490368, 65362, 16777235})
KEY_RIGHT = frozenset({2555904, 65363, 16777236})
KEY_DOWN = frozenset({2621440, 65364, 16777237})
KEY_HOME = frozenset({2359296, 65360, 16777232, 0xFF50})
KEY_ENTER = frozenset({13, 10, 16777220, 16777221})
KEY_ESC = frozenset({27, 16777216})
KEY_PLUS = frozenset({ord("+"), ord("="), 7012352, 65451})
KEY_MINUS = frozenset({ord("-"), ord("_"), 7143424, 65453})

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
        self._menu_open = False
        self._menu_index = 0
        self._menu_page = "root"
        self._menu_hitboxes: list[tuple[str, int, int, int, int]] = []
        self._reboot_job: RebootJob | None = None
        self._reboot_notice = ""
        self._view_w, self._view_h = self.display.canvas_size
        self._cell_w = self.display.cell_width
        self._cell_h = self.display.cell_height

    def run(self) -> int:
        if not self.sources:
            print("No enabled cameras in the current grid. Edit config.yaml.", file=sys.stderr)
            return 1
        _prefer_x11_on_wayland()
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
            "Controls: Esc back/options | q quit | f fullscreen | 1-9 focus | "
            "wheel/+/- zoom | arrows pan | h help"
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
                wait = getattr(cv2, "waitKeyEx", cv2.waitKey)
                key = wait(delay)
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

    def _window_size(self) -> tuple[int, int]:
        try:
            _x, _y, ww, wh = cv2.getWindowImageRect(self.window)
            if ww >= 320 and wh >= 180:
                return int(ww), int(wh)
        except cv2.error:
            pass
        return self.display.canvas_size

    def _sync_layout(self) -> tuple[int, int, int, int]:
        """Fit the mosaic to the live window so HUD text is not upscaled."""
        ww, wh = self._window_size()
        cols, rows = self.display.columns, self.display.rows
        cell_w = max(160, ww // cols)
        cell_h = max(90, wh // rows)
        self._cell_w, self._cell_h = cell_w, cell_h
        self._view_w, self._view_h = cell_w * cols, cell_h * rows
        return cell_w, cell_h, self._view_w, self._view_h

    def _compose(self) -> np.ndarray:
        d = self.display
        cell_w, cell_h, width, height = self._sync_layout()
        if self.zoom_index is not None and 0 <= self.zoom_index < len(self.sources):
            snap = self.sources[self.zoom_index].snapshot()
            name = self.sources[self.zoom_index].name
            cell = self._render_cell(snap, name, width, height, overlay=False)
            cell = magnify(cell, self.view_zoom, self.pan_x, self.pan_y)
            self._draw_cell_overlay(cell, snap, name)
            self._draw_zoom_badge(cell)
            return self._draw_menu(self._draw_reboot(self._draw_help(cell)))

        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (12, 12, 14)
        zoomed = self.view_zoom > 1.001
        for index in range(d.tile_count):
            row, col = divmod(index, d.columns)
            y, x = row * cell_h, col * cell_w
            if index < len(self.sources):
                snap = self.sources[index].snapshot()
                tile = self._render_cell(
                    snap,
                    self.sources[index].name,
                    cell_w,
                    cell_h,
                    overlay=not zoomed,
                )
            else:
                tile = placeholder(cell_w, cell_h, "Empty", "No camera assigned")
            canvas[y : y + cell_h, x : x + cell_w] = tile
        if not zoomed:
            self._draw_grid_lines(canvas)
        canvas = magnify(canvas, self.view_zoom, self.pan_x, self.pan_y)
        self._draw_zoom_badge(canvas)
        return self._draw_menu(self._draw_reboot(self._draw_help(canvas)))

    def _draw_grid_lines(self, canvas: np.ndarray) -> None:
        cols, rows = self.display.columns, self.display.rows
        cell_w, cell_h = self._cell_w, self._cell_h
        tint = np.array([18, 18, 20], dtype=np.float32)
        alpha = 0.45
        for col in range(1, cols):
            x = col * cell_w
            col_pixels = canvas[:, x].astype(np.float32)
            canvas[:, x] = (col_pixels * (1.0 - alpha) + tint * alpha).astype(np.uint8)
        for row in range(1, rows):
            y = row * cell_h
            row_pixels = canvas[y, :].astype(np.float32)
            canvas[y, :] = (row_pixels * (1.0 - alpha) + tint * alpha).astype(np.uint8)

    def _render_cell(
        self,
        snap: Snapshot,
        name: str,
        width: int,
        height: int,
        *,
        overlay: bool = True,
    ) -> np.ndarray:
        if snap.frame is None:
            message = snap.detail or snap.status.replace("_", " ")
            tile = placeholder(width, height, name, message.upper() or "NO SIGNAL")
        else:
            mode = self.display.scale_mode if self.view_zoom <= 1.001 else "fill"
            tile = scale_frame(snap.frame, width, height, mode)
        if overlay:
            self._draw_cell_overlay(tile, snap, name)
        return tile

    def _draw_cell_overlay(self, tile: np.ndarray, snap: Snapshot, name: str) -> None:
        if not self.display.show_labels:
            return
        fps_text = None
        if self.display.show_fps:
            shown = self._stable_fps(name, snap.fps)
            fps_text = f"{shown:2d} fps" if shown else "-- fps"
        draw_status_bar(tile, name, snap, fps_text)

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
        shade_round_rect(canvas, (16, 16, 360, 16 + box_h), alpha=0.82, radius=12)
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
        ch = key if 0 <= key < 256 else key & 0xFF
        if self._reboot_job is not None:
            self._handle_reboot_key(key, ch)
            return
        if key in KEY_ESC or ch == 27:
            self._on_escape()
            return
        if self._menu_open:
            self._handle_menu_key(key, ch)
            return

        if key in KEY_LEFT:
            self._pan(-1, 0)
            return
        if key in KEY_RIGHT:
            self._pan(1, 0)
            return
        if key in KEY_UP:
            self._pan(0, -1)
            return
        if key in KEY_DOWN:
            self._pan(0, 1)
            return
        if key in KEY_HOME:
            self._reset_view()
            return

        if key in KEY_PLUS or ch in KEY_PLUS:
            self._nudge_zoom(1)
            return
        if key in KEY_MINUS or ch in KEY_MINUS:
            self._nudge_zoom(-1)
            return
        if key >= 256 and ch == 0:
            return

        if ch in (ord("q"), ord("Q")):
            self._running = False
        elif ch in (ord("f"), ord("F")):
            self.fullscreen = not self.fullscreen
            self._apply_fullscreen()
        elif ch in (ord("g"), ord("G"), ord("0")):
            self._go_main_layout()
        elif ch in (ord("r"), ord("R")):
            self._reconnect_all()
        elif ch in (ord("h"), ord("H"), ord("?")):
            self._show_help = not self._show_help
        elif ord("1") <= ch <= ord("9"):
            index = ch - ord("1")
            if index < len(self.sources):
                self._focus_camera(index)

    def _handle_menu_key(self, key: int, ch: int) -> None:
        items = self._menu_items()
        if key in KEY_UP:
            self._menu_index = (self._menu_index - 1) % len(items)
            return
        if key in KEY_DOWN:
            self._menu_index = (self._menu_index + 1) % len(items)
            return
        if key in KEY_ENTER or ch in (13, 10):
            self._activate_menu(items[self._menu_index][0])
            return
        if ch in (ord("q"), ord("Q")):
            if self._menu_page == "reboot_confirm":
                self._menu_page = "root"
                self._menu_index = 0
                return
            self._activate_menu("exit")
            return
        if ord("1") <= ch <= ord("9"):
            index = ch - ord("1")
            if index < len(items):
                self._menu_index = index
                self._activate_menu(items[index][0])

    def _on_escape(self) -> None:
        if self._menu_open and self._menu_page == "reboot_confirm":
            self._menu_page = "root"
            self._menu_index = 0
            return
        action = escape_action(menu_open=self._menu_open, on_main_layout=self._is_main_layout())
        if action == "close_menu":
            self._menu_open = False
            self._menu_page = "root"
        elif action == "main_layout":
            self._go_main_layout()
        else:
            self._menu_open = True
            self._menu_index = 0
            self._menu_page = "root"
            self._show_help = False
            self._reboot_notice = ""

    def _is_main_layout(self) -> bool:
        return self.zoom_index is None and self.view_zoom <= 1.001

    def _go_main_layout(self) -> None:
        self.zoom_index = None
        self._reset_view()
        self._menu_open = False
        self._menu_page = "root"

    def _menu_items(self) -> list[tuple[str, str]]:
        if self._menu_page == "reboot_confirm":
            return [
                ("reboot_run", "Yes, reboot all cameras"),
                ("reboot_cancel", "Cancel"),
            ]
        fullscreen = "Windowed mode" if self.fullscreen else "Fullscreen"
        return [
            ("resume", "Resume"),
            ("fullscreen", fullscreen),
            ("reconnect", "Reconnect streams"),
            ("reboot", "Reboot cameras"),
            ("exit", "Exit"),
        ]

    def _activate_menu(self, action: str) -> None:
        if action == "resume":
            self._menu_open = False
            self._menu_page = "root"
        elif action == "fullscreen":
            self.fullscreen = not self.fullscreen
            self._apply_fullscreen()
        elif action == "reconnect":
            self._reconnect_all()
            self._menu_open = False
            self._menu_page = "root"
        elif action == "reboot":
            if not reboot_targets(self.config.cameras):
                self._reboot_notice = "No rebootable cameras. Set type: ubiquiti or reolink."
                return
            self._reboot_notice = ""
            self._menu_page = "reboot_confirm"
            self._menu_index = 1
        elif action == "reboot_cancel":
            self._menu_page = "root"
            self._menu_index = 0
        elif action == "reboot_run":
            self._start_reboot()
        elif action == "exit":
            self._running = False

    def _start_reboot(self) -> None:
        devices = reboot_targets(self.config.cameras)
        if not devices:
            self._reboot_notice = "No rebootable cameras. Set type: ubiquiti or reolink."
            self._menu_page = "root"
            return
        print(f"Rebooting {len(devices)} camera(s)…")
        self._menu_open = False
        self._menu_page = "root"
        self._reboot_job = RebootJob(devices)
        self._reboot_job.start()

    def _handle_reboot_key(self, key: int, ch: int) -> None:
        if ch in (ord("q"), ord("Q")):
            self._running = False
            return
        if self._reboot_job is None:
            return
        if not self._reboot_job.finished:
            return
        if key in KEY_ESC or ch == 27 or key in KEY_ENTER or ch in (13, 10):
            self._finish_reboot()

    def _finish_reboot(self) -> None:
        if self._reboot_job is not None:
            for line in self._reboot_job.summaries():
                print(line)
        self._reboot_job = None
        self._reconnect_all()

    def _draw_reboot(self, canvas: np.ndarray) -> np.ndarray:
        job = self._reboot_job
        if job is None:
            return canvas
        canvas[:] = (canvas.astype(np.float32) * 0.38).astype(np.uint8)
        rows = job.rows()
        card_w = min(canvas.shape[1] - 24, 920)
        row_h = 36
        title_h = 52
        footer_h = 40
        card_h = title_h + 8 + row_h * max(1, len(rows)) + footer_h
        h, w = canvas.shape[:2]
        x0 = max(12, (w - card_w) // 2)
        y0 = max(12, (h - card_h) // 2)
        shade_round_rect(
            canvas,
            (x0, y0, x0 + card_w, y0 + card_h),
            color=(18, 18, 22),
            alpha=0.92,
            radius=16,
        )
        title = "Reboot complete" if job.finished else "Rebooting cameras"
        draw_text(
            canvas,
            title,
            (x0 + card_w // 2, y0 + 14),
            size=20,
            align="center",
            valign="top",
        )
        colors = {
            "white": (220, 220, 220),
            "cyan": (210, 190, 70),
            "yellow": (40, 200, 220),
            "red": (60, 60, 220),
            "green": (70, 190, 90),
        }
        for i, row in enumerate(rows):
            ry = y0 + title_h + i * row_h
            progress = ""
            if row.total > 0:
                pct = int(min(max(row.elapsed / row.total, 0), 1) * 100)
                progress = f"{int(row.elapsed):>3}/{int(row.total)}s {pct}%"
            status = row.status
            if row.detail:
                status = f"{status}  {row.detail}"
            color = colors.get(row.color, (220, 220, 220))
            draw_text(canvas, row.name, (x0 + 24, ry), size=15, valign="top")
            draw_text(
                canvas,
                row.kind,
                (x0 + 200, ry),
                size=14,
                color=(170, 170, 170),
                valign="top",
            )
            draw_text(
                canvas,
                row.phase,
                (x0 + 300, ry),
                size=14,
                color=(190, 190, 190),
                valign="top",
            )
            draw_text(
                canvas,
                progress,
                (x0 + 460, ry),
                size=14,
                color=(180, 180, 180),
                valign="top",
            )
            draw_text(
                canvas,
                status,
                (x0 + card_w - 24, ry),
                size=14,
                color=color,
                align="right",
                valign="top",
            )
        footer = "Enter or Esc to resume" if job.finished else "Waiting for cameras to return…"
        draw_text(
            canvas,
            footer,
            (x0 + card_w // 2, y0 + card_h - 28),
            size=13,
            color=(150, 150, 150),
            align="center",
            valign="top",
        )
        return canvas

    def _reconnect_all(self) -> None:
        print("Reconnecting all cameras...")
        for source in self.sources:
            source.reconnect()

    def _draw_menu(self, canvas: np.ndarray) -> np.ndarray:
        self._menu_hitboxes = []
        if not self._menu_open or self._reboot_job is not None:
            return canvas
        canvas[:] = (canvas.astype(np.float32) * 0.38).astype(np.uint8)
        items = self._menu_items()
        card_w, row_h, pad = 480, 46, 20
        title_h = 56
        footer_h = 36
        notice_h = 28 if self._reboot_notice else 0
        card_h = title_h + notice_h + row_h * len(items) + footer_h
        h, w = canvas.shape[:2]
        x0 = max(12, (w - card_w) // 2)
        y0 = max(12, (h - card_h) // 2)
        shade_round_rect(
            canvas,
            (x0, y0, x0 + card_w, y0 + card_h),
            color=(18, 18, 22),
            alpha=0.92,
            radius=16,
        )
        heading = "Confirm reboot" if self._menu_page == "reboot_confirm" else "Options"
        draw_text(
            canvas,
            heading,
            (x0 + card_w // 2, y0 + 16),
            size=22,
            align="center",
            valign="top",
        )
        if self._reboot_notice:
            draw_text(
                canvas,
                self._reboot_notice,
                (x0 + card_w // 2, y0 + 44),
                size=13,
                color=(70, 90, 220),
                align="center",
                valign="top",
            )
        for i, (action, label) in enumerate(items):
            ry = y0 + title_h + notice_h + i * row_h
            box = (x0 + pad, ry, x0 + card_w - pad, ry + row_h - 8)
            if i == self._menu_index:
                shade_round_rect(canvas, box, color=(56, 120, 78), alpha=0.7, radius=8)
            draw_text(
                canvas,
                f"{i + 1}",
                (box[0] + 14, ry + 8),
                size=15,
                color=(180, 180, 180),
                valign="top",
            )
            draw_text(
                canvas,
                label,
                (box[0] + 44, ry + 8),
                size=17,
                color=(236, 236, 236),
                valign="top",
            )
            self._menu_hitboxes.append((action, *box))
        footer = "This will restart every camera" if self._menu_page == "reboot_confirm" else "Enter to select    Esc to close"
        draw_text(
            canvas,
            footer,
            (x0 + card_w // 2, y0 + card_h - 28),
            size=13,
            color=(150, 150, 150),
            align="center",
            valign="top",
        )
        return canvas

    def _focus_camera(self, index: int | None) -> None:
        self.zoom_index = index
        self._reset_view()
        self._menu_open = False
        self._menu_page = "root"

    def _reset_view(self) -> None:
        self.view_zoom = 1.0
        self.pan_x = 0.5
        self.pan_y = 0.5

    def _nudge_zoom(self, direction: int, focus: tuple[float, float] | None = None) -> None:
        focus_x, focus_y = focus if focus is not None else (0.5, 0.5)
        new_zoom = self.view_zoom * (ZOOM_FACTOR ** direction)
        new_zoom = min(ZOOM_MAX, max(ZOOM_MIN, new_zoom))
        if direction < 0 and new_zoom <= 1.02:
            new_zoom = 1.0
        self.pan_x, self.pan_y = zoom_toward(
            self.view_zoom, self.pan_x, self.pan_y, new_zoom, focus_x, focus_y
        )
        self.view_zoom = new_zoom
        if self.view_zoom <= 1.001:
            self.pan_x = 0.5
            self.pan_y = 0.5

    def _pan(self, dx: int, dy: int) -> None:
        if self.view_zoom <= 1.001:
            return
        step = (1.0 / self.view_zoom) * PAN_VIEW_FRACTION
        self.pan_x, self.pan_y = clamp_center(
            self.view_zoom,
            self.pan_x + dx * step,
            self.pan_y + dy * step,
        )

    def _draw_zoom_badge(self, canvas: np.ndarray) -> None:
        if self.view_zoom <= 1.001:
            return
        label = f"{self.view_zoom:.1f}×"
        draw_text(
            canvas,
            label,
            (canvas.shape[1] - 16, 14),
            size=16,
            color=(230, 230, 230),
            align="right",
            valign="top",
        )

    def _apply_fullscreen(self) -> None:
        prop = getattr(cv2, "WND_PROP_FULLSCREEN", None)
        mode_full = getattr(cv2, "WINDOW_FULLSCREEN", 1)
        mode_normal = getattr(cv2, "WINDOW_NORMAL", 0)
        if prop is None:
            return
        cv2.setWindowProperty(self.window, prop, mode_full if self.fullscreen else mode_normal)

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _userdata: object) -> None:
        if self._reboot_job is not None:
            if event == cv2.EVENT_LBUTTONUP and self._reboot_job.finished:
                self._finish_reboot()
            return
        if self._menu_open:
            self._on_menu_mouse(event, x, y, flags)
            return
        wheel = getattr(cv2, "EVENT_MOUSEWHEEL", None)
        if wheel is not None and event == wheel:
            steps = _wheel_steps(flags)
            if steps:
                self._nudge_zoom(steps, focus=self._event_norm(x, y))
            return
        if event != cv2.EVENT_LBUTTONUP:
            return
        if self.view_zoom > 1.001:
            self._reset_view()
            return
        if self.zoom_index is not None:
            self._go_main_layout()
            return
        nx, ny = self._event_norm(x, y)
        col = min(self.display.columns - 1, max(0, int(nx * self.display.columns)))
        row = min(self.display.rows - 1, max(0, int(ny * self.display.rows)))
        index = row * self.display.columns + col
        if index < len(self.sources):
            self._focus_camera(index)

    def _on_menu_mouse(self, event: int, x: int, y: int, flags: int) -> None:
        wheel = getattr(cv2, "EVENT_MOUSEWHEEL", None)
        if wheel is not None and event == wheel:
            items = self._menu_items()
            if not items:
                return
            step = _wheel_steps(flags)
            if step:
                self._menu_index = (self._menu_index - step) % len(items)
            return
        if event != cv2.EVENT_LBUTTONUP:
            return
        px, py = self._event_to_pixel(x, y)
        for i, (action, x0, y0, x1, y1) in enumerate(self._menu_hitboxes):
            if x0 <= px <= x1 and y0 <= py <= y1:
                self._menu_index = i
                self._activate_menu(action)
                return
        self._menu_open = False
        self._menu_page = "root"

    def _event_to_pixel(self, x: int, y: int) -> tuple[int, int]:
        nx, ny = self._event_norm(x, y)
        return int(nx * self._view_w), int(ny * self._view_h)

    def _event_norm(self, x: int, y: int) -> tuple[float, float]:
        ww, wh = self.display.canvas_size
        try:
            _ox, _oy, rw, rh = cv2.getWindowImageRect(self.window)
            if rw > 1 and rh > 1:
                ww, wh = rw, rh
        except cv2.error:
            pass
        return (
            min(max(x / max(ww, 1), 0.0), 1.0),
            min(max(y / max(wh, 1), 0.0), 1.0),
        )


def scale_frame(frame: np.ndarray, cell_w: int, cell_h: int, mode: str) -> np.ndarray:
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    src_h, src_w = frame.shape[:2]
    if src_w == 0 or src_h == 0:
        return placeholder(cell_w, cell_h, "", "BAD FRAME")

    def _resize(src: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        sw, sh = src.shape[1], src.shape[0]
        shrinking = size[0] < sw or size[1] < sh
        interp = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
        return cv2.resize(src, size, interpolation=interp)

    if mode == "stretch":
        return _resize(frame, (cell_w, cell_h))
    if mode == "fill":
        scale = max(cell_w / src_w, cell_h / src_h)
        new_w, new_h = max(1, int(src_w * scale)), max(1, int(src_h * scale))
        resized = _resize(frame, (new_w, new_h))
        x = max(0, (new_w - cell_w) // 2)
        y = max(0, (new_h - cell_h) // 2)
        cropped = resized[y : y + cell_h, x : x + cell_w]
        if cropped.shape[0] != cell_h or cropped.shape[1] != cell_w:
            return _resize(cropped, (cell_w, cell_h))
        return cropped
    scale = min(cell_w / src_w, cell_h / src_h)
    new_w, new_h = max(1, int(src_w * scale)), max(1, int(src_h * scale))
    resized = _resize(frame, (new_w, new_h))
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
    bar_h = max(34, int(h * 0.08))
    fade = max(10, bar_h // 3)
    shade_bottom_bar(tile, bar_h, alpha=0.70, fade=fade)
    color = STATUS_COLOR.get(snap.status, (180, 180, 180))
    cy = h - bar_h // 2
    name_size = max(14, min(22, int(h * 0.038)))
    meta_size = max(12, min(18, int(h * 0.032)))
    draw_dot(tile, (16 + name_size // 8, cy), max(5.0, name_size * 0.32), color)
    draw_text(tile, name, (28 + name_size // 4, h - 8), size=name_size, valign="bottom")
    status = snap.status.upper()
    if snap.detail and snap.status not in ("live", "demo"):
        status = f"{status}  {snap.detail}"
    if fps_text:
        status = f"{status}   {fps_text}"
    draw_text(
        tile,
        status,
        (w - 12, h - 8),
        size=meta_size,
        color=color,
        align="right",
        valign="bottom",
    )


def run_monitor(config: AppConfig) -> int:
    return MosaicApp(config).run()


def escape_action(*, menu_open: bool, on_main_layout: bool) -> str:
    """Where Escape should go: close_menu, main_layout, or open_menu."""
    if menu_open:
        return "close_menu"
    if not on_main_layout:
        return "main_layout"
    return "open_menu"


def clamp_center(zoom: float, cx: float, cy: float) -> tuple[float, float]:
    if zoom <= 1.001:
        return 0.5, 0.5
    half = 0.5 / zoom
    return (
        min(max(cx, half), 1.0 - half),
        min(max(cy, half), 1.0 - half),
    )


def zoom_toward(
    zoom: float,
    cx: float,
    cy: float,
    new_zoom: float,
    focus_x: float,
    focus_y: float,
) -> tuple[float, float]:
    """Keep the image point under (focus_x, focus_y) stable while changing zoom."""
    vis = 1.0 / max(zoom, 1.0)
    src_x = cx - vis / 2.0 + focus_x * vis
    src_y = cy - vis / 2.0 + focus_y * vis
    new_vis = 1.0 / max(new_zoom, 1.0)
    new_cx = src_x - (focus_x - 0.5) * new_vis
    new_cy = src_y - (focus_y - 0.5) * new_vis
    return clamp_center(new_zoom, new_cx, new_cy)


def magnify(image: np.ndarray, zoom: float, cx: float, cy: float) -> np.ndarray:
    if zoom <= 1.001:
        return image
    h, w = image.shape[:2]
    vis_w = max(1, min(w, int(round(w / zoom))))
    vis_h = max(1, min(h, int(round(h / zoom))))
    cx, cy = clamp_center(zoom, cx, cy)
    x0 = int(round(cx * w - vis_w / 2))
    y0 = int(round(cy * h - vis_h / 2))
    x0 = max(0, min(x0, w - vis_w))
    y0 = max(0, min(y0, h - vis_h))
    crop = image[y0 : y0 + vis_h, x0 : x0 + vis_w]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def _wheel_steps(flags: int) -> int:
    getter = getattr(cv2, "getMouseWheelDelta", None)
    if getter is not None:
        delta = int(getter(flags))
    else:
        delta = (flags >> 16) & 0xFFFF
        if delta >= 0x8000:
            delta -= 0x10000
    if delta == 0:
        return 0
    return 1 if delta > 0 else -1
