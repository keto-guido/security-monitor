"""OpenCV mosaic window: grid of camera tiles, zoom, overlays."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def _prefer_x11_on_wayland() -> None:
    """OpenCV window backends are more reliable on XWayland than native Wayland."""
    if not sys.platform.startswith("linux"):
        return
    if os.environ.get("WAYLAND_DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"


def _configure_linux_gui() -> None:
    """
    Stabilize OpenCV's Qt HighGUI on Linux.

    Without this, Qt may apply HiDPI scaling and return unstable
    getWindowImageRect() sizes — the mosaic then paints at the wrong aspect
    and the toolkit stretches it (classic "squished" look).
    """
    if not sys.platform.startswith("linux"):
        return
    _prefer_x11_on_wayland()
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "0")
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")
    os.environ.setdefault("QT_SCALE_FACTOR", "1")
    _ensure_opencv_qt_fonts()


def _ensure_opencv_qt_fonts() -> None:
    """OpenCV's Qt build looks for fonts under cv2/qt/fonts; link system DejaVu if missing."""
    try:
        import cv2
    except ImportError:
        return
    font_dir = Path(cv2.__file__).resolve().parent / "qt" / "fonts"
    marker = font_dir / "DejaVuSans.ttf"
    if marker.exists() or marker.is_symlink():
        return
    for source in (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
    ):
        if not source.is_file():
            continue
        try:
            font_dir.mkdir(parents=True, exist_ok=True)
            if not marker.exists():
                marker.symlink_to(source)
        except OSError:
            return
        break

from security_monitor.buffer import (
    REWIND_BUFFER_CHOICES,
    SMOOTH_BUFFER_CHOICES,
    next_choice,
)
from security_monitor.capture import (
    CLIP_LENGTH_CHOICES,
    CaptureError,
    LiveClipJob,
    default_save_directory,
    resolve_save_directory,
    save_snapshot,
    write_clip,
)
from security_monitor.config import (
    AppConfig,
    CYCLE_FOCUS_CHOICES,
    CameraConfig,
    ensure_layout_fits,
    move_camera,
    next_layout_preset,
    save_display_settings,
    unique_camera_name,
)
from security_monitor.detection import DetectionEngine, draw_boxes
from security_monitor.overlay import draw_dot, draw_text, shade_bottom_bar, shade_round_rect
from security_monitor.reboot import RebootJob, reboot_targets
from security_monitor.stream import Snapshot, build_sources

HELP_LINES = (
    "Esc      back / options",
    "q        quit",
    "f        fullscreen",
    "g / 0    grid view",
    "1-9      focus camera",
    "n / p    next / prev camera",
    "wheel    zoom in/out",
    "+ / -    zoom in/out",
    "arrows   pan / strafe",
    ", / .    rewind back / forward",
    "l        jump to live",
    "s        save snapshot",
    "c        save clip",
    "Home     reset zoom",
    "r        reconnect all",
    "click    focus tile",
)

ZOOM_MIN = 1.0
ZOOM_MAX = 12.0
ZOOM_FACTOR = 1.2
PAN_VIEW_FRACTION = 0.18
REWIND_STEP_SECONDS = 1.0
REWIND_STEP_COARSE = 5.0

_CAMERA_MENU_PAGES = frozenset(
    {
        "cameras",
        "cameras_arrange",
        "cameras_toggle",
        "cameras_add",
        "cameras_remove",
    }
)
_NESTED_MENU_PAGES = frozenset(
    {
        "reboot_confirm",
        "video",
        "capture",
        "detection",
        "detection_cams",
        *_CAMERA_MENU_PAGES,
    }
)


@dataclass
class TextPrompt:
    """Simple on-screen text entry (OpenCV has no native input box)."""

    title: str
    value: str = ""
    kind: str = "add_name"  # add_name | add_url
    pending_name: str = ""


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
        self._camera_by_name = {cam.name: cam for cam in self.cameras}
        self.sources = build_sources(self.cameras, self.display)
        self._detection = DetectionEngine()
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
        self._clip_job: LiveClipJob | None = None
        self._capture_flash = ""
        self._capture_flash_until = 0.0
        self._view_w, self._view_h = self.display.canvas_size
        self._cell_w = self.display.cell_width
        self._cell_h = self.display.cell_height
        self._grid_x = 0
        self._grid_y = 0
        self._prompt: TextPrompt | None = None
        self._cycle_deadline = 0.0

    def run(self) -> int:
        if not self.sources:
            print("No enabled cameras in the current grid. Edit config.yaml.", file=sys.stderr)
            return 1
        _configure_linux_gui()
        self._apply_screen_rotate()
        for source in self.sources:
            source.start()
            print(f"Started {source.name}")

        try:
            flags = cv2.WINDOW_NORMAL
            keep = getattr(cv2, "WINDOW_KEEPRATIO", 0)
            flags |= keep
            cv2.namedWindow(self.window, flags)
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
        try:
            aspect = getattr(cv2, "WND_PROP_ASPECT_RATIO", None)
            keep = getattr(cv2, "WINDOW_KEEPRATIO", None)
            if aspect is not None and keep is not None:
                cv2.setWindowProperty(self.window, aspect, keep)
        except cv2.error:
            pass
        cv2.setMouseCallback(self.window, self._on_mouse)
        self._apply_fullscreen()
        print(
            "Controls: Esc back/options | q quit | f fullscreen | 1-9 focus | "
            "n/p next/prev | wheel/+/- zoom | arrows pan | h help"
        )

        delay = max(1, int(1000 / self.display.fps))
        stamps: list[float] = []
        try:
            while self._running:
                self._tick_cycle_focus()
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

    def _apply_screen_rotate(self) -> None:
        """Optional display.screen_rotate (Linux xrandr) from config."""
        from security_monitor.display_setup import maybe_apply_config_rotation

        message = maybe_apply_config_rotation(
            self.display.screen_rotate,
            output=self.display.screen_output,
        )
        if message:
            print(message)

    def _shutdown(self) -> None:
        for source in self.sources:
            source.stop()
        cv2.destroyAllWindows()

    def _screen_size(self) -> tuple[int, int] | None:
        from security_monitor.display_setup import screen_size

        return screen_size(self.display.screen_output)

    def _reported_window_size(self) -> tuple[int, int] | None:
        try:
            _x, _y, ww, wh = cv2.getWindowImageRect(self.window)
        except cv2.error:
            return None
        if ww < 1 or wh < 1:
            return None
        return int(ww), int(wh)

    def _window_size(self) -> tuple[int, int]:
        from security_monitor.display_setup import sanitize_window_size

        return sanitize_window_size(
            self._reported_window_size(),
            fullscreen=self.fullscreen,
            screen=self._screen_size(),
            fallback=self.display.canvas_size,
        )

    def _sync_layout(self) -> tuple[int, int, int, int]:
        """Fit the mosaic to a trustworthy window size so HUD text is not upscaled."""
        ww, wh = self._window_size()
        cols, rows = self.display.columns, self.display.rows
        cell_w = max(160, ww // cols)
        cell_h = max(90, wh // rows)
        self._cell_w, self._cell_h = cell_w, cell_h
        # Paint at the exact window size so HighGUI does not stretch a shorter canvas.
        self._view_w, self._view_h = ww, wh
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
            self._draw_buffer_badge(cell, snap)
            self._feed_clip_job(cell)
            self._draw_capture_hud(cell)
            return self._draw_prompt(self._draw_menu(self._draw_reboot(self._draw_help(cell))))

        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (12, 12, 14)
        grid_w = cell_w * d.columns
        grid_h = cell_h * d.rows
        x_off = max(0, (width - grid_w) // 2)
        y_off = max(0, (height - grid_h) // 2)
        self._grid_x, self._grid_y = x_off, y_off
        zoomed = self.view_zoom > 1.001
        any_rewind = False
        for index in range(d.tile_count):
            row, col = divmod(index, d.columns)
            y, x = y_off + row * cell_h, x_off + col * cell_w
            if index < len(self.sources):
                snap = self.sources[index].snapshot()
                any_rewind = any_rewind or snap.rewinding
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
            self._draw_grid_lines(canvas, x_off=x_off, y_off=y_off)
        canvas = magnify(canvas, self.view_zoom, self.pan_x, self.pan_y)
        self._draw_zoom_badge(canvas)
        if any_rewind or self.display.smooth_buffer or self.display.rewind_buffer:
            # Aggregate badge from first live source when in grid view.
            sample = self.sources[0].snapshot() if self.sources else None
            if sample is not None:
                self._draw_buffer_badge(canvas, sample)
        self._feed_clip_job(canvas)
        self._draw_capture_hud(canvas)
        return self._draw_prompt(self._draw_menu(self._draw_reboot(self._draw_help(canvas))))

    def _draw_grid_lines(
        self,
        canvas: np.ndarray,
        *,
        x_off: int = 0,
        y_off: int = 0,
    ) -> None:
        cols, rows = self.display.columns, self.display.rows
        cell_w, cell_h = self._cell_w, self._cell_h
        tint = np.array([18, 18, 20], dtype=np.float32)
        alpha = 0.45
        for col in range(1, cols):
            x = x_off + col * cell_w
            if 0 <= x < canvas.shape[1]:
                col_pixels = canvas[:, x].astype(np.float32)
                canvas[:, x] = (col_pixels * (1.0 - alpha) + tint * alpha).astype(np.uint8)
        for row in range(1, rows):
            y = y_off + row * cell_h
            if 0 <= y < canvas.shape[0]:
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
            frame = snap.frame
            cam = self._camera_by_name.get(name)
            if cam is not None and (
                (self.display.people_detection and cam.detect_people)
                or (self.display.object_detection and cam.detect_objects)
            ):
                boxes = self._detection.process(
                    name,
                    frame,
                    detect_people=bool(self.display.people_detection and cam.detect_people),
                    detect_objects=bool(self.display.object_detection and cam.detect_objects),
                )
                if boxes:
                    frame = frame.copy()
                    draw_boxes(frame, boxes)
            mode = self.display.scale_mode if self.view_zoom <= 1.001 else "fill"
            tile = scale_frame(frame, width, height, mode)
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
        if self._prompt is not None:
            self._handle_prompt_key(key, ch)
            return
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
            if self.view_zoom > 1.001:
                self._pan(-1, 0)
            elif self.display.rewind_buffer:
                self._nudge_rewind(REWIND_STEP_SECONDS)
            return
        if key in KEY_RIGHT:
            if self.view_zoom > 1.001:
                self._pan(1, 0)
            elif self.display.rewind_buffer:
                self._nudge_rewind(-REWIND_STEP_SECONDS)
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
        elif ch in (ord("n"), ord("N")):
            self._cycle_focus(1)
        elif ch in (ord("p"), ord("P")):
            self._cycle_focus(-1)
        elif ch in (ord("r"), ord("R")):
            self._reconnect_all()
        elif ch in (ord("h"), ord("H"), ord("?")):
            self._show_help = not self._show_help
        elif ch in (ord(","), ord("<")):
            # Shift-, comes through as < on some layouts.
            self._nudge_rewind(REWIND_STEP_SECONDS if ch == ord(",") else REWIND_STEP_COARSE)
        elif ch in (ord("."), ord(">")):
            self._nudge_rewind(-(REWIND_STEP_SECONDS if ch == ord(".") else REWIND_STEP_COARSE))
        elif ch in (ord("l"), ord("L")):
            self._go_live()
        elif ch in (ord("s"), ord("S")):
            self._save_snapshot()
        elif ch in (ord("c"), ord("C")):
            self._save_clip()
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
        if key in KEY_LEFT:
            self._adjust_menu_item(items[self._menu_index][0], -1)
            return
        if key in KEY_RIGHT:
            self._adjust_menu_item(items[self._menu_index][0], 1)
            return
        if key in KEY_ENTER or ch in (13, 10):
            self._activate_menu(items[self._menu_index][0])
            return
        if ch in (ord("q"), ord("Q")):
            if self._menu_page == "detection_cams":
                self._menu_page = "detection"
                self._menu_index = 0
                return
            if self._menu_page in _CAMERA_MENU_PAGES and self._menu_page != "cameras":
                self._menu_page = "cameras"
                self._menu_index = 0
                return
            if self._menu_page in _NESTED_MENU_PAGES:
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
        if self._prompt is not None:
            self._prompt = None
            return
        if self._menu_open and self._menu_page == "detection_cams":
            self._menu_page = "detection"
            self._menu_index = 0
            return
        if self._menu_open and self._menu_page in _CAMERA_MENU_PAGES and self._menu_page != "cameras":
            self._menu_page = "cameras"
            self._menu_index = 0
            return
        if self._menu_open and self._menu_page in {
            "reboot_confirm",
            "video",
            "capture",
            "detection",
            "cameras",
        }:
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
        if self._menu_page == "video":
            d = self.display
            smooth = "On" if d.smooth_buffer else "Off"
            rewind = "On" if d.rewind_buffer else "Off"
            return [
                ("smooth_toggle", f"Smooth buffer: {smooth}"),
                ("smooth_length", f"Buffer length: {d.smooth_buffer_seconds:g}s"),
                ("rewind_toggle", f"Rewind buffer: {rewind}"),
                ("rewind_length", f"Rewind length: {d.rewind_buffer_seconds:g}s"),
                ("video_back", "Back"),
            ]
        if self._menu_page == "capture":
            d = self.display
            folder = self._save_dir()
            return [
                ("snap_now", "Save snapshot"),
                ("clip_now", f"Save clip ({d.clip_seconds:g}s)"),
                ("clip_length", f"Clip length: {d.clip_seconds:g}s"),
                ("snap_format", f"Snapshot format: {d.snapshot_format.upper()}"),
                ("capture_folder", f"Folder: {folder}"),
                ("capture_back", "Back"),
            ]
        if self._menu_page == "detection":
            d = self.display
            people = "On" if d.people_detection else "Off"
            objects = "On" if d.object_detection else "Off"
            cam = self._target_camera()
            baseline_label = cam.name if cam else "(focus a camera)"
            return [
                ("people_master", f"People detection: {people}"),
                ("object_master", f"Object detection: {objects}"),
                ("detection_cams", "Cameras included…"),
                ("set_baseline", f"Set empty-area baseline: {baseline_label}"),
                ("detection_back", "Back"),
            ]
        if self._menu_page == "detection_cams":
            items: list[tuple[str, str]] = []
            for index, cam in enumerate(self.cameras):
                p = "On" if cam.detect_people else "Off"
                o = "On" if cam.detect_objects else "Off"
                items.append((f"cam_people:{index}", f"{cam.name} — people: {p}"))
                items.append((f"cam_objects:{index}", f"{cam.name} — objects: {o}"))
            items.append(("detection_cams_back", "Back"))
            return items
        if self._menu_page == "cameras":
            d = self.display
            enabled = sum(1 for cam in self.config.cameras if cam.enabled)
            return [
                ("layout", f"Layout: {d.columns}×{d.rows}  ({enabled} shown / {d.tile_count} slots)"),
                ("cycle_focus", f"Cycle focus: {d.cycle_focus_label}"),
                ("cameras_arrange", "Arrange tiles…"),
                ("cameras_toggle", "Show / hide cameras…"),
                ("cameras_add", "Add camera…"),
                ("cameras_remove", "Remove camera…"),
                ("cameras_back", "Back"),
            ]
        if self._menu_page == "cameras_arrange":
            items = []
            for index, cam in enumerate(self.config.cameras):
                flag = "" if cam.enabled else " (hidden)"
                items.append(
                    (
                        f"arrange:{index}",
                        f"{index + 1}. {cam.name}{flag}",
                    )
                )
            items.append(("cameras_arrange_back", "Back"))
            return items
        if self._menu_page == "cameras_toggle":
            items = []
            for index, cam in enumerate(self.config.cameras):
                state = "On" if cam.enabled else "Off"
                items.append((f"toggle:{index}", f"{cam.name} — {state}"))
            items.append(("cameras_toggle_back", "Back"))
            return items
        if self._menu_page == "cameras_add":
            return [
                ("add_rtsp", "Add RTSP / URL camera…"),
                ("add_webcam0", "Add webcam (device 0)"),
                ("add_webcam1", "Add webcam (device 1)"),
                ("add_demo", "Add demo camera"),
                ("cameras_add_back", "Back"),
            ]
        if self._menu_page == "cameras_remove":
            items = []
            for index, cam in enumerate(self.config.cameras):
                items.append((f"remove:{index}", f"Remove {cam.name}"))
            items.append(("cameras_remove_back", "Back"))
            return items
        fullscreen = "Windowed mode" if self.fullscreen else "Fullscreen"
        return [
            ("resume", "Resume"),
            ("fullscreen", fullscreen),
            ("cameras", "Cameras…"),
            ("capture", "Capture"),
            ("detection", "Detection"),
            ("video", "Video settings"),
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
        elif action == "capture":
            self._menu_page = "capture"
            self._menu_index = 0
            self._reboot_notice = ""
        elif action == "capture_back":
            self._menu_page = "root"
            self._menu_index = 0
        elif action == "detection":
            self._menu_page = "detection"
            self._menu_index = 0
            self._reboot_notice = ""
            if self.display.people_detection:
                backend = self._detection.ensure_ready()
                if backend == "unavailable":
                    self._reboot_notice = "People detector unavailable"
                else:
                    self._reboot_notice = f"People backend: {backend}"
        elif action == "detection_back":
            self._menu_page = "root"
            self._menu_index = 0
        elif action == "detection_cams":
            self._menu_page = "detection_cams"
            self._menu_index = 0
            self._reboot_notice = ""
        elif action == "detection_cams_back":
            self._menu_page = "detection"
            self._menu_index = 0
        elif action == "people_master":
            self._set_people_detection(not self.display.people_detection)
        elif action == "object_master":
            self._set_object_detection(not self.display.object_detection)
        elif action == "set_baseline":
            self._set_baseline_for_target()
        elif action.startswith("cam_people:"):
            index = int(action.split(":", 1)[1])
            self._toggle_camera_detection(index, people=True)
        elif action.startswith("cam_objects:"):
            index = int(action.split(":", 1)[1])
            self._toggle_camera_detection(index, people=False)
        elif action == "snap_now":
            self._save_snapshot()
            self._menu_open = False
            self._menu_page = "root"
        elif action == "clip_now":
            self._save_clip()
            self._menu_open = False
            self._menu_page = "root"
        elif action == "clip_length":
            self._adjust_menu_item("clip_length", 1)
        elif action == "snap_format":
            self._toggle_snapshot_format()
        elif action == "capture_folder":
            self._reboot_notice = str(self._save_dir())
        elif action == "video":
            self._menu_page = "video"
            self._menu_index = 0
            self._reboot_notice = ""
        elif action == "video_back":
            self._menu_page = "root"
            self._menu_index = 0
        elif action == "smooth_toggle":
            self._set_smooth_buffer(not self.display.smooth_buffer)
        elif action == "smooth_length":
            self._adjust_menu_item("smooth_length", 1)
        elif action == "rewind_toggle":
            self._set_rewind_buffer(not self.display.rewind_buffer)
        elif action == "rewind_length":
            self._adjust_menu_item("rewind_length", 1)
        elif action == "cameras":
            self._menu_page = "cameras"
            self._menu_index = 0
            self._reboot_notice = ""
        elif action == "cameras_back":
            self._menu_page = "root"
            self._menu_index = 0
        elif action == "layout":
            self._adjust_menu_item("layout", 1)
        elif action == "cycle_focus":
            self._adjust_menu_item("cycle_focus", 1)
        elif action == "cameras_arrange":
            self._menu_page = "cameras_arrange"
            self._menu_index = 0
            self._reboot_notice = "← → move camera in grid order"
        elif action == "cameras_arrange_back":
            self._menu_page = "cameras"
            self._menu_index = 0
        elif action.startswith("arrange:"):
            # Enter focuses; ← → in adjust moves.
            self._reboot_notice = "Use ← → to move this camera"
        elif action == "cameras_toggle":
            self._menu_page = "cameras_toggle"
            self._menu_index = 0
            self._reboot_notice = ""
        elif action == "cameras_toggle_back":
            self._menu_page = "cameras"
            self._menu_index = 0
        elif action.startswith("toggle:"):
            index = int(action.split(":", 1)[1])
            self._toggle_camera_enabled(index)
        elif action == "cameras_add":
            self._menu_page = "cameras_add"
            self._menu_index = 0
            self._reboot_notice = ""
        elif action == "cameras_add_back":
            self._menu_page = "cameras"
            self._menu_index = 0
        elif action == "add_rtsp":
            self._prompt = TextPrompt(title="Camera name", kind="add_name")
        elif action == "add_webcam0":
            self._add_webcam(0)
        elif action == "add_webcam1":
            self._add_webcam(1)
        elif action == "add_demo":
            self._add_demo_camera()
        elif action == "cameras_remove":
            self._menu_page = "cameras_remove"
            self._menu_index = 0
            self._reboot_notice = "Enter removes permanently from config"
        elif action == "cameras_remove_back":
            self._menu_page = "cameras"
            self._menu_index = 0
        elif action.startswith("remove:"):
            index = int(action.split(":", 1)[1])
            self._remove_camera(index)
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

    def _adjust_menu_item(self, action: str, step: int) -> None:
        if action == "smooth_toggle":
            self._set_smooth_buffer(not self.display.smooth_buffer)
        elif action == "rewind_toggle":
            self._set_rewind_buffer(not self.display.rewind_buffer)
        elif action == "smooth_length":
            was = self.display.smooth_buffer
            value = next_choice(SMOOTH_BUFFER_CHOICES, self.display.smooth_buffer_seconds, step)
            self.display.smooth_buffer_seconds = float(value)
            self.display.smooth_buffer = True
            self._apply_buffer_settings(persist=True)
            if not was:
                self._reconnect_all()
        elif action == "rewind_length":
            value = next_choice(REWIND_BUFFER_CHOICES, self.display.rewind_buffer_seconds, step)
            self.display.rewind_buffer_seconds = float(value)
            self.display.rewind_buffer = True
            self._apply_buffer_settings(persist=True)
        elif action == "clip_length":
            value = next_choice(CLIP_LENGTH_CHOICES, self.display.clip_seconds, step)
            self.display.clip_seconds = float(value)
            self._apply_buffer_settings(persist=True)
        elif action == "snap_format":
            self._toggle_snapshot_format()
        elif action == "people_master":
            self._set_people_detection(not self.display.people_detection)
        elif action == "object_master":
            self._set_object_detection(not self.display.object_detection)
        elif action.startswith("cam_people:"):
            self._activate_menu(action)
        elif action.startswith("cam_objects:"):
            self._activate_menu(action)
        elif action == "layout":
            cols, rows = next_layout_preset(self.display.columns, self.display.rows, step)
            self.display.columns = cols
            self.display.rows = rows
            self._rebuild_sources(persist=True)
            self._reboot_notice = f"Layout {cols}×{rows}"
        elif action == "cycle_focus":
            current = (
                float(self.display.cycle_focus_seconds)
                if self.display.cycle_focus
                else 0.0
            )
            value = next_choice(CYCLE_FOCUS_CHOICES, current, step)
            if float(value) <= 0:
                self.display.cycle_focus = False
            else:
                self.display.cycle_focus = True
                self.display.cycle_focus_seconds = float(value)
            self._cycle_deadline = 0.0
            self._apply_buffer_settings(persist=True)
        elif action.startswith("arrange:"):
            index = int(action.split(":", 1)[1])
            new_index = move_camera(self.config.cameras, index, step)
            self._menu_index = new_index
            self._rebuild_sources(persist=True)
            self._reboot_notice = "Tile order saved"
        elif action.startswith("toggle:"):
            self._activate_menu(action)

    def _set_people_detection(self, enabled: bool) -> None:
        self.display.people_detection = bool(enabled)
        if enabled:
            backend = self._detection.ensure_ready()
            self._reboot_notice = (
                "People detector unavailable"
                if backend == "unavailable"
                else f"People backend: {backend}"
            )
        self._apply_buffer_settings(persist=True)
        print(f"People detection {'on' if enabled else 'off'}")

    def _set_object_detection(self, enabled: bool) -> None:
        self.display.object_detection = bool(enabled)
        self._apply_buffer_settings(persist=True)
        print(f"Object detection {'on' if enabled else 'off'}")

    def _toggle_camera_detection(self, index: int, *, people: bool) -> None:
        if index < 0 or index >= len(self.cameras):
            return
        cam = self.cameras[index]
        # Keep the same object identity as config.cameras for persistence.
        for cfg_cam in self.config.cameras:
            if cfg_cam.name == cam.name:
                if people:
                    cfg_cam.detect_people = not cfg_cam.detect_people
                    cam.detect_people = cfg_cam.detect_people
                else:
                    cfg_cam.detect_objects = not cfg_cam.detect_objects
                    cam.detect_objects = cfg_cam.detect_objects
                break
        self._camera_by_name[cam.name] = cam
        self._apply_buffer_settings(persist=True)

    def _target_camera(self) -> CameraConfig | None:
        if self.zoom_index is not None and 0 <= self.zoom_index < len(self.cameras):
            return self.cameras[self.zoom_index]
        if self.cameras:
            return self.cameras[0]
        return None

    def _set_baseline_for_target(self) -> None:
        cam = self._target_camera()
        if cam is None:
            self._reboot_notice = "No camera available for baseline"
            return
        # Prefer the live (undelayed) frame for a clean empty-area reference.
        source = next((s for s in self.sources if s.name == cam.name), None)
        frame = None
        if source is not None:
            snap = source.snapshot()
            frame = snap.frame
        if frame is None:
            self._reboot_notice = f"No frame for {cam.name}"
            return
        try:
            path = self._detection.objects.set_baseline(cam.name, frame)
        except (OSError, ValueError) as exc:
            self._reboot_notice = f"Baseline failed: {exc}"
            return
        # Opt the camera into object detection when capturing a baseline.
        cam.detect_objects = True
        for cfg_cam in self.config.cameras:
            if cfg_cam.name == cam.name:
                cfg_cam.detect_objects = True
                break
        if not self.display.object_detection:
            self.display.object_detection = True
        self._apply_buffer_settings(persist=True)
        self._reboot_notice = f"Baseline saved: {path.name}"
        print(f"Baseline for {cam.name} → {path}")

    def _set_smooth_buffer(self, enabled: bool) -> None:
        previous = self.display.smooth_buffer
        self.display.smooth_buffer = bool(enabled)
        self._apply_buffer_settings(persist=True)
        print(
            f"Smooth buffer {'on' if enabled else 'off'}"
            f" ({self.display.smooth_buffer_seconds:g}s)"
        )
        # Capture open-options change with the smooth toggle — reconnect once.
        if previous != enabled:
            self._reconnect_all()

    def _set_rewind_buffer(self, enabled: bool) -> None:
        self.display.rewind_buffer = bool(enabled)
        if not enabled:
            self._go_live()
        self._apply_buffer_settings(persist=True)
        print(
            f"Rewind buffer {'on' if enabled else 'off'}"
            f" ({self.display.rewind_buffer_seconds:g}s)"
        )

    def _apply_buffer_settings(self, *, persist: bool = False) -> None:
        for source in self.sources:
            source.apply_buffer_settings(self.display)
        if persist:
            path = save_display_settings(self.config)
            if path is not None:
                self._reboot_notice = f"Saved to {path.name}"
            elif self._menu_page in {
                "video",
                "detection",
                "detection_cams",
                *_CAMERA_MENU_PAGES,
            }:
                self._reboot_notice = "Settings applied (demo — not saved)"

    def _rebuild_sources(self, *, persist: bool = False) -> None:
        """Stop/start workers to match config.visible_cameras() and layout."""
        previous_focus_name = None
        if self.zoom_index is not None and 0 <= self.zoom_index < len(self.sources):
            previous_focus_name = self.sources[self.zoom_index].name
        for source in self.sources:
            source.stop()
        self.cameras = self.config.visible_cameras()
        self._camera_by_name = {cam.name: cam for cam in self.cameras}
        self.sources = build_sources(self.cameras, self.display)
        for source in self.sources:
            source.start()
            source.apply_buffer_settings(self.display)
            print(f"Started {source.name}")
        if previous_focus_name:
            for i, source in enumerate(self.sources):
                if source.name == previous_focus_name:
                    self.zoom_index = i
                    break
            else:
                self.zoom_index = None
                self._reset_view()
        elif self.zoom_index is not None and self.zoom_index >= len(self.sources):
            self.zoom_index = None
            self._reset_view()
        self._cycle_deadline = 0.0
        if persist:
            path = save_display_settings(self.config)
            if path is not None:
                self._reboot_notice = f"Saved to {path.name}"
            else:
                self._reboot_notice = "Applied (demo — not saved)"

    def _toggle_camera_enabled(self, index: int) -> None:
        if index < 0 or index >= len(self.config.cameras):
            return
        cam = self.config.cameras[index]
        cam.enabled = not cam.enabled
        if cam.enabled:
            enabled = sum(1 for c in self.config.cameras if c.enabled)
            if ensure_layout_fits(self.display, enabled):
                self._reboot_notice = (
                    f"Enabled {cam.name}; layout → "
                    f"{self.display.columns}×{self.display.rows}"
                )
            else:
                self._reboot_notice = f"Enabled {cam.name}"
        else:
            self._reboot_notice = f"Hidden {cam.name}"
        self._rebuild_sources(persist=True)

    def _add_camera(self, cam: CameraConfig) -> None:
        self.config.cameras.append(cam)
        enabled = sum(1 for c in self.config.cameras if c.enabled)
        ensure_layout_fits(self.display, enabled)
        self._rebuild_sources(persist=True)
        self._menu_page = "cameras"
        self._menu_index = 0
        self._reboot_notice = f"Added {cam.name}"
        print(f"Added camera {cam.name} ({cam.redacted_source()})")

    def _add_webcam(self, device: int) -> None:
        name = unique_camera_name(self.config.cameras, f"Webcam {device}")
        self._add_camera(CameraConfig(name=name, device=int(device), enabled=True))

    def _add_demo_camera(self) -> None:
        n = sum(1 for c in self.config.cameras if (c.url or "").startswith("demo://"))
        name = unique_camera_name(self.config.cameras, f"Demo Cam {n + 1}")
        self._add_camera(CameraConfig(name=name, url=f"demo://{n}", enabled=True))

    def _remove_camera(self, index: int) -> None:
        if index < 0 or index >= len(self.config.cameras):
            return
        cam = self.config.cameras.pop(index)
        self._rebuild_sources(persist=True)
        self._menu_index = min(self._menu_index, max(0, len(self._menu_items()) - 1))
        self._reboot_notice = f"Removed {cam.name}"
        print(f"Removed camera {cam.name}")

    def _cycle_focus(self, step: int = 1) -> None:
        if not self.sources:
            return
        if self.zoom_index is None:
            index = 0 if step >= 0 else len(self.sources) - 1
        else:
            index = (self.zoom_index + int(step)) % len(self.sources)
        self._focus_camera(index)
        self._cycle_deadline = time.monotonic() + max(1.0, self.display.cycle_focus_seconds)

    def _tick_cycle_focus(self) -> None:
        if not self.display.cycle_focus or self.display.cycle_focus_seconds <= 0:
            return
        if self._menu_open or self._prompt is not None or self._reboot_job is not None:
            return
        if not self.sources:
            return
        now = time.monotonic()
        if self._cycle_deadline <= 0:
            self._cycle_deadline = now + self.display.cycle_focus_seconds
            return
        if now < self._cycle_deadline:
            return
        self._cycle_focus(1)

    def _handle_prompt_key(self, key: int, ch: int) -> None:
        prompt = self._prompt
        if prompt is None:
            return
        if key in KEY_ESC or ch == 27:
            self._prompt = None
            self._reboot_notice = "Add cancelled"
            return
        if key in KEY_ENTER or ch in (13, 10):
            self._finish_prompt()
            return
        if ch in (8, 127):  # backspace / delete
            prompt.value = prompt.value[:-1]
            return
        # Printable ASCII (skip control chars). Qt sometimes sends full key codes.
        if 32 <= ch <= 126:
            if len(prompt.value) < 240:
                prompt.value += chr(ch)

    def _finish_prompt(self) -> None:
        prompt = self._prompt
        if prompt is None:
            return
        text = prompt.value.strip()
        if prompt.kind == "add_name":
            if not text:
                self._reboot_notice = "Name required"
                return
            self._prompt = TextPrompt(
                title="RTSP / HTTP / file URL",
                kind="add_url",
                pending_name=text,
            )
            return
        if prompt.kind == "add_url":
            self._prompt = None
            if not text:
                self._reboot_notice = "URL required"
                return
            name = unique_camera_name(self.config.cameras, prompt.pending_name)
            self._add_camera(CameraConfig(name=name, url=text, enabled=True))
            return
        self._prompt = None

    def _draw_prompt(self, canvas: np.ndarray) -> np.ndarray:
        prompt = self._prompt
        if prompt is None:
            return canvas
        h, w = canvas.shape[:2]
        card_w, card_h = min(720, w - 24), 160
        x0 = max(12, (w - card_w) // 2)
        y0 = max(12, (h - card_h) // 2)
        shade_round_rect(
            canvas,
            (x0, y0, x0 + card_w, y0 + card_h),
            color=(18, 18, 22),
            alpha=0.94,
            radius=16,
        )
        draw_text(
            canvas,
            prompt.title,
            (x0 + card_w // 2, y0 + 18),
            size=20,
            align="center",
            valign="top",
        )
        shown = prompt.value if prompt.value else " "
        cursor = shown + ("▌" if int(time.monotonic() * 2) % 2 == 0 else " ")
        shade_round_rect(
            canvas,
            (x0 + 20, y0 + 58, x0 + card_w - 20, y0 + 100),
            color=(40, 40, 48),
            alpha=0.9,
            radius=8,
        )
        draw_text(
            canvas,
            cursor[-72:],
            (x0 + 32, y0 + 68),
            size=16,
            color=(236, 236, 236),
            valign="top",
        )
        draw_text(
            canvas,
            "Enter confirm    Esc cancel    type to edit",
            (x0 + card_w // 2, y0 + card_h - 28),
            size=13,
            color=(150, 150, 150),
            align="center",
            valign="top",
        )
        return canvas

    def _nudge_rewind(self, delta_seconds: float) -> None:
        if not self.display.rewind_buffer:
            return
        for source in self.sources:
            source.history.nudge_rewind(delta_seconds)

    def _go_live(self) -> None:
        for source in self.sources:
            source.history.go_live()

    def _save_dir(self) -> Path:
        return resolve_save_directory(self.display.save_directory or str(default_save_directory()))

    def _capture_label(self) -> str:
        if self.zoom_index is not None and 0 <= self.zoom_index < len(self.sources):
            return self.sources[self.zoom_index].name
        return "mosaic"

    def _current_capture_frame(self) -> np.ndarray | None:
        """Best frame for snapshot: focused camera if any, else live mosaic compose."""
        if self.zoom_index is not None and 0 <= self.zoom_index < len(self.sources):
            return self.sources[self.zoom_index].snapshot().frame
        # Build a clean grid without menus/help for the saved image.
        d = self.display
        cell_w, cell_h, width, height = self._sync_layout()
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (12, 12, 14)
        grid_w = cell_w * d.columns
        grid_h = cell_h * d.rows
        x_off = max(0, (width - grid_w) // 2)
        y_off = max(0, (height - grid_h) // 2)
        for index in range(d.tile_count):
            row, col = divmod(index, d.columns)
            y, x = y_off + row * cell_h, x_off + col * cell_w
            if index < len(self.sources):
                snap = self.sources[index].snapshot()
                tile = self._render_cell(
                    snap,
                    self.sources[index].name,
                    cell_w,
                    cell_h,
                    overlay=True,
                )
            else:
                tile = placeholder(cell_w, cell_h, "Empty", "No camera assigned")
            canvas[y : y + cell_h, x : x + cell_w] = tile
        return canvas

    def _flash_capture(self, message: str, *, seconds: float = 3.0) -> None:
        self._capture_flash = message
        self._capture_flash_until = time.monotonic() + seconds
        self._reboot_notice = message
        print(message)

    def _toggle_snapshot_format(self) -> None:
        self.display.snapshot_format = "png" if self.display.snapshot_format == "jpg" else "jpg"
        self._apply_buffer_settings(persist=True)

    def _save_snapshot(self) -> None:
        frame = self._current_capture_frame()
        if frame is None:
            self._flash_capture("Snapshot failed: no frame")
            return
        try:
            path = save_snapshot(
                frame,
                self._save_dir(),
                self._capture_label(),
                fmt=self.display.snapshot_format,
            )
        except CaptureError as exc:
            self._flash_capture(f"Snapshot failed: {exc}")
            return
        self._flash_capture(f"Saved snapshot {path.name}")

    def _save_clip(self) -> None:
        if self._clip_job is not None and not self._clip_job.finished:
            self._flash_capture("Clip already recording…")
            return
        seconds = float(self.display.clip_seconds)
        label = self._capture_label()
        source = None
        if self.zoom_index is not None and 0 <= self.zoom_index < len(self.sources):
            source = self.sources[self.zoom_index]
        elif len(self.sources) == 1:
            source = self.sources[0]

        # Prefer exporting recent history when we have enough samples.
        if source is not None:
            frames, fps = source.history.export_frames(seconds)
            if len(frames) >= max(3, int(seconds)):
                try:
                    path = write_clip(
                        frames,
                        self._save_dir(),
                        label,
                        fps=fps or float(self.display.fps),
                    )
                except CaptureError as exc:
                    self._flash_capture(f"Clip failed: {exc}")
                    return
                self._flash_capture(f"Saved clip {path.name}")
                return

        # Fall back to recording the next N seconds of the live view.
        self._clip_job = LiveClipJob.start(
            label=label,
            directory=self._save_dir(),
            duration=seconds,
            fps=float(self.display.fps),
        )
        self._flash_capture(f"Recording clip {seconds:g}s…", seconds=seconds + 1.0)

    def _feed_clip_job(self, frame: np.ndarray) -> None:
        job = self._clip_job
        if job is None or job.finished:
            return
        # Prefer the focused camera's native frame while recording a single cam.
        feed = frame
        if self.zoom_index is not None and 0 <= self.zoom_index < len(self.sources):
            cam = self.sources[self.zoom_index].snapshot().frame
            if cam is not None:
                feed = cam
        if job.feed(feed):
            if job.path is not None:
                self._flash_capture(f"Saved clip {job.path.name}")
            else:
                self._flash_capture(f"Clip failed: {job.error or 'unknown error'}")
            self._clip_job = None

    def _draw_capture_hud(self, canvas: np.ndarray) -> None:
        job = self._clip_job
        if job is not None and not job.finished:
            label = f"REC {job.remaining:0.1f}s"
            draw_text(
                canvas,
                label,
                (canvas.shape[1] // 2, 14),
                size=18,
                color=(60, 60, 230),
                align="center",
                valign="top",
            )
        if self._capture_flash and time.monotonic() <= self._capture_flash_until:
            draw_text(
                canvas,
                self._capture_flash,
                (canvas.shape[1] // 2, canvas.shape[0] - 18),
                size=15,
                color=(220, 220, 220),
                align="center",
                valign="bottom",
            )

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
        if not self._menu_open or self._reboot_job is not None or self._prompt is not None:
            return canvas
        canvas[:] = (canvas.astype(np.float32) * 0.38).astype(np.uint8)
        items = self._menu_items()
        wide = self._menu_page in {
            "video",
            "capture",
            "detection",
            "detection_cams",
            *_CAMERA_MENU_PAGES,
        }
        card_w, row_h, pad = (600 if wide else 480), 46, 20
        title_h = 56
        footer_h = 36
        notice_h = 28 if self._reboot_notice else 0
        # Cap visible rows so huge camera lists still fit; scroll via menu index.
        max_rows = max(4, min(len(items), max(4, (canvas.shape[0] - 120) // row_h)))
        start = 0
        if len(items) > max_rows:
            start = min(
                max(0, self._menu_index - max_rows // 2),
                len(items) - max_rows,
            )
        visible = items[start : start + max_rows]
        card_h = title_h + notice_h + row_h * len(visible) + footer_h
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
        heading = {
            "reboot_confirm": "Confirm reboot",
            "video": "Video settings",
            "capture": "Capture",
            "detection": "Detection",
            "detection_cams": "Cameras for detection",
            "cameras": "Cameras",
            "cameras_arrange": "Arrange tiles",
            "cameras_toggle": "Show / hide",
            "cameras_add": "Add camera",
            "cameras_remove": "Remove camera",
        }.get(self._menu_page, "Options")
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
        for i, (action, label) in enumerate(visible):
            absolute = start + i
            ry = y0 + title_h + notice_h + i * row_h
            box = (x0 + pad, ry, x0 + card_w - pad, ry + row_h - 8)
            if absolute == self._menu_index:
                shade_round_rect(canvas, box, color=(56, 120, 78), alpha=0.7, radius=8)
            draw_text(
                canvas,
                f"{absolute + 1}",
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
        if self._menu_page == "reboot_confirm":
            footer = "This will restart every camera"
        elif self._menu_page == "video":
            footer = "Enter toggle/cycle    ← → adjust    Esc back"
        elif self._menu_page == "capture":
            footer = "s snapshot  c clip    ← → length    Esc back"
        elif self._menu_page == "cameras_arrange":
            footer = "← → move in grid order    Esc back"
        elif self._menu_page in _CAMERA_MENU_PAGES or self._menu_page in {
            "detection",
            "detection_cams",
        }:
            footer = "Enter select    ← → adjust    Esc back"
        else:
            footer = "Enter to select    Esc to close"
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
        self._cycle_deadline = time.monotonic() + max(1.0, self.display.cycle_focus_seconds)

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

    def _draw_buffer_badge(self, canvas: np.ndarray, snap: Snapshot) -> None:
        d = self.display
        if not d.smooth_buffer and not d.rewind_buffer:
            return
        parts: list[str] = []
        if snap.rewinding or (d.rewind_buffer and snap.behind > (d.smooth_buffer_seconds if d.smooth_buffer else 0) + 0.2):
            parts.append(f"REWIND -{snap.behind:.1f}s")
        elif d.smooth_buffer and snap.behind > 0.05:
            parts.append(f"SMOOTH -{snap.behind:.1f}s")
        elif d.smooth_buffer:
            parts.append(f"SMOOTH {d.smooth_buffer_seconds:g}s")
        if d.rewind_buffer:
            parts.append(f"buf {snap.buffered:.0f}/{d.rewind_buffer_seconds:g}s")
        if not parts:
            return
        label = "   ".join(parts)
        color = (60, 180, 255) if snap.rewinding else (200, 200, 200)
        draw_text(
            canvas,
            label,
            (16, 14),
            size=15,
            color=color,
            valign="top",
        )

    def _apply_fullscreen(self) -> None:
        prop = getattr(cv2, "WND_PROP_FULLSCREEN", None)
        mode_full = getattr(cv2, "WINDOW_FULLSCREEN", 1)
        mode_normal = getattr(cv2, "WINDOW_NORMAL", 0)
        if prop is None:
            return
        if self.fullscreen:
            screen = self._screen_size()
            if screen is not None:
                try:
                    cv2.resizeWindow(self.window, screen[0], screen[1])
                    cv2.moveWindow(self.window, 0, 0)
                except cv2.error:
                    pass
            cv2.setWindowProperty(self.window, prop, mode_full)
        else:
            cv2.setWindowProperty(self.window, prop, mode_normal)
            try:
                width, height = self.display.canvas_size
                cv2.resizeWindow(self.window, width, height)
            except cv2.error:
                pass

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
        px, py = self._event_to_pixel(x, y)
        local_x = px - self._grid_x
        local_y = py - self._grid_y
        grid_w = self._cell_w * self.display.columns
        grid_h = self._cell_h * self.display.rows
        if local_x < 0 or local_y < 0 or local_x >= grid_w or local_y >= grid_h:
            return
        col = min(self.display.columns - 1, max(0, int(local_x // max(self._cell_w, 1))))
        row = min(self.display.rows - 1, max(0, int(local_y // max(self._cell_h, 1))))
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
        ww, wh = self._window_size()
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
    if snap.rewinding:
        status = f"REWIND -{snap.behind:.1f}s"
    elif snap.detail and snap.status not in ("live", "demo"):
        status = f"{status}  {snap.detail}"
    if fps_text and not snap.rewinding:
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
