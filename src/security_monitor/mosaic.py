"""OpenCV mosaic window: grid of camera tiles, zoom, overlays."""

from __future__ import annotations

import os
import sys
import time
import math
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
    CaptureItem,
    LiveClipJob,
    default_save_directory,
    delete_capture,
    delete_captures,
    list_captures,
    load_capture_preview,
    resolve_save_directory,
    reveal_in_file_manager,
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
from security_monitor.decode import (
    decode_mode_label,
    next_decode_mode,
    next_hwaccel,
    opencv_decode_summary,
    probe_hwaccels,
)
from security_monitor.alarm import play_alert_beep
from security_monitor.detection import DetectionEngine, draw_boxes
from security_monitor.encroachment import (
    DEFAULT_ENCROACH_LINE,
    EncroachZone,
    draw_alarm_banner,
    draw_encroach_highlight,
    draw_zones,
    effective_zones,
    evaluate_zones,
    line_preset_label,
    map_tile_click_to_frame_norm,
    next_line_preset,
    next_polygon_preset,
    polygon_preset_label,
    unique_zone_name,
)
from security_monitor.events import (
    PERSON_POST_ROLL_CHOICES,
    PERSON_PRE_ROLL_CHOICES,
    PersonEventItem,
    PersonEventRecorder,
    delete_person_event,
    list_person_events,
)
from security_monitor.overlay import draw_dot, draw_text, shade_bottom_bar, shade_round_rect
from security_monitor.retention import (
    RETENTION_DAY_CHOICES,
    RETENTION_GB_CHOICES,
    PurgeResult,
    purge_old_media,
    retention_label_days,
    retention_label_gb,
    set_capture_locked,
    set_event_locked,
)
from security_monitor.stream import Snapshot, build_sources
from security_monitor.weather import (
    HUD_OPACITY_CHOICES,
    WEATHER_OPACITY_CHOICES,
    WeatherRect,
    WeatherService,
    compute_tile_rects,
    draw_weather_widget,
    next_opacity,
    next_weather_slot,
    nudge_norm,
    opacity_label,
    resolve_weather_rect,
    slot_label,
)

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
_CAPTURE_BROWSER_PAGES = frozenset(
    {
        "captures",
        "captures_view",
        "captures_delete",
        "captures_delete_all",
    }
)
_EVENT_BROWSER_PAGES = frozenset(
    {
        "events",
        "events_view",
        "events_delete",
        "events_play",
    }
)
_NESTED_MENU_PAGES = frozenset(
    {
        "reboot_confirm",
        "video",
        "capture",
        "detection",
        "detection_cams",
        "decode_status",
        "weather",
        "weather_place",
        *_CAMERA_MENU_PAGES,
        *_CAPTURE_BROWSER_PAGES,
        *_EVENT_BROWSER_PAGES,
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
        self._captures: list[CaptureItem] = []
        self._captures_origin = "capture"  # capture | root
        self._capture_view_index: int | None = None
        self._capture_preview: np.ndarray | None = None
        self._person_seen: dict[str, bool] = {}
        self._person_recorders: dict[str, PersonEventRecorder] = {}
        self._person_cooldown_until: dict[str, float] = {}
        self._encroach_active: dict[str, bool] = {}
        self._encroach_zone_hits: dict[str, tuple[str, ...]] = {}
        self._encroach_prev: dict[str, bool] = {}
        self._encroach_owned_focus = False
        self._zone_edit_name: str | None = None
        self._zone_edit_mode: str | None = None  # line | polygon
        self._zone_edit_points: list[tuple[float, float]] = []
        self._alarm_until = 0.0
        self._alarm_last_beep = 0.0
        self._line_edit_name: str | None = None  # alias kept for older checks
        self._line_edit_point: tuple[float, float] | None = None
        self._events: list[PersonEventItem] = []
        self._events_origin = "detection"
        self._event_view_index: int | None = None
        self._event_preview: np.ndarray | None = None
        self._event_playback: cv2.VideoCapture | None = None
        self._event_playback_label = ""
        self._last_purge_at = 0.0
        self._weather = WeatherService()
        self._weather_rect: WeatherRect | None = None
        self._weather_place_mode = False
        self._weather_place_still: np.ndarray | None = None
        self._weather_drag_offset: tuple[int, int] | None = None
        self._configure_weather_service()

    def _configure_weather_service(self) -> None:
        d = self.display
        self._weather.configure(
            enabled=bool(d.weather_enabled),
            latitude=d.weather_latitude,
            longitude=d.weather_longitude,
            place=d.weather_place or "",
            refresh_seconds=float(d.weather_refresh_seconds or 300),
        )

    def _start_weather_place_editor(self) -> None:
        # Freeze a clean grid still; drag the widget on top.
        still = self._current_capture_frame(include_weather=False)
        if still is None:
            self._reboot_notice = "Could not capture layout still"
            return
        self._weather_place_still = still
        self._weather_place_mode = True
        self._weather_drag_offset = None
        self.display.weather_enabled = True
        self.display.weather_slot = "custom"
        self._configure_weather_service()
        self._menu_open = False
        self._menu_page = "root"
        self._flash_capture("Drag the weather widget — Enter saves, Esc cancels", seconds=4.0)

    def _finish_weather_place_editor(self, *, save: bool) -> None:
        if not self._weather_place_mode:
            return
        self._weather_place_mode = False
        self._weather_place_still = None
        self._weather_drag_offset = None
        if save:
            self.display.weather_enabled = True
            self._apply_buffer_settings(persist=True)
            self._flash_capture("Weather placement saved")
            self._menu_open = True
            self._menu_page = "weather"
            self._menu_index = 0
        else:
            self._flash_capture("Weather placement cancelled")
            self._menu_open = True
            self._menu_page = "weather"
            self._menu_index = 0

    def _nudge_weather_place(self, dx: int, dy: int) -> None:
        step = 0.02
        self.display.weather_slot = "custom"
        self.display.weather_x = nudge_norm(self.display.weather_x, step * dx, lo=0.0, hi=0.88)
        self.display.weather_y = nudge_norm(self.display.weather_y, step * dy, lo=0.0, hi=0.88)

    def _set_weather_norm_from_pixel(self, px: int, py: int) -> None:
        """Set custom top-left from canvas pixel (accounting for drag grab offset)."""
        d = self.display
        grid_w = max(1, self._cell_w * d.columns)
        grid_h = max(1, self._cell_h * d.rows)
        ox, oy = self._weather_drag_offset or (0, 0)
        left = px - ox
        top = py - oy
        nx = (left - self._grid_x) / grid_w
        ny = (top - self._grid_y) / grid_h
        d.weather_slot = "custom"
        d.weather_x = max(0.0, min(1.0 - d.weather_w, nx))
        d.weather_y = max(0.0, min(1.0 - d.weather_h, ny))

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
                self._maybe_purge_media()
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
        self._stop_event_playback()
        self._weather.stop()
        for name, recorder in list(self._person_recorders.items()):
            # Force-complete any open person event so the clip is written.
            recorder.lost_at = time.monotonic() - recorder.post_roll - 1.0
            recorder.feed(None, person_present=False)
            self._person_recorders.pop(name, None)
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
        self._tick_person_events()
        self._update_encroachment_state()
        d = self.display
        cell_w, cell_h, width, height = self._sync_layout()
        if self._weather_place_mode and self._weather_place_still is not None:
            return self._finalize_ui(self._paint_weather_place_editor(width, height))
        if self._event_playback is not None:
            return self._finalize_ui(self._paint_event_playback(width, height))
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
            return self._finalize_ui(cell)

        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (12, 12, 14)
        grid_w = cell_w * d.columns
        grid_h = cell_h * d.rows
        x_off = max(0, (width - grid_w) // 2)
        y_off = max(0, (height - grid_h) // 2)
        self._grid_x, self._grid_y = x_off, y_off
        reserved = self._resolve_weather_rect(width, height, x_off, y_off, grid_w, grid_h)
        self._weather_rect = reserved
        # Overlay mode keeps full camera tiles and paints weather on top.
        shrink = None if (reserved is None or d.weather_overlay) else reserved
        tile_rects = compute_tile_rects(
            columns=d.columns,
            rows=d.rows,
            cell_w=cell_w,
            cell_h=cell_h,
            grid_x=x_off,
            grid_y=y_off,
            reserved=shrink,
        )
        zoomed = self.view_zoom > 1.001
        any_rewind = False
        for index in range(d.tile_count):
            tx, ty, tw, th = tile_rects[index]
            if tw < 2 or th < 2:
                continue
            if index < len(self.sources):
                snap = self.sources[index].snapshot()
                any_rewind = any_rewind or snap.rewinding
                tile = self._render_cell(
                    snap,
                    self.sources[index].name,
                    tw,
                    th,
                    overlay=not zoomed,
                )
            else:
                tile = placeholder(tw, th, "Empty", "No camera assigned")
            canvas[ty : ty + th, tx : tx + tw] = tile
        if reserved is not None and not zoomed:
            self._paint_weather_widget(canvas, reserved, editing=False)
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
        return self._finalize_ui(canvas)

    def _resolve_weather_rect(
        self,
        width: int,
        height: int,
        grid_x: int,
        grid_y: int,
        grid_w: int,
        grid_h: int,
    ) -> WeatherRect | None:
        d = self.display
        if not d.weather_enabled and not self._weather_place_mode:
            return None
        return resolve_weather_rect(
            slot=d.weather_slot,
            norm_x=d.weather_x,
            norm_y=d.weather_y,
            norm_w=d.weather_w,
            norm_h=d.weather_h,
            grid_x=grid_x,
            grid_y=grid_y,
            grid_w=grid_w,
            grid_h=grid_h,
            columns=d.columns,
            rows=d.rows,
            canvas_w=width,
            canvas_h=height,
        )

    def _paint_weather_widget(
        self,
        canvas: np.ndarray,
        rect: WeatherRect,
        *,
        editing: bool = False,
    ) -> None:
        d = self.display
        draw_weather_widget(
            canvas,
            rect,
            self._weather.snapshot,
            units=d.weather_units,
            show_temp=d.weather_show_temp,
            show_conditions=d.weather_show_conditions,
            show_storm=d.weather_show_storm,
            show_lightning=d.weather_show_lightning,
            opacity=d.weather_opacity,
            editing=editing,
        )

    def _paint_weather_place_editor(self, width: int, height: int) -> np.ndarray:
        still = self._weather_place_still
        if still is None:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            canvas = still.copy()
            if canvas.shape[0] != height or canvas.shape[1] != width:
                canvas = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)
        # Dim slightly so the draggable widget reads clearly.
        canvas[:] = (canvas.astype(np.float32) * 0.72).astype(np.uint8)
        d = self.display
        grid_w = self._cell_w * d.columns
        grid_h = self._cell_h * d.rows
        reserved = self._resolve_weather_rect(
            width, height, self._grid_x, self._grid_y, grid_w, grid_h
        )
        self._weather_rect = reserved
        if reserved is not None:
            # Preview layout impact: shrink outlines unless overlay mode.
            shrink = None if d.weather_overlay else reserved
            tile_rects = compute_tile_rects(
                columns=d.columns,
                rows=d.rows,
                cell_w=self._cell_w,
                cell_h=self._cell_h,
                grid_x=self._grid_x,
                grid_y=self._grid_y,
                reserved=shrink,
            )
            for tx, ty, tw, th in tile_rects:
                if tw > 2 and th > 2:
                    cv2.rectangle(canvas, (tx, ty), (tx + tw - 1, ty + th - 1), (70, 90, 70), 1)
            self._paint_weather_widget(canvas, reserved, editing=True)
        draw_text(
            canvas,
            "Drag widget  ·  ←→↑↓ nudge  ·  Enter save  ·  Esc cancel",
            (width // 2, height - 18),
            size=14,
            color=(220, 220, 230),
            align="center",
            valign="bottom",
        )
        return canvas

    def _finalize_ui(self, canvas: np.ndarray) -> np.ndarray:
        if self.display.encroachment_alarm and (
            any(self._encroach_active.values()) or time.monotonic() < self._alarm_until
        ):
            labels = [
                name
                for name, on in self._encroach_active.items()
                if on
            ]
            if not labels:
                labels = ["alert"]
            pulse = 0.55 + 0.45 * abs(math.sin(time.monotonic() * 7.0))
            draw_alarm_banner(canvas, labels, pulse=pulse)
        if self._menu_open and self._menu_page in {"captures_view", "captures_delete"}:
            if self._capture_preview is not None:
                canvas = self._paint_capture_preview(
                    canvas.shape[1], canvas.shape[0], self._capture_preview
                )
        if self._menu_open and self._menu_page in {"events_view", "events_delete"}:
            if self._event_preview is not None:
                canvas = self._paint_capture_preview(
                    canvas.shape[1], canvas.shape[0], self._event_preview
                )
        if self._event_playback is not None and self._menu_page == "events_play":
            # Playback canvas already painted; keep menu card on top.
            pass
        return self._draw_prompt(self._draw_menu(self._draw_reboot(self._draw_help(canvas))))

    def _paint_capture_preview(
        self, width: int, height: int, frame: np.ndarray
    ) -> np.ndarray:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (10, 10, 12)
        fitted = scale_frame(frame, width, height, "fit")
        y = max(0, (height - fitted.shape[0]) // 2)
        x = max(0, (width - fitted.shape[1]) // 2)
        canvas[y : y + fitted.shape[0], x : x + fitted.shape[1]] = fitted
        # Dim slightly so the action card stays readable.
        canvas[:] = (canvas.astype(np.float32) * 0.72).astype(np.uint8)
        return canvas

    def _paint_event_playback(self, width: int, height: int) -> np.ndarray:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (8, 8, 10)
        cap = self._event_playback
        if cap is None:
            return canvas
        ok, frame = cap.read()
        if not ok or frame is None:
            # Loop the clip.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if ok and frame is not None:
            fitted = scale_frame(frame, width, height, "fit")
            y = max(0, (height - fitted.shape[0]) // 2)
            x = max(0, (width - fitted.shape[1]) // 2)
            canvas[y : y + fitted.shape[0], x : x + fitted.shape[1]] = fitted
            shade_bottom_bar(canvas, height=36, alpha=0.55)
            draw_text(
                canvas,
                f"Playing  {self._event_playback_label}   Esc stop",
                (width // 2, height - 12),
                size=15,
                align="center",
                valign="bottom",
                color=(230, 230, 230),
            )
        return canvas

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
            want_people = bool(
                cam is not None
                and (
                    (self.display.people_detection and cam.detect_people)
                    or (
                        self.display.encroachment_detection
                        and cam.detect_encroachment
                    )
                )
            )
            want_objects = bool(
                cam is not None
                and self.display.object_detection
                and cam.detect_objects
            )
            boxes: list = []
            if cam is not None and (want_people or want_objects):
                boxes = self._detection.process(
                    name,
                    frame,
                    detect_people=want_people,
                    detect_objects=want_objects,
                )
            encroach_on = bool(
                cam is not None
                and self.display.encroachment_detection
                and cam.detect_encroachment
            )
            active = bool(self._encroach_active.get(name))
            draft = (
                self._zone_edit_points
                if self._zone_edit_name == name and self._zone_edit_points
                else None
            )
            if boxes or encroach_on or draft:
                frame = frame.copy()
                if encroach_on and cam is not None:
                    zones = self._camera_zones(cam)
                    draw_zones(
                        frame,
                        zones,
                        active_names=self._encroach_zone_hits.get(name, ()),
                        draft_points=draft,
                    )
                elif draft:
                    draw_zones(frame, [], draft_points=draft)
                if boxes:
                    draw_boxes(frame, boxes)
            mode = self.display.scale_mode if self.view_zoom <= 1.001 else "fill"
            tile = scale_frame(frame, width, height, mode)
            if active:
                pulse = 0.55 + 0.45 * abs(math.sin(time.monotonic() * 6.0))
                draw_encroach_highlight(
                    tile,
                    active=True,
                    pulse=pulse,
                    strong=bool(self.display.encroachment_alarm),
                )
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
        alert = bool(self._encroach_active.get(name))
        draw_status_bar(
            tile,
            name,
            snap,
            fps_text,
            encroach=alert,
            hud_opacity=self.display.hud_opacity,
        )
        if self._zone_edit_name == name:
            if self._zone_edit_mode == "polygon":
                n = len(self._zone_edit_points)
                tip = (
                    f"Polygon: {n} pts — click add, Enter finish (≥3), Esc cancel"
                    if n
                    else "Polygon: click corners  Enter finish  Esc cancel"
                )
            else:
                tip = (
                    "Tripwire: click second point"
                    if self._zone_edit_points
                    else "Tripwire: click first point  (Esc cancel)"
                )
            draw_text(
                tile,
                tip,
                (tile.shape[1] // 2, 18),
                size=14,
                color=(60, 160, 255),
                align="center",
                valign="top",
            )

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
        if self._weather_place_mode:
            if key in KEY_ESC or ch == 27:
                self._finish_weather_place_editor(save=False)
                return
            if key in KEY_ENTER or ch in (13, 10):
                self._finish_weather_place_editor(save=True)
                return
            if key in KEY_LEFT:
                self._nudge_weather_place(-1, 0)
                return
            if key in KEY_RIGHT:
                self._nudge_weather_place(1, 0)
                return
            if key in KEY_UP:
                self._nudge_weather_place(0, -1)
                return
            if key in KEY_DOWN:
                self._nudge_weather_place(0, 1)
                return
            return
        editing = self._zone_edit_name is not None or self._line_edit_name is not None
        if editing:
            if key in KEY_ESC or ch == 27 or ch in (ord("q"), ord("Q")):
                self._cancel_zone_edit()
                return
            if key in KEY_ENTER or ch in (13, 10):
                if self._zone_edit_mode == "polygon":
                    self._finish_zone_edit()
                return
            # Ignore other keys while drawing a zone.
            return
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
            if self._menu_page == "decode_status":
                self._menu_page = "video"
                self._menu_index = 0
                return
            if self._menu_page == "detection_cams":
                self._menu_page = "detection"
                self._menu_index = 0
                return
            if self._menu_page in {"captures_view", "captures_delete", "captures_delete_all"}:
                self._close_capture_view()
                self._menu_page = "captures"
                self._menu_index = 0
                return
            if self._menu_page == "captures":
                self._menu_page = self._captures_origin
                self._menu_index = 0
                return
            if self._menu_page == "events_play":
                self._stop_event_playback()
                self._menu_page = "events_view"
                self._menu_index = 0
                return
            if self._menu_page in {"events_view", "events_delete"}:
                self._close_event_view()
                self._menu_page = "events"
                self._menu_index = 0
                return
            if self._menu_page == "events":
                self._menu_page = self._events_origin
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
        if self._weather_place_mode:
            self._finish_weather_place_editor(save=False)
            return
        if self._zone_edit_name is not None or self._line_edit_name is not None:
            self._cancel_zone_edit()
            return
        if self._prompt is not None:
            self._prompt = None
            return
        if self._menu_open and self._menu_page == "decode_status":
            self._menu_page = "video"
            self._menu_index = 0
            return
        if self._menu_open and self._menu_page == "detection_cams":
            self._menu_page = "detection"
            self._menu_index = 0
            return
        if self._menu_open and self._menu_page in {
            "captures_view",
            "captures_delete",
            "captures_delete_all",
        }:
            self._close_capture_view()
            self._menu_page = "captures"
            self._menu_index = 0
            return
        if self._menu_open and self._menu_page == "captures":
            self._menu_page = self._captures_origin
            self._menu_index = 0
            return
        if self._menu_open and self._menu_page == "events_play":
            self._stop_event_playback()
            self._menu_page = "events_view"
            self._menu_index = 0
            return
        if self._menu_open and self._menu_page in {"events_view", "events_delete"}:
            self._close_event_view()
            self._menu_page = "events"
            self._menu_index = 0
            return
        if self._menu_open and self._menu_page == "events":
            self._menu_page = self._events_origin
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
            "weather",
            "cameras",
        }:
            self._menu_page = "root"
            self._menu_index = 0
            return
        action = escape_action(menu_open=self._menu_open, on_main_layout=self._is_main_layout())
        if action == "close_menu":
            self._menu_open = False
            self._menu_page = "root"
            self._close_capture_view()
            self._stop_event_playback()
            self._close_event_view()
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
            decode = decode_mode_label(d.decode_mode, d.hwaccel)
            return [
                ("smooth_toggle", f"Smooth buffer: {smooth}"),
                ("smooth_length", f"Buffer length: {d.smooth_buffer_seconds:g}s"),
                ("rewind_toggle", f"Rewind buffer: {rewind}"),
                ("rewind_length", f"Rewind length: {d.rewind_buffer_seconds:g}s"),
                ("decode_mode", f"Decode: {d.decode_mode.upper()} — {decode}"),
                ("hwaccel", f"HW backend: {d.hwaccel}"),
                ("hud_opacity", f"HUD opacity: {opacity_label(d.hud_opacity)}"),
                ("decode_status", "Decode status…"),
                ("video_back", "Back"),
            ]
        if self._menu_page == "decode_status":
            items: list[tuple[str, str]] = [
                ("decode_summary", opencv_decode_summary()[:70]),
            ]
            accels = probe_hwaccels()
            items.append(
                (
                    "decode_available",
                    "Available: " + (", ".join(accels) if accels else "none detected"),
                )
            )
            for source in self.sources:
                snap = source.snapshot()
                label = snap.decode or "—"
                items.append((f"decode_cam:{source.name}", f"{source.name}: {label}"))
            items.append(("decode_status_back", "Back"))
            return items
        if self._menu_page == "capture":
            d = self.display
            folder = self._save_dir()
            return [
                ("snap_now", "Save snapshot"),
                ("clip_now", f"Save clip ({d.clip_seconds:g}s)"),
                ("captures_browse", "Browse saved captures…"),
                ("clip_length", f"Clip length: {d.clip_seconds:g}s"),
                ("snap_format", f"Snapshot format: {d.snapshot_format.upper()}"),
                (
                    "retention_days",
                    f"Auto-erase after: {retention_label_days(d.capture_retention_days)}",
                ),
                (
                    "retention_gb",
                    f"Max storage: {retention_label_gb(d.capture_max_gb)}",
                ),
                ("purge_now", "Erase old unlocked now…"),
                ("capture_folder", f"Folder: {folder}"),
                ("capture_back", "Back"),
            ]
        if self._menu_page == "captures":
            items: list[tuple[str, str]] = [
                ("captures_refresh", f"Refresh list ({len(self._captures)} files)"),
                ("captures_open_folder", "Open captures folder"),
            ]
            if self._captures:
                items.append(("captures_delete_all", "Delete all captures…"))
            for index, item in enumerate(self._captures):
                items.append((f"cap:{index}", item.label))
            if not self._captures:
                items.append(("captures_empty", "(no snapshots or clips yet)"))
            items.append(("captures_back", "Back"))
            return items
        if self._menu_page == "captures_view":
            item = self._selected_capture()
            title = item.name if item else "(missing)"
            kind = item.kind if item else "?"
            lock = "Unlock" if item and item.locked else "Lock (keep forever)"
            return [
                ("captures_info", f"{title}  ({kind})"),
                ("captures_prev", "◀ Previous"),
                ("captures_next", "Next ▶"),
                ("captures_lock", lock),
                ("captures_open_folder", "Show in folder"),
                ("captures_delete", "Delete…"),
                ("captures_view_back", "Back to list"),
            ]
        if self._menu_page == "captures_delete":
            item = self._selected_capture()
            name = item.name if item else "file"
            return [
                ("captures_delete_yes", f"Yes, delete {name}"),
                ("captures_delete_no", "Cancel"),
            ]
        if self._menu_page == "captures_delete_all":
            return [
                ("captures_delete_all_yes", f"Yes, delete all {len(self._captures)} files"),
                ("captures_delete_all_no", "Cancel"),
            ]
        if self._menu_page == "detection":
            d = self.display
            people = "On" if d.people_detection else "Off"
            objects = "On" if d.object_detection else "Off"
            auto = "On" if d.auto_person_capture else "Off"
            encroach = "On" if d.encroachment_detection else "Off"
            autofocus = "On" if d.encroachment_autofocus else "Off"
            alarm = "On" if d.encroachment_alarm else "Off"
            sound = "On" if d.encroachment_alarm_sound else "Off"
            cam = self._target_camera()
            baseline_label = cam.name if cam else "(focus a camera)"
            zones = self._camera_zones(cam) if cam else []
            zone_count = len(zones)
            line_label = line_preset_label(cam.encroach_line if cam else None)
            poly_label = "Bottom half"
            if cam and cam.encroach_zones:
                last_poly = next((z for z in reversed(cam.encroach_zones) if z.is_polygon), None)
                if last_poly is not None:
                    poly_label = polygon_preset_label(last_poly.points)
            side = (cam.encroach_side if cam else "positive") or "positive"
            return [
                ("people_master", f"People detection: {people}"),
                ("object_master", f"Object detection: {objects}"),
                ("encroach_master", f"Encroachment: {encroach}"),
                ("encroach_autofocus", f"Autofocus on encroach: {autofocus}"),
                ("encroach_alarm", f"On-screen alarm: {alarm}"),
                ("encroach_sound", f"Alarm sound: {sound}"),
                ("auto_person", f"Auto person capture: {auto}"),
                ("person_pre", f"Pre-roll: {d.person_pre_roll_seconds:g}s"),
                ("person_post", f"Post-roll: {d.person_post_roll_seconds:g}s"),
                ("events_browse", "Person events…"),
                ("detection_cams", "Cameras included…"),
                ("encroach_zones_info", f"Zones on {baseline_label}: {zone_count}"),
                ("encroach_preset", f"Add tripwire preset: {line_label}"),
                ("encroach_side", f"Tripwire zone side: {side}"),
                ("encroach_edit", f"Draw tripwire: {baseline_label}"),
                ("encroach_poly_preset", f"Add polygon preset: {poly_label}"),
                ("encroach_poly_edit", f"Draw polygon ROI: {baseline_label}"),
                ("encroach_clear_zones", f"Clear all zones: {baseline_label}"),
                ("set_baseline", f"Set empty-area baseline: {baseline_label}"),
                ("detection_back", "Back"),
            ]
        if self._menu_page == "events":
            items: list[tuple[str, str]] = [
                ("events_refresh", f"Refresh list ({len(self._events)} events)"),
            ]
            for index, item in enumerate(self._events):
                items.append((f"event:{index}", item.label))
            if not self._events:
                items.append(("events_empty", "(no person events yet)"))
            items.append(("events_back", "Back"))
            return items
        if self._menu_page == "events_view":
            item = self._selected_event()
            title = item.label if item else "(missing)"
            play = "Play recording" if item and item.has_clip else "Play recording (unavailable)"
            lock = "Unlock" if item and item.locked else "Lock (keep forever)"
            return [
                ("events_info", title),
                ("events_play", play),
                ("events_prev", "◀ Previous"),
                ("events_next", "Next ▶"),
                ("events_lock", lock),
                ("events_delete", "Delete event…"),
                ("events_view_back", "Back to list"),
            ]
        if self._menu_page == "events_play":
            return [
                ("events_play_stop", "Stop playback"),
                ("events_play_back", "Back to event"),
            ]
        if self._menu_page == "events_delete":
            item = self._selected_event()
            name = item.when.strftime("%Y-%m-%d %H:%M:%S") if item else "event"
            return [
                ("events_delete_yes", f"Yes, delete {name}"),
                ("events_delete_no", "Cancel"),
            ]
        if self._menu_page == "detection_cams":
            items: list[tuple[str, str]] = []
            for index, cam in enumerate(self.cameras):
                p = "On" if cam.detect_people else "Off"
                o = "On" if cam.detect_objects else "Off"
                e = "On" if cam.detect_encroachment else "Off"
                items.append((f"cam_people:{index}", f"{cam.name} — people: {p}"))
                items.append((f"cam_objects:{index}", f"{cam.name} — objects: {o}"))
                items.append((f"cam_encroach:{index}", f"{cam.name} — encroach: {e}"))
            items.append(("detection_cams_back", "Back"))
            return items
        if self._menu_page == "weather":
            d = self.display
            enabled = "On" if d.weather_enabled else "Off"
            units = "°F" if d.weather_units == "f" else "°C"
            snap = self._weather.snapshot
            loc = snap.place or d.weather_place or (
                f"{d.weather_latitude:.2f},{d.weather_longitude:.2f}"
                if d.weather_latitude is not None and d.weather_longitude is not None
                else "Auto (IP)"
            )
            return [
                ("weather_toggle", f"Weather HUD: {enabled}"),
                ("weather_slot", f"Placement: {slot_label(d.weather_slot)}"),
                ("weather_place", "Place widget on layout…"),
                ("weather_opacity", f"Opacity: {opacity_label(d.weather_opacity)}  (← →)"),
                (
                    "weather_overlay",
                    f"Overlay cameras: {'On' if d.weather_overlay else 'Off'}",
                ),
                ("weather_nudge_x", f"Fine X: {d.weather_x:+.2f}  (← →)"),
                ("weather_nudge_y", f"Fine Y: {d.weather_y:+.2f}  (← →)"),
                ("weather_size_w", f"Width: {d.weather_w:.2f}"),
                ("weather_size_h", f"Height: {d.weather_h:.2f}"),
                ("weather_units", f"Temperature units: {units}"),
                ("weather_temp", f"Show temperature: {'On' if d.weather_show_temp else 'Off'}"),
                (
                    "weather_conditions",
                    f"Show conditions: {'On' if d.weather_show_conditions else 'Off'}",
                ),
                ("weather_storm", f"Show storm warnings: {'On' if d.weather_show_storm else 'Off'}"),
                (
                    "weather_lightning",
                    f"Show lightning tracker: {'On' if d.weather_show_lightning else 'Off'}",
                ),
                ("weather_refresh", "Refresh weather now"),
                ("weather_loc", f"Location: {loc[:42]}"),
                ("weather_back", "Back"),
            ]
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
            ("captures_root", "Saved captures…"),
            ("events_root", "Person events…"),
            ("detection", "Detection"),
            ("weather", "Weather HUD…"),
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
        elif action == "weather":
            self._menu_page = "weather"
            self._menu_index = 0
            self._configure_weather_service()
            snap = self._weather.snapshot
            self._reboot_notice = (
                snap.error
                if snap.error and not snap.ok
                else f"{snap.place or 'Local'}: {snap.temp_label(self.display.weather_units)} {snap.condition}"
            )
        elif action == "weather_back":
            self._menu_page = "root"
            self._menu_index = 0
        elif action == "weather_toggle":
            self.display.weather_enabled = not self.display.weather_enabled
            self._configure_weather_service()
            self._apply_buffer_settings(persist=True)
            self._reboot_notice = (
                "Weather HUD on" if self.display.weather_enabled else "Weather HUD off"
            )
        elif action == "weather_slot":
            self._adjust_menu_item("weather_slot", 1)
        elif action == "weather_opacity":
            self._adjust_menu_item("weather_opacity", 1)
        elif action == "weather_overlay":
            self.display.weather_overlay = not self.display.weather_overlay
            self._apply_buffer_settings(persist=True)
            self._reboot_notice = (
                "Weather overlays cameras"
                if self.display.weather_overlay
                else "Weather reserves camera space"
            )
        elif action == "weather_place":
            self._start_weather_place_editor()
        elif action in {
            "weather_nudge_x",
            "weather_nudge_y",
            "weather_size_w",
            "weather_size_h",
        }:
            self._adjust_menu_item(action, 1)
        elif action == "weather_units":
            self.display.weather_units = "c" if self.display.weather_units == "f" else "f"
            self._apply_buffer_settings(persist=True)
        elif action == "weather_temp":
            self.display.weather_show_temp = not self.display.weather_show_temp
            self._apply_buffer_settings(persist=True)
        elif action == "weather_conditions":
            self.display.weather_show_conditions = not self.display.weather_show_conditions
            self._apply_buffer_settings(persist=True)
        elif action == "weather_storm":
            self.display.weather_show_storm = not self.display.weather_show_storm
            self._apply_buffer_settings(persist=True)
        elif action == "weather_lightning":
            self.display.weather_show_lightning = not self.display.weather_show_lightning
            self._apply_buffer_settings(persist=True)
        elif action == "weather_refresh":
            self._configure_weather_service()
            self._reboot_notice = "Refreshing weather…"
        elif action == "weather_loc":
            self._reboot_notice = (
                "Set weather_latitude / weather_longitude in config.yaml (blank = auto)"
            )
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
        elif action == "encroach_master":
            self._set_encroachment_detection(not self.display.encroachment_detection)
        elif action == "encroach_autofocus":
            self._set_encroachment_autofocus(not self.display.encroachment_autofocus)
        elif action == "encroach_alarm":
            self.display.encroachment_alarm = not self.display.encroachment_alarm
            self._apply_buffer_settings(persist=True)
            self._reboot_notice = (
                "On-screen alarm on" if self.display.encroachment_alarm else "On-screen alarm off"
            )
        elif action == "encroach_sound":
            self.display.encroachment_alarm_sound = not self.display.encroachment_alarm_sound
            self._apply_buffer_settings(persist=True)
            self._reboot_notice = (
                "Alarm sound on" if self.display.encroachment_alarm_sound else "Alarm sound off"
            )
            if self.display.encroachment_alarm_sound:
                play_alert_beep(double=False)
        elif action == "encroach_preset":
            self._adjust_menu_item("encroach_preset", 1)
        elif action == "encroach_side":
            self._adjust_menu_item("encroach_side", 1)
        elif action == "encroach_edit":
            self._start_zone_edit("line")
        elif action == "encroach_poly_preset":
            self._adjust_menu_item("encroach_poly_preset", 1)
        elif action == "encroach_poly_edit":
            self._start_zone_edit("polygon")
        elif action == "encroach_clear_zones":
            self._clear_camera_zones()
        elif action == "encroach_zones_info":
            cam = self._target_camera()
            if cam is None:
                self._reboot_notice = "Focus a camera first"
            else:
                zones = self._camera_zones(cam)
                names = ", ".join(z.name for z in zones) or "(none)"
                self._reboot_notice = f"{cam.name}: {names}"
        elif action == "auto_person":
            self._set_auto_person_capture(not self.display.auto_person_capture)
        elif action == "person_pre":
            self._adjust_menu_item("person_pre", 1)
        elif action == "person_post":
            self._adjust_menu_item("person_post", 1)
        elif action == "events_browse" or action == "events_root":
            self._events_origin = "root" if action == "events_root" else "detection"
            self._open_events_browser()
        elif action == "events_back":
            self._close_event_view()
            self._stop_event_playback()
            self._menu_page = self._events_origin
            self._menu_index = 0
        elif action == "events_refresh":
            self._refresh_events()
            self._reboot_notice = f"{len(self._events)} event(s)"
        elif action == "events_empty":
            self._reboot_notice = "Enable Auto person capture + people on a camera"
        elif action.startswith("event:"):
            index = int(action.split(":", 1)[1])
            self._open_event_view(index)
        elif action == "events_info":
            item = self._selected_event()
            if item is not None and item.has_clip:
                self._start_event_playback()
            elif item is not None:
                self._reboot_notice = str(item.path)
        elif action == "events_prev":
            self._nudge_event_view(-1)
        elif action == "events_next":
            self._nudge_event_view(1)
        elif action == "events_lock":
            self._toggle_selected_event_lock()
        elif action == "events_play":
            self._start_event_playback()
        elif action == "events_play_stop" or action == "events_play_back":
            self._stop_event_playback()
            self._menu_page = "events_view"
            self._menu_index = 0
        elif action == "events_delete":
            item = self._selected_event()
            if item is not None and item.locked:
                self._reboot_notice = "Unlock before deleting"
            else:
                self._menu_page = "events_delete"
                self._menu_index = 1
        elif action == "events_delete_yes":
            self._delete_selected_event()
        elif action == "events_delete_no" or action == "events_view_back":
            if action == "events_view_back":
                self._close_event_view()
                self._menu_page = "events"
                self._menu_index = 0
            else:
                self._menu_page = "events_view"
                self._menu_index = 0
        elif action == "set_baseline":
            self._set_baseline_for_target()
        elif action.startswith("cam_people:"):
            index = int(action.split(":", 1)[1])
            self._toggle_camera_detection(index, people=True)
        elif action.startswith("cam_objects:"):
            index = int(action.split(":", 1)[1])
            self._toggle_camera_detection(index, people=False)
        elif action.startswith("cam_encroach:"):
            index = int(action.split(":", 1)[1])
            self._toggle_camera_encroachment(index)
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
        elif action == "retention_days":
            self._adjust_menu_item("retention_days", 1)
        elif action == "retention_gb":
            self._adjust_menu_item("retention_gb", 1)
        elif action == "purge_now":
            result = self._run_media_purge(force=True)
            self._reboot_notice = result.summary
        elif action == "capture_folder":
            self._reboot_notice = str(self._save_dir())
        elif action == "captures_browse" or action == "captures_root":
            self._captures_origin = "root" if action == "captures_root" else "capture"
            self._open_captures_browser()
        elif action == "captures_back":
            self._close_capture_view()
            self._menu_page = self._captures_origin
            self._menu_index = 0
        elif action == "captures_refresh":
            self._refresh_captures()
            self._reboot_notice = f"{len(self._captures)} file(s)"
        elif action == "captures_open_folder":
            self._reveal_captures_folder()
        elif action == "captures_empty":
            self._reboot_notice = "Save a snapshot (s) or clip (c) first"
        elif action.startswith("cap:"):
            index = int(action.split(":", 1)[1])
            self._open_capture_view(index)
        elif action == "captures_info":
            item = self._selected_capture()
            if item is not None:
                self._reboot_notice = str(item.path)
        elif action == "captures_prev":
            self._nudge_capture_view(-1)
        elif action == "captures_next":
            self._nudge_capture_view(1)
        elif action == "captures_lock":
            self._toggle_selected_capture_lock()
        elif action == "captures_delete":
            item = self._selected_capture()
            if item is not None and item.locked:
                self._reboot_notice = "Unlock before deleting"
            else:
                self._menu_page = "captures_delete"
                self._menu_index = 1
        elif action == "captures_delete_yes":
            self._delete_selected_capture()
        elif action == "captures_delete_no" or action == "captures_view_back":
            if action == "captures_view_back":
                self._close_capture_view()
                self._menu_page = "captures"
                self._menu_index = 0
            else:
                self._menu_page = "captures_view"
                self._menu_index = 0
        elif action == "captures_delete_all":
            self._menu_page = "captures_delete_all"
            self._menu_index = 1
        elif action == "captures_delete_all_yes":
            self._delete_all_captures()
        elif action == "captures_delete_all_no":
            self._menu_page = "captures"
            self._menu_index = 0
        elif action == "video":
            self._menu_page = "video"
            self._menu_index = 0
            self._reboot_notice = ""
        elif action == "video_back":
            self._menu_page = "root"
            self._menu_index = 0
        elif action == "decode_mode":
            self._adjust_menu_item("decode_mode", 1)
        elif action == "hwaccel":
            self._adjust_menu_item("hwaccel", 1)
        elif action == "hud_opacity":
            self._adjust_menu_item("hud_opacity", 1)
        elif action == "decode_status":
            self._menu_page = "decode_status"
            self._menu_index = 0
            self._reboot_notice = opencv_decode_summary()
        elif action == "decode_status_back":
            self._menu_page = "video"
            self._menu_index = 0
        elif action == "decode_summary" or action == "decode_available":
            self._reboot_notice = opencv_decode_summary()
        elif action.startswith("decode_cam:"):
            name = action.split(":", 1)[1]
            source = next((s for s in self.sources if s.name == name), None)
            if source is not None:
                self._reboot_notice = f"{name}: {source.snapshot().decode or '—'}"
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
        elif action == "decode_mode":
            previous = self.display.decode_mode
            self.display.decode_mode = next_decode_mode(self.display.decode_mode, step)
            self._apply_buffer_settings(persist=True)
            self._reboot_notice = decode_mode_label(
                self.display.decode_mode, self.display.hwaccel
            )
            if previous != self.display.decode_mode:
                self._reconnect_all()
        elif action == "hud_opacity":
            self.display.hud_opacity = next_opacity(
                HUD_OPACITY_CHOICES, self.display.hud_opacity, step
            )
            self._apply_buffer_settings(persist=True)
            self._reboot_notice = f"HUD opacity {opacity_label(self.display.hud_opacity)}"
        elif action == "hwaccel":
            previous = self.display.hwaccel
            self.display.hwaccel = next_hwaccel(self.display.hwaccel, step)
            self._apply_buffer_settings(persist=True)
            self._reboot_notice = f"HW backend: {self.display.hwaccel}"
            if previous != self.display.hwaccel:
                self._reconnect_all()
        elif action == "clip_length":
            value = next_choice(CLIP_LENGTH_CHOICES, self.display.clip_seconds, step)
            self.display.clip_seconds = float(value)
            self._apply_buffer_settings(persist=True)
        elif action == "retention_days":
            value = next_choice(
                RETENTION_DAY_CHOICES, self.display.capture_retention_days, step
            )
            self.display.capture_retention_days = float(value)
            self._apply_buffer_settings(persist=True)
        elif action == "retention_gb":
            value = next_choice(RETENTION_GB_CHOICES, self.display.capture_max_gb, step)
            self.display.capture_max_gb = float(value)
            self._apply_buffer_settings(persist=True)
        elif action == "snap_format":
            self._toggle_snapshot_format()
        elif action == "people_master":
            self._set_people_detection(not self.display.people_detection)
        elif action == "object_master":
            self._set_object_detection(not self.display.object_detection)
        elif action == "encroach_master":
            self._set_encroachment_detection(not self.display.encroachment_detection)
        elif action == "encroach_autofocus":
            self._set_encroachment_autofocus(not self.display.encroachment_autofocus)
        elif action == "encroach_alarm":
            self._activate_menu("encroach_alarm")
        elif action == "encroach_sound":
            self._activate_menu("encroach_sound")
        elif action == "encroach_preset":
            cam = self._target_camera()
            if cam is None:
                self._reboot_notice = "Focus a camera first"
                return
            coords = next_line_preset(cam.encroach_line, step)
            cam.encroach_line = coords
            zone = EncroachZone(
                name=unique_zone_name(cam.encroach_zones, "Tripwire"),
                points=((coords[0], coords[1]), (coords[2], coords[3])),
                side=cam.encroach_side or "positive",
            )
            self._append_camera_zone(cam, zone, also_set_legacy_line=True)
            self._reboot_notice = f"Added tripwire: {line_preset_label(coords)}"
        elif action == "encroach_side":
            cam = self._target_camera()
            if cam is None:
                self._reboot_notice = "Focus a camera first"
                return
            side = "negative" if (cam.encroach_side or "positive") == "positive" else "positive"
            cam.encroach_side = side
            # Flip side on legacy line zones / last tripwire zone.
            updated: list[EncroachZone] = []
            for zone in cam.encroach_zones:
                if zone.is_line:
                    updated.append(
                        EncroachZone(name=zone.name, points=zone.points, side=side)
                    )
                else:
                    updated.append(zone)
            cam.encroach_zones = updated
            for cfg_cam in self.config.cameras:
                if cfg_cam.name == cam.name:
                    cfg_cam.encroach_side = side
                    cfg_cam.encroach_zones = list(updated)
                    break
            self._apply_buffer_settings(persist=True)
            self._reboot_notice = f"Tripwire zone side: {side}"
        elif action == "encroach_poly_preset":
            cam = self._target_camera()
            if cam is None:
                self._reboot_notice = "Focus a camera first"
                return
            last = next((z for z in reversed(cam.encroach_zones) if z.is_polygon), None)
            name, pts = next_polygon_preset(last.points if last else None, step)
            zone = EncroachZone(
                name=unique_zone_name(cam.encroach_zones, name),
                points=pts,
            )
            self._append_camera_zone(cam, zone)
            self._reboot_notice = f"Added polygon: {name}"
        elif action == "auto_person":
            self._set_auto_person_capture(not self.display.auto_person_capture)
        elif action == "person_pre":
            value = next_choice(
                PERSON_PRE_ROLL_CHOICES, self.display.person_pre_roll_seconds, step
            )
            self.display.person_pre_roll_seconds = float(value)
            self._apply_buffer_settings(persist=True)
        elif action == "person_post":
            value = next_choice(
                PERSON_POST_ROLL_CHOICES, self.display.person_post_roll_seconds, step
            )
            self.display.person_post_roll_seconds = float(value)
            self._apply_buffer_settings(persist=True)
        elif action.startswith("cam_people:"):
            self._activate_menu(action)
        elif action.startswith("cam_objects:"):
            self._activate_menu(action)
        elif action.startswith("cam_encroach:"):
            self._activate_menu(action)
        elif action == "weather_slot":
            self.display.weather_slot = next_weather_slot(self.display.weather_slot, step)
            # Preset placements reset fine offsets for a clean jump.
            if self.display.weather_slot != "custom":
                self.display.weather_x = 0.0
                self.display.weather_y = 0.0
            self._apply_buffer_settings(persist=True)
            self._reboot_notice = slot_label(self.display.weather_slot)
        elif action == "weather_opacity":
            self.display.weather_opacity = next_opacity(
                WEATHER_OPACITY_CHOICES, self.display.weather_opacity, step
            )
            self._apply_buffer_settings(persist=True)
            self._reboot_notice = f"Weather opacity {opacity_label(self.display.weather_opacity)}"
        elif action == "weather_nudge_x":
            self.display.weather_x = nudge_norm(self.display.weather_x, 0.02 * step)
            if self.display.weather_slot != "custom":
                # Fine-tuning a preset keeps the slot but records offsets.
                pass
            self._apply_buffer_settings(persist=True)
        elif action == "weather_nudge_y":
            self.display.weather_y = nudge_norm(self.display.weather_y, 0.02 * step)
            self._apply_buffer_settings(persist=True)
        elif action == "weather_size_w":
            self.display.weather_w = max(0.12, min(0.55, self.display.weather_w + 0.02 * step))
            self._apply_buffer_settings(persist=True)
        elif action == "weather_size_h":
            self.display.weather_h = max(0.10, min(0.45, self.display.weather_h + 0.02 * step))
            self._apply_buffer_settings(persist=True)
        elif action in {
            "weather_toggle",
            "weather_units",
            "weather_temp",
            "weather_conditions",
            "weather_storm",
            "weather_lightning",
            "weather_overlay",
        }:
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
        elif self._menu_page == "events_view":
            self._nudge_event_view(step)
        elif action.startswith("event:"):
            index = int(action.split(":", 1)[1])
            target = index + (1 if step > 0 else -1)
            if 0 <= target < len(self._events):
                items = self._menu_items()
                for i, (act, _) in enumerate(items):
                    if act == f"event:{target}":
                        self._menu_index = i
                        break
        elif self._menu_page == "captures_view":
            self._nudge_capture_view(step)
        elif action.startswith("cap:"):
            # Jump selection with arrows while browsing the list.
            index = int(action.split(":", 1)[1])
            target = index + (1 if step > 0 else -1)
            if 0 <= target < len(self._captures):
                # Move highlight to neighboring file row.
                items = self._menu_items()
                for i, (act, _) in enumerate(items):
                    if act == f"cap:{target}":
                        self._menu_index = i
                        break

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

    def _set_auto_person_capture(self, enabled: bool) -> None:
        self.display.auto_person_capture = bool(enabled)
        if enabled:
            self.display.people_detection = True
            backend = self._detection.ensure_ready()
            self._reboot_notice = (
                "Auto capture on — enable cameras under Cameras included"
                if backend != "unavailable"
                else "People detector unavailable"
            )
            if backend != "unavailable":
                self._reboot_notice = f"Auto capture on ({backend}) — include cameras"
        else:
            # Let active recorders finish with post-roll by marking absent.
            self._reboot_notice = "Auto person capture off"
        self._apply_buffer_settings(persist=True)
        print(f"Auto person capture {'on' if enabled else 'off'}")

    def _set_encroachment_detection(self, enabled: bool) -> None:
        self.display.encroachment_detection = bool(enabled)
        if enabled:
            self.display.people_detection = True
            backend = self._detection.ensure_ready()
            if backend == "unavailable":
                self._reboot_notice = "People detector unavailable"
            else:
                self._reboot_notice = f"Encroachment on ({backend}) — include cameras"
        else:
            self._encroach_owned_focus = False
            self._reboot_notice = "Encroachment off"
        self._apply_buffer_settings(persist=True)
        print(f"Encroachment detection {'on' if enabled else 'off'}")

    def _set_encroachment_autofocus(self, enabled: bool) -> None:
        self.display.encroachment_autofocus = bool(enabled)
        if not enabled and self._encroach_owned_focus:
            self._encroach_owned_focus = False
            self._go_main_layout()
        self._apply_buffer_settings(persist=True)
        self._reboot_notice = (
            "Autofocus on encroach" if enabled else "Autofocus on encroach off"
        )
        print(f"Encroachment autofocus {'on' if enabled else 'off'}")

    def _camera_zones(self, cam: CameraConfig | None) -> list[EncroachZone]:
        if cam is None:
            return []
        return effective_zones(
            cam.encroach_zones,
            legacy_line=cam.encroach_line,
            legacy_side=cam.encroach_side,
        )

    def _sync_camera_zones(self, cam: CameraConfig) -> None:
        for cfg_cam in self.config.cameras:
            if cfg_cam.name == cam.name:
                cfg_cam.encroach_zones = list(cam.encroach_zones)
                cfg_cam.encroach_line = cam.encroach_line
                cfg_cam.encroach_side = cam.encroach_side
                cfg_cam.detect_encroachment = cam.detect_encroachment
                cfg_cam.detect_people = cam.detect_people
                break
        self._camera_by_name[cam.name] = cam

    def _append_camera_zone(
        self,
        cam: CameraConfig,
        zone: EncroachZone,
        *,
        also_set_legacy_line: bool = False,
    ) -> None:
        cam.encroach_zones = [*cam.encroach_zones, zone]
        cam.detect_encroachment = True
        cam.detect_people = True
        if also_set_legacy_line and zone.is_line:
            line = zone.as_line_tuple()
            if line is not None:
                cam.encroach_line = line
                cam.encroach_side = zone.side
        self.display.encroachment_detection = True
        self.display.people_detection = True
        self._sync_camera_zones(cam)
        self._apply_buffer_settings(persist=True)

    def _clear_camera_zones(self) -> None:
        cam = self._target_camera()
        if cam is None:
            self._reboot_notice = "Focus a camera first"
            return
        cam.encroach_zones = []
        cam.encroach_line = None
        self._sync_camera_zones(cam)
        self._apply_buffer_settings(persist=True)
        self._reboot_notice = f"Cleared zones on {cam.name}"

    def _toggle_camera_encroachment(self, index: int) -> None:
        if index < 0 or index >= len(self.cameras):
            return
        cam = self.cameras[index]
        for cfg_cam in self.config.cameras:
            if cfg_cam.name == cam.name:
                cfg_cam.detect_encroachment = not cfg_cam.detect_encroachment
                cam.detect_encroachment = cfg_cam.detect_encroachment
                if cam.detect_encroachment:
                    cfg_cam.detect_people = True
                    cam.detect_people = True
                    if not cam.encroach_zones and cam.encroach_line is None:
                        cfg_cam.encroach_line = DEFAULT_ENCROACH_LINE
                        cam.encroach_line = DEFAULT_ENCROACH_LINE
                break
        self._camera_by_name[cam.name] = cam
        if cam.detect_encroachment and not self.display.encroachment_detection:
            self.display.encroachment_detection = True
            self.display.people_detection = True
            self._detection.ensure_ready()
        self._apply_buffer_settings(persist=True)

    def _start_zone_edit(self, mode: str) -> None:
        cam = self._target_camera()
        if cam is None:
            self._reboot_notice = "Focus a camera first"
            return
        index = next((i for i, c in enumerate(self.cameras) if c.name == cam.name), None)
        if index is None:
            self._reboot_notice = "Camera not visible"
            return
        self._zone_edit_name = cam.name
        self._zone_edit_mode = mode
        self._zone_edit_points = []
        self._line_edit_name = cam.name  # keep alias for cancel paths
        self._line_edit_point = None
        self._menu_open = False
        self._menu_page = "root"
        self._focus_camera(index)
        if mode == "polygon":
            self._flash_capture(
                f"Draw polygon on {cam.name}: click corners, Enter to finish",
                seconds=4.0,
            )
        else:
            self._flash_capture(f"Draw tripwire on {cam.name}: click two points", seconds=4.0)
        print(f"Zone edit ({mode}): {cam.name}")

    def _cancel_zone_edit(self) -> None:
        if self._zone_edit_name is None and self._line_edit_name is None:
            return
        self._zone_edit_name = None
        self._zone_edit_mode = None
        self._zone_edit_points = []
        self._line_edit_name = None
        self._line_edit_point = None
        self._flash_capture("Zone edit cancelled")

    def _finish_zone_edit(self) -> None:
        name = self._zone_edit_name
        mode = self._zone_edit_mode
        points = list(self._zone_edit_points)
        self._zone_edit_name = None
        self._zone_edit_mode = None
        self._zone_edit_points = []
        self._line_edit_name = None
        self._line_edit_point = None
        if name is None or mode is None:
            return
        cam = self._camera_by_name.get(name)
        if cam is None:
            return
        if mode == "line":
            if len(points) < 2:
                self._flash_capture("Need two points for a tripwire")
                return
            p1, p2 = points[0], points[1]
            zone = EncroachZone(
                name=unique_zone_name(cam.encroach_zones, "Tripwire"),
                points=(p1, p2),
                side=cam.encroach_side or "positive",
            )
            self._append_camera_zone(cam, zone, also_set_legacy_line=True)
            self._flash_capture(f"Tripwire saved: {name}")
        else:
            if len(points) < 3:
                self._flash_capture("Need at least 3 points for a polygon")
                return
            zone = EncroachZone(
                name=unique_zone_name(cam.encroach_zones, "ROI"),
                points=tuple(points),
            )
            self._append_camera_zone(cam, zone)
            self._flash_capture(f"Polygon ROI saved: {name} ({len(points)} pts)")
        print(f"Zone saved for {name}: {zone.name} ({zone.kind})")

    # Back-compat wrappers used by older key/mouse paths.
    def _start_line_edit(self) -> None:
        self._start_zone_edit("line")

    def _cancel_line_edit(self) -> None:
        self._cancel_zone_edit()

    def _finish_line_edit(self, p1: tuple[float, float], p2: tuple[float, float]) -> None:
        self._zone_edit_points = [p1, p2]
        self._finish_zone_edit()

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
        history_clip = float(self.display.clip_seconds)
        if self.display.auto_person_capture:
            history_clip = max(history_clip, float(self.display.person_pre_roll_seconds))
        for source in self.sources:
            source.history.configure(
                smooth_enabled=self.display.smooth_buffer,
                smooth_seconds=self.display.smooth_buffer_seconds,
                rewind_enabled=self.display.rewind_buffer,
                rewind_seconds=self.display.rewind_buffer_seconds,
                clip_seconds=history_clip,
            )
            source.apply_buffer_settings(self.display)
            # Re-apply history clip after apply_buffer_settings overwrote it.
            source.history.configure(clip_seconds=history_clip)
        self._configure_weather_service()
        if persist:
            path = save_display_settings(self.config)
            if path is not None:
                self._reboot_notice = f"Saved to {path.name}"
            elif self._menu_page in {
                "video",
                "detection",
                "detection_cams",
                "weather",
                *_CAMERA_MENU_PAGES,
                *_EVENT_BROWSER_PAGES,
            }:
                self._reboot_notice = "Settings applied (demo — not saved)"

    def _update_encroachment_state(self) -> None:
        """Evaluate ROI zones, alarm on entry, optional autofocus."""
        active: dict[str, bool] = {}
        hits: dict[str, tuple[str, ...]] = {}
        editing = self._zone_edit_name is not None or self._line_edit_name is not None
        if not self.display.encroachment_detection or editing:
            for name in list(self._encroach_active):
                active[name] = False
            self._encroach_active = active
            self._encroach_zone_hits = hits
            if self._encroach_owned_focus and not editing:
                self._encroach_owned_focus = False
                self._go_main_layout()
            return

        for source in self.sources:
            cam = self._camera_by_name.get(source.name)
            if cam is None or not cam.detect_encroachment:
                active[source.name] = False
                hits[source.name] = ()
                continue
            snap = source.snapshot()
            frame = snap.frame
            if frame is None:
                active[source.name] = False
                hits[source.name] = ()
                continue
            boxes = self._detection.process(
                source.name,
                frame,
                detect_people=True,
                detect_objects=False,
            )
            people = [b for b in boxes if b.label == "person"]
            h, w = frame.shape[:2]
            zones = self._camera_zones(cam)
            is_on, zone_names = evaluate_zones(zones, people, w, h)
            active[source.name] = is_on
            hits[source.name] = zone_names

        now = time.monotonic()
        for name, is_on in active.items():
            if is_on and not self._encroach_prev.get(name, False):
                zone_bit = ""
                if hits.get(name):
                    zone_bit = f" [{', '.join(hits[name])}]"
                self._flash_capture(f"ENCROACHMENT: {name}{zone_bit}", seconds=3.0)
                print(f"Encroachment alert: {name}{zone_bit}")
                self._alarm_until = max(self._alarm_until, now + 6.0)
                if self.display.encroachment_alarm_sound:
                    play_alert_beep(double=True)
                    self._alarm_last_beep = now
        # Periodic reminder beep while anyone remains in a zone.
        any_active = any(active.values())
        if (
            any_active
            and self.display.encroachment_alarm_sound
            and now - self._alarm_last_beep >= 4.0
        ):
            play_alert_beep(double=False)
            self._alarm_last_beep = now
        if any_active and self.display.encroachment_alarm:
            self._alarm_until = max(self._alarm_until, now + 0.8)

        self._encroach_prev = dict(active)
        self._encroach_active = active
        self._encroach_zone_hits = hits
        self._apply_encroachment_focus()

    def _apply_encroachment_focus(self) -> None:
        if not self.display.encroachment_autofocus:
            if self._encroach_owned_focus:
                self._encroach_owned_focus = False
                self._go_main_layout()
            return
        if self._menu_open or self._prompt is not None or self._reboot_job is not None:
            return
        if self._zone_edit_name is not None or self._line_edit_name is not None:
            return
        active_indices = [
            i
            for i, source in enumerate(self.sources)
            if self._encroach_active.get(source.name)
        ]
        if active_indices:
            if self.zoom_index in active_indices:
                self._encroach_owned_focus = True
                return
            self._encroach_owned_focus = True
            self.zoom_index = active_indices[0]
            self._reset_view()
            self._cycle_deadline = time.monotonic() + max(
                1.0, self.display.cycle_focus_seconds
            )
            return
        if self._encroach_owned_focus:
            self._encroach_owned_focus = False
            self._go_main_layout()

    def _tick_person_events(self) -> None:
        """Rising-edge person capture: snapshot + pre/during/post clip per camera."""
        active = dict(self._person_recorders)
        if not self.display.auto_person_capture or not self.display.people_detection:
            for name, recorder in active.items():
                source = next((s for s in self.sources if s.name == name), None)
                frame = source.snapshot().frame if source is not None else None
                if recorder.feed(frame, person_present=False):
                    self._finish_person_recorder(name, recorder)
            return

        now = time.monotonic()
        for source in self.sources:
            cam = self._camera_by_name.get(source.name)
            if cam is None or not cam.detect_people:
                continue
            snap = source.snapshot()
            frame = snap.frame
            if frame is None:
                continue
            boxes = self._detection.process(
                source.name,
                frame,
                detect_people=True,
                detect_objects=False,
            )
            people = [b for b in boxes if b.label == "person"]
            present = bool(people)
            annotated = frame
            if people:
                annotated = frame.copy()
                draw_boxes(annotated, people)
            self._handle_person_presence(source, frame, annotated, present, now)

    def _handle_person_presence(
        self,
        source,
        frame: np.ndarray,
        annotated: np.ndarray,
        present: bool,
        now: float,
    ) -> None:
        name = source.name
        was = self._person_seen.get(name, False)
        recorder = self._person_recorders.get(name)

        if recorder is not None:
            if recorder.feed(frame, person_present=present):
                self._finish_person_recorder(name, recorder)
            self._person_seen[name] = present
            return

        if present and not was:
            cooldown = self._person_cooldown_until.get(name, 0.0)
            if now >= cooldown:
                self._start_person_recorder(source, frame, annotated)
        self._person_seen[name] = present

    def _start_person_recorder(self, source, frame: np.ndarray, annotated: np.ndarray) -> None:
        d = self.display
        pre_frames, fps = source.history.export_frames(float(d.person_pre_roll_seconds))
        try:
            recorder = PersonEventRecorder.start(
                camera=source.name,
                save_directory=self._save_dir(),
                pre_frames=pre_frames,
                snapshot_frame=annotated,
                fps=fps or float(d.fps) or 12.0,
                post_roll=float(d.person_post_roll_seconds),
                max_seconds=float(d.person_max_event_seconds),
                snapshot_format=d.snapshot_format,
            )
        except CaptureError as exc:
            self._flash_capture(f"Person event failed: {exc}")
            return
        recorder.feed(frame, person_present=True)
        self._person_recorders[source.name] = recorder
        self._flash_capture(f"Person event: {source.name}", seconds=2.5)
        print(f"Person event started → {recorder.event_dir}")

    def _finish_person_recorder(self, name: str, recorder: PersonEventRecorder) -> None:
        self._person_recorders.pop(name, None)
        self._person_cooldown_until[name] = time.monotonic() + 2.0
        if recorder.error:
            self._flash_capture(f"Person event error: {recorder.error}")
            print(f"Person event error ({name}): {recorder.error}")
            return
        clip = recorder.clip_path.name if recorder.clip_path else "no clip"
        self._flash_capture(f"Saved person event ({clip})")
        print(f"Person event finished → {recorder.event_dir}")

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
        self._apply_buffer_settings(persist=False)
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
        if self._encroach_owned_focus or self._zone_edit_name is not None:
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

    def _refresh_captures(self) -> None:
        self._captures = list_captures(self._save_dir())

    def _open_captures_browser(self) -> None:
        self._run_media_purge()
        self._refresh_captures()
        self._close_capture_view()
        self._menu_page = "captures"
        self._menu_index = 0
        self._reboot_notice = str(self._save_dir())

    def _selected_capture(self) -> CaptureItem | None:
        index = self._capture_view_index
        if index is None or index < 0 or index >= len(self._captures):
            return None
        return self._captures[index]

    def _close_capture_view(self) -> None:
        self._capture_view_index = None
        self._capture_preview = None

    def _open_capture_view(self, index: int) -> None:
        if index < 0 or index >= len(self._captures):
            self._reboot_notice = "File not found"
            return
        self._capture_view_index = index
        item = self._captures[index]
        self._capture_preview = load_capture_preview(item.path)
        self._menu_page = "captures_view"
        self._menu_index = 0
        if self._capture_preview is None:
            self._reboot_notice = f"Could not preview {item.name}"
        else:
            self._reboot_notice = item.label

    def _nudge_capture_view(self, step: int) -> None:
        if not self._captures:
            return
        if self._capture_view_index is None:
            self._open_capture_view(0 if step >= 0 else len(self._captures) - 1)
            return
        index = (self._capture_view_index + int(step)) % len(self._captures)
        self._open_capture_view(index)

    def _reveal_captures_folder(self) -> None:
        item = self._selected_capture()
        target = item.path if item is not None else self._save_dir()
        try:
            folder = reveal_in_file_manager(target)
        except CaptureError as exc:
            self._reboot_notice = str(exc)
            return
        self._reboot_notice = f"Opened {folder}"

    def _delete_selected_capture(self) -> None:
        item = self._selected_capture()
        if item is None:
            self._menu_page = "captures"
            return
        try:
            delete_capture(item.path)
        except CaptureError as exc:
            self._reboot_notice = str(exc)
            self._menu_page = "captures_view"
            return
        name = item.name
        self._refresh_captures()
        if self._captures:
            index = min(self._capture_view_index or 0, len(self._captures) - 1)
            self._open_capture_view(index)
            self._reboot_notice = f"Deleted {name}"
        else:
            self._close_capture_view()
            self._menu_page = "captures"
            self._menu_index = 0
            self._reboot_notice = f"Deleted {name}"
        print(f"Deleted capture {name}")

    def _delete_all_captures(self) -> None:
        paths = [item.path for item in self._captures if not item.locked]
        skipped = sum(1 for item in self._captures if item.locked)
        deleted = delete_captures(paths)
        self._refresh_captures()
        self._close_capture_view()
        self._menu_page = "captures"
        self._menu_index = 0
        notice = f"Deleted {deleted} file(s)"
        if skipped:
            notice += f" (kept {skipped} locked)"
        self._reboot_notice = notice
        print(notice)

    def _toggle_selected_capture_lock(self) -> None:
        item = self._selected_capture()
        if item is None:
            return
        try:
            set_capture_locked(item.path, not item.locked)
        except CaptureError as exc:
            self._reboot_notice = str(exc)
            return
        index = self._capture_view_index or 0
        self._refresh_captures()
        if index < len(self._captures):
            self._open_capture_view(index)
        state = "locked" if not item.locked else "unlocked"
        self._reboot_notice = f"Capture {state}"

    def _toggle_selected_event_lock(self) -> None:
        item = self._selected_event()
        if item is None:
            return
        try:
            set_event_locked(item.path, not item.locked)
        except CaptureError as exc:
            self._reboot_notice = str(exc)
            return
        index = self._event_view_index or 0
        self._refresh_events()
        if index < len(self._events):
            self._open_event_view(index)
        state = "locked" if not item.locked else "unlocked"
        self._reboot_notice = f"Event {state}"

    def _run_media_purge(self, *, force: bool = False) -> PurgeResult:
        d = self.display
        if not force and d.capture_retention_days <= 0 and d.capture_max_gb <= 0:
            return PurgeResult()
        result = purge_old_media(
            self._save_dir(),
            max_age_days=float(d.capture_retention_days),
            max_total_gb=float(d.capture_max_gb),
        )
        self._last_purge_at = time.monotonic()
        if result.deleted:
            self._refresh_captures()
            self._refresh_events()
            print(result.summary)
        return result

    def _maybe_purge_media(self) -> None:
        d = self.display
        if d.capture_retention_days <= 0 and d.capture_max_gb <= 0:
            return
        now = time.monotonic()
        # Run shortly after start, then about every 15 minutes.
        if self._last_purge_at > 0 and (now - self._last_purge_at) < 900:
            return
        if self._last_purge_at <= 0 and now < 30:
            # Allow streams to settle before first sweep.
            return
        result = self._run_media_purge()
        if result.deleted:
            self._flash_capture(result.summary, seconds=4.0)

    def _refresh_events(self) -> None:
        self._events = list_person_events(self._save_dir())

    def _open_events_browser(self) -> None:
        self._run_media_purge()
        self._refresh_events()
        self._close_event_view()
        self._stop_event_playback()
        self._menu_page = "events"
        self._menu_index = 0
        self._reboot_notice = "Person detection snapshots + clips"

    def _selected_event(self) -> PersonEventItem | None:
        index = self._event_view_index
        if index is None or index < 0 or index >= len(self._events):
            return None
        return self._events[index]

    def _close_event_view(self) -> None:
        self._event_view_index = None
        self._event_preview = None

    def _open_event_view(self, index: int) -> None:
        if index < 0 or index >= len(self._events):
            self._reboot_notice = "Event not found"
            return
        self._stop_event_playback()
        self._event_view_index = index
        item = self._events[index]
        self._event_preview = load_capture_preview(item.snapshot)
        self._menu_page = "events_view"
        self._menu_index = 1 if item.has_clip else 0
        self._reboot_notice = item.label

    def _nudge_event_view(self, step: int) -> None:
        if not self._events:
            return
        if self._event_view_index is None:
            self._open_event_view(0 if step >= 0 else len(self._events) - 1)
            return
        index = (self._event_view_index + int(step)) % len(self._events)
        self._open_event_view(index)

    def _start_event_playback(self) -> None:
        item = self._selected_event()
        if item is None or not item.has_clip or item.clip is None:
            self._reboot_notice = "No recording for this event yet"
            return
        self._stop_event_playback()
        cap = cv2.VideoCapture(str(item.clip))
        if not cap.isOpened():
            self._reboot_notice = f"Could not open {item.clip.name}"
            return
        self._event_playback = cap
        self._event_playback_label = item.label
        self._menu_page = "events_play"
        self._menu_index = 0
        self._reboot_notice = "Playing clip — Esc to stop"

    def _stop_event_playback(self) -> None:
        if self._event_playback is not None:
            self._event_playback.release()
        self._event_playback = None
        self._event_playback_label = ""

    def _delete_selected_event(self) -> None:
        item = self._selected_event()
        if item is None:
            self._menu_page = "events"
            return
        self._stop_event_playback()
        try:
            delete_person_event(item.path)
        except CaptureError as exc:
            self._reboot_notice = str(exc)
            self._menu_page = "events_view"
            return
        label = item.label
        self._refresh_events()
        if self._events:
            index = min(self._event_view_index or 0, len(self._events) - 1)
            self._open_event_view(index)
            self._reboot_notice = f"Deleted {label}"
        else:
            self._close_event_view()
            self._menu_page = "events"
            self._menu_index = 0
            self._reboot_notice = f"Deleted {label}"
        print(f"Deleted person event {label}")

    def _capture_label(self) -> str:
        if self.zoom_index is not None and 0 <= self.zoom_index < len(self.sources):
            return self.sources[self.zoom_index].name
        return "mosaic"

    def _current_capture_frame(self, *, include_weather: bool = True) -> np.ndarray | None:
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
        self._grid_x, self._grid_y = x_off, y_off
        reserved = None
        if include_weather and d.weather_enabled:
            reserved = self._resolve_weather_rect(width, height, x_off, y_off, grid_w, grid_h)
        shrink = None if (reserved is None or d.weather_overlay) else reserved
        tile_rects = compute_tile_rects(
            columns=d.columns,
            rows=d.rows,
            cell_w=cell_w,
            cell_h=cell_h,
            grid_x=x_off,
            grid_y=y_off,
            reserved=shrink,
        )
        for index in range(d.tile_count):
            tx, ty, tw, th = tile_rects[index]
            if tw < 2 or th < 2:
                continue
            if index < len(self.sources):
                snap = self.sources[index].snapshot()
                tile = self._render_cell(
                    snap,
                    self.sources[index].name,
                    tw,
                    th,
                    overlay=True,
                )
            else:
                tile = placeholder(tw, th, "Empty", "No camera assigned")
            canvas[ty : ty + th, tx : tx + tw] = tile
        if reserved is not None:
            self._paint_weather_widget(canvas, reserved, editing=False)
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
        preview_pages = {
            "captures_view",
            "captures_delete",
            "events_view",
            "events_delete",
            "events_play",
        }
        if self._menu_page not in preview_pages:
            canvas[:] = (canvas.astype(np.float32) * 0.38).astype(np.uint8)
        items = self._menu_items()
        wide = self._menu_page in {
            "video",
            "decode_status",
            "capture",
            "detection",
            "detection_cams",
            "weather",
            *_CAMERA_MENU_PAGES,
            *_CAPTURE_BROWSER_PAGES,
            *_EVENT_BROWSER_PAGES,
        }
        card_w, row_h, pad = (720 if self._menu_page == "decode_status" else 600 if wide else 480), 46, 20
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
            "decode_status": "Decode status",
            "capture": "Capture",
            "detection": "Detection",
            "detection_cams": "Cameras for detection",
            "weather": "Weather HUD",
            "cameras": "Cameras",
            "cameras_arrange": "Arrange tiles",
            "cameras_toggle": "Show / hide",
            "cameras_add": "Add camera",
            "cameras_remove": "Remove camera",
            "captures": "Saved captures",
            "captures_view": "View capture",
            "captures_delete": "Delete capture",
            "captures_delete_all": "Delete all captures",
            "events": "Person events",
            "events_view": "Person event",
            "events_delete": "Delete event",
            "events_play": "Playing recording",
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
        elif self._menu_page == "decode_status":
            footer = "Per-camera decode path    Esc back"
        elif self._menu_page == "weather":
            footer = "Place on layout…    ← → adjust    Esc back"
        elif self._menu_page == "capture":
            footer = "s snapshot  c clip    ← → length    Esc back"
        elif self._menu_page == "events_play":
            footer = "Esc stop playback"
        elif self._menu_page == "events_view":
            footer = "Enter play recording    ← → browse    Esc back"
        elif self._menu_page == "events_delete":
            footer = "Enter confirm    Esc cancel"
        elif self._menu_page == "events":
            footer = "Enter open snapshot    Esc back"
        elif self._menu_page == "captures_view":
            footer = "← → browse files    Enter select    Esc back"
        elif self._menu_page in {"captures_delete", "captures_delete_all"}:
            footer = "Enter confirm    Esc cancel"
        elif self._menu_page == "captures":
            footer = "Enter open    ← → move    Esc back"
        elif self._menu_page == "cameras_arrange":
            footer = "← → move in grid order    Esc back"
        elif self._menu_page in _CAMERA_MENU_PAGES or self._menu_page in {
            "detection",
            "detection_cams",
            "capture",
            "weather",
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
        if self._weather_place_mode:
            self._on_weather_place_mouse(event, x, y)
            return
        if self._line_edit_name is not None or self._zone_edit_name is not None:
            self._on_zone_edit_mouse(event, x, y)
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
        # Clicks on the weather widget should not focus a camera underneath.
        if self._weather_rect is not None and self.display.weather_enabled:
            px, py = self._event_to_pixel(x, y)
            rx, ry, rw, rh = self._weather_rect.as_tuple
            if rx <= px <= rx + rw and ry <= py <= ry + rh:
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

    def _on_weather_place_mouse(self, event: int, x: int, y: int) -> None:
        px, py = self._event_to_pixel(x, y)
        rect = self._weather_rect
        down = getattr(cv2, "EVENT_LBUTTONDOWN", 1)
        up = getattr(cv2, "EVENT_LBUTTONUP", 4)
        move = getattr(cv2, "EVENT_MOUSEMOVE", 0)
        if event == down and rect is not None:
            rx, ry, rw, rh = rect.as_tuple
            if rx <= px <= rx + rw and ry <= py <= ry + rh:
                self._weather_drag_offset = (px - rx, py - ry)
            else:
                # Click outside: jump widget top-left to cursor.
                self._weather_drag_offset = (0, 0)
                self._set_weather_norm_from_pixel(px, py)
            return
        if event == move and self._weather_drag_offset is not None:
            self._set_weather_norm_from_pixel(px, py)
            return
        if event == up:
            if self._weather_drag_offset is not None:
                self._set_weather_norm_from_pixel(px, py)
            self._weather_drag_offset = None
            return

    def _on_zone_edit_mouse(self, event: int, x: int, y: int) -> None:
        if event != cv2.EVENT_LBUTTONUP:
            return
        name = self._zone_edit_name or self._line_edit_name
        if name is None:
            return
        source = next((s for s in self.sources if s.name == name), None)
        if source is None:
            self._cancel_zone_edit()
            return
        snap = source.snapshot()
        if snap.frame is None:
            self._flash_capture("No frame — try again")
            return
        fh, fw = snap.frame.shape[:2]
        px, py = self._event_to_pixel(x, y)
        mapped = map_tile_click_to_frame_norm(
            px,
            py,
            tile_w=self._view_w,
            tile_h=self._view_h,
            frame_w=fw,
            frame_h=fh,
            mode=self.display.scale_mode if self.view_zoom <= 1.001 else "fill",
        )
        if mapped is None:
            self._flash_capture("Click on the video (not letterbox)")
            return
        mode = self._zone_edit_mode or "line"
        self._zone_edit_points.append(mapped)
        if mode == "line":
            if len(self._zone_edit_points) == 1:
                self._flash_capture("Click second point", seconds=3.0)
                return
            self._finish_zone_edit()
            return
        n = len(self._zone_edit_points)
        self._flash_capture(
            f"Point {n} — click more or Enter to finish (≥3)",
            seconds=2.5,
        )

    def _on_line_edit_mouse(self, event: int, x: int, y: int) -> None:
        self._on_zone_edit_mouse(event, x, y)

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
    *,
    encroach: bool = False,
    hud_opacity: float = 0.70,
) -> None:
    h, w = tile.shape[:2]
    bar_h = max(34, int(h * 0.08))
    fade = max(10, bar_h // 3)
    alpha = max(0.12, min(1.0, float(hud_opacity)))
    shade_bottom_bar(tile, bar_h, alpha=alpha, fade=fade)
    color = (40, 70, 255) if encroach else STATUS_COLOR.get(snap.status, (180, 180, 180))
    cy = h - bar_h // 2
    name_size = max(14, min(22, int(h * 0.038)))
    meta_size = max(12, min(18, int(h * 0.032)))
    draw_dot(tile, (16 + name_size // 8, cy), max(5.0, name_size * 0.32), color)
    draw_text(tile, name, (28 + name_size // 4, h - 8), size=name_size, valign="bottom")
    status = snap.status.upper()
    if encroach:
        status = "ENCROACH"
    elif snap.rewinding:
        status = f"REWIND -{snap.behind:.1f}s"
    elif snap.detail and snap.status not in ("live", "demo"):
        status = f"{status}  {snap.detail}"
    if fps_text and not snap.rewinding and not encroach:
        status = f"{status}   {fps_text}"
    elif fps_text and encroach:
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
