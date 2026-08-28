"""Load and validate the YAML configuration file."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse, urlunparse

import yaml

from security_monitor.encroachment import EncroachZone
from security_monitor.home_assistant import (
    HADoorMapping,
    HALightControl,
    door_mapping_to_dict,
    light_control_to_dict,
    normalize_ha_url,
    parse_door_mappings,
    parse_light_controls,
)
from security_monitor.webhooks import (
    OutgoingWebhook,
    WebhookMapping,
    outgoing_webhook_to_dict,
    parse_incoming_webhooks,
    parse_outgoing_webhooks,
    webhook_mapping_to_dict,
)

VALID_SCALE_MODES = ("fit", "fill", "stretch")
VALID_TRANSPORTS = ("tcp", "udp", "auto")
VALID_CAMERA_KINDS = ("ubiquiti", "reolink", "amcrest", "dahua")
VALID_SCREEN_ROTATIONS = ("none", "normal", "left", "right", "inverted")
VALID_CAMERA_ROTATIONS = (0, 90, 180, 270)
VALID_DECODE_MODES = ("auto", "cpu", "gpu")
VALID_HWACCELS = ("auto", "none", "cuda", "qsv", "vaapi", "d3d11va", "videotoolbox")
VALID_ENCROACH_SIDES = ("positive", "negative")
VALID_WEATHER_SLOTS = (
    "bottom_left",
    "bottom_right",
    "between_h",
    "between_v",
    "custom",
)
VALID_WEATHER_UNITS = ("f", "c")
VALID_POWER_MODES = ("auto", "on", "off")
URL_SCHEMES = ("rtsp", "rtp", "http", "https", "file", "rtmp")

# Esc → Cameras → Layout presets (columns, rows).
LAYOUT_PRESETS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
    (2, 3),
    (3, 2),
    (3, 3),
    (4, 2),
    (3, 4),
    (4, 3),
    (4, 4),
)
# 0 = cycle focus off; otherwise seconds between auto focus advances.
CYCLE_FOCUS_CHOICES: tuple[float, ...] = (0.0, 5.0, 10.0, 30.0, 60.0)

_CREDENTIALS_RE = re.compile(r"://([^:/?#]+):([^@/?#]+)@")


class ConfigError(ValueError):
    """Raised when the config file is missing or invalid."""


@dataclass
class CameraConfig:
    name: str
    url: str | None = None
    device: int | None = None
    enabled: bool = True
    transport: str | None = None
    username: str | None = None
    password: str | None = None
    kind: str | None = None
    reboot: bool = True
    ssh_port: int = 22
    http_port: int = 80
    rotate: int = 0  # clockwise degrees: 0 | 90 | 180 | 270
    detect_people: bool = False
    detect_objects: bool = False
    # Encroachment: tripwire / polygon ROIs (normalized coords).
    detect_encroachment: bool = False
    encroach_zones: list[EncroachZone] = field(default_factory=list)
    # Legacy single tripwire (used when encroach_zones is empty).
    encroach_line: tuple[float, float, float, float] | None = None
    encroach_side: str = "positive"  # positive | negative
    # Optional 1:1 Home Assistant door → this camera shortcut.
    ha_door_entity: str = ""
    ha_door_label: str = ""

    @property
    def is_device(self) -> bool:
        return self.device is not None

    @property
    def host(self) -> str | None:
        if not self.url:
            return None
        return urlparse(self.url).hostname

    def capture_source(self) -> str | int:
        if self.device is not None:
            return self.device
        if not self.url:
            raise ConfigError(f"Camera {self.name!r} has neither url nor device")
        return inject_credentials(self.url, self.username, self.password)

    def redacted_source(self) -> str:
        if self.device is not None:
            return f"device:{self.device}"
        return redact_url(self.url or "")


@dataclass
class DisplayConfig:
    columns: int = 2
    rows: int = 2
    cell_width: int = 640
    cell_height: int = 360
    scale_mode: str = "fit"
    window_title: str = "Security Monitor"
    fullscreen: bool = False
    show_labels: bool = True
    show_fps: bool = True
    fps: int = 25
    reconnect_seconds: float = 5.0
    default_transport: str = "tcp"
    open_timeout_ms: int = 8000
    read_timeout_ms: int = 5000
    # Linux X11/XWayland: rotate the active output before opening the window.
    # Use left/right when a panel is mounted portrait but the desktop is landscape
    # (classic "squished" / sideways look).
    screen_rotate: str = "none"  # none | normal | left | right | inverted
    screen_output: str = "auto"  # auto | HDMI-1 | DP-1 | ...
    # Smooth playback: show frames delayed by smooth_buffer_seconds to absorb jitter.
    smooth_buffer: bool = False
    smooth_buffer_seconds: float = 1.0
    # Rolling history for quick rewind (comma/period or menu).
    rewind_buffer: bool = False
    rewind_buffer_seconds: float = 30.0
    # Snapshots / clips
    save_directory: str = ""  # empty → ~/security-monitor/captures
    snapshot_format: str = "jpg"  # jpg | png
    clip_seconds: float = 15.0
    # Detection masters (still requires per-camera opt-in).
    people_detection: bool = False
    object_detection: bool = False
    # Encroachment: person in ROI → highlight (+ optional autofocus / alarm).
    encroachment_detection: bool = False
    encroachment_autofocus: bool = False
    encroachment_alarm: bool = True  # stronger on-screen alarm while in zone
    encroachment_alarm_sound: bool = True  # beep on entry (+ periodic while active)
    # Auto person events: snapshot + pre/during/post clip (Esc → Detection).
    auto_person_capture: bool = False
    person_pre_roll_seconds: float = 5.0
    person_post_roll_seconds: float = 5.0
    person_max_event_seconds: float = 120.0
    # Auto-erase unlocked captures/events (0 = off). Locked items are kept.
    capture_retention_days: float = 14.0
    capture_max_gb: float = 20.0
    # Stream decode: auto | cpu | gpu — hwaccel: auto | none | cuda | qsv | vaapi | d3d11va
    decode_mode: str = "auto"
    hwaccel: str = "auto"
    hwaccel_device: str = ""  # e.g. /dev/dri/renderD128
    # Auto-advance focused camera (0 / False = off). Menu: Esc → Cameras.
    cycle_focus: bool = False
    cycle_focus_seconds: float = 10.0
    # Local weather HUD widget (Esc → Weather).
    weather_enabled: bool = False
    weather_slot: str = "bottom_right"  # bottom_left|bottom_right|between_h|between_v|custom
    weather_x: float = 0.0  # custom top-left or fine-tune offset (grid-normalized)
    weather_y: float = 0.0
    weather_w: float = 0.24
    weather_h: float = 0.20
    weather_units: str = "f"  # f | c
    weather_show_temp: bool = True
    weather_show_conditions: bool = True
    weather_show_storm: bool = True
    weather_show_lightning: bool = True
    weather_show_forecast: bool = False
    # Scan radius for nearby thunderstorms (miles). Menu: 5–100.
    weather_lightning_miles: float = 25.0
    # Weather widget blend (1.0 = solid). Useful when overlaying a camera feed.
    weather_opacity: float = 0.85
    # When true, cameras keep full tiles and the widget paints on top (see-through).
    weather_overlay: bool = False
    # Camera name/status strip opacity on each tile.
    hud_opacity: float = 0.70
    weather_latitude: float | None = None
    weather_longitude: float | None = None
    weather_place: str = ""
    weather_refresh_seconds: float = 300.0
    # Local Home Assistant door sensors (Esc → Home Assistant).
    ha_enabled: bool = False
    ha_url: str = "http://homeassistant.local:8123"
    ha_token: str = ""
    ha_poll_seconds: float = 2.0
    ha_show_hud: bool = True
    ha_highlight: bool = True
    ha_autofocus: bool = True
    ha_alarm_sound: bool = True
    ha_hold_seconds: float = 20.0
    ha_popup_seconds: float = 5.0
    ha_panel_enabled: bool = True
    ha_doors: list[HADoorMapping] = field(default_factory=list)
    ha_lights: list[HALightControl] = field(default_factory=list)
    # Incoming HTTP webhooks (same popup/HUD/peek/sound path as HA sensors).
    webhook_enabled: bool = False
    webhook_listen_host: str = "0.0.0.0"
    webhook_listen_port: int = 8765
    webhook_secret: str = ""
    webhook_pulse_seconds: float = 8.0
    webhook_incoming: list[WebhookMapping] = field(default_factory=list)
    webhook_outgoing: list[OutgoingWebhook] = field(default_factory=list)
    # Low-power: skip extras (detection, weather, HA, rewind) when the UI FPS drops.
    power_mode: str = "auto"  # auto | on | off
    low_power_fps: float = 12.0
    # After an unclean exit, start with video + HUD only until the user recovers.
    safe_mode_on_crash: bool = True

    @property
    def tile_count(self) -> int:
        return self.columns * self.rows

    @property
    def canvas_size(self) -> tuple[int, int]:
        return self.columns * self.cell_width, self.rows * self.cell_height

    @property
    def cycle_focus_label(self) -> str:
        if not self.cycle_focus or self.cycle_focus_seconds <= 0:
            return "Off"
        return f"{self.cycle_focus_seconds:g}s"


@dataclass
class AppConfig:
    display: DisplayConfig
    cameras: list[CameraConfig] = field(default_factory=list)
    path: Path | None = None

    def visible_cameras(self) -> list[CameraConfig]:
        enabled = [cam for cam in self.cameras if cam.enabled]
        return enabled[: self.display.tile_count]


def redact_url(url: str) -> str:
    """Strip userinfo from a URL so logs never print passwords."""
    return _CREDENTIALS_RE.sub("://***:***@", url)


def inject_credentials(url: str, username: str | None, password: str | None) -> str:
    if not username:
        return url
    parsed = urlparse(url)
    if parsed.username:
        return url
    user = quote(username, safe="")
    creds = user if password is None else f"{user}:{quote(password, safe='')}"
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    netloc = f"{creds}@{host}"
    return urlunparse(
        (parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment)
    )


def xdg_user_config_path() -> Path:
    """Per-user config.yaml location (Linux/macOS XDG, or %APPDATA% on Windows)."""
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if not appdata:
            raise ConfigError("APPDATA is not set; cannot resolve the user config path")
        return Path(appdata) / "security-monitor" / "config.yaml"
    xdg = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    return Path(xdg) / "security-monitor" / "config.yaml"


def default_config_paths() -> list[Path]:
    paths = [Path.cwd() / "config.yaml"]
    try:
        paths.append(xdg_user_config_path())
    except ConfigError:
        pass
    env_path = os.environ.get("SECURITY_MONITOR_CONFIG")
    if env_path:
        paths.insert(0, Path(env_path))
    return paths


def resolve_config_path(explicit: str | Path | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise ConfigError(f"Config file not found: {path}")
        return path
    for candidate in default_config_paths():
        if candidate.is_file():
            return candidate
    searched = "\n  ".join(str(p) for p in default_config_paths())
    raise ConfigError(
        "No config.yaml found. Copy config.example.yaml to config.yaml "
        f"or pass --config.\nLooked in:\n  {searched}"
    )


def load_config(path: str | Path | None = None) -> AppConfig:
    resolved = resolve_config_path(path)
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {resolved}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("Config root must be a mapping")
    return parse_config(raw, resolved)


def parse_config(raw: dict[str, Any], path: Path | None = None) -> AppConfig:
    display = _parse_display(raw.get("display") or {})
    cameras = [_parse_camera(item, index) for index, item in enumerate(_as_list(raw.get("cameras")))]
    seen: set[str] = set()
    for cam in cameras:
        key = cam.name.strip().lower()
        if key in seen:
            raise ConfigError(f"Duplicate camera name: {cam.name!r}")
        seen.add(key)
    enabled = [cam for cam in cameras if cam.enabled]
    if len(enabled) > display.tile_count:
        extra = len(enabled) - display.tile_count
        print(
            f"Warning: {extra} enabled camera(s) exceed the "
            f"{display.columns}x{display.rows} grid and will be ignored.",
            file=sys.stderr,
        )
    return AppConfig(display=display, cameras=cameras, path=path)


def example_config_text() -> str:
    return files("security_monitor").joinpath("data/config.example.yaml").read_text(encoding="utf-8")


def camera_to_dict(cam: CameraConfig) -> dict[str, Any]:
    """Serialize a camera for YAML (only meaningful fields)."""
    item: dict[str, Any] = {"name": cam.name}
    if cam.device is not None:
        item["device"] = int(cam.device)
    if cam.url:
        item["url"] = cam.url
    item["enabled"] = bool(cam.enabled)
    if cam.transport:
        item["transport"] = cam.transport
    if cam.username:
        item["username"] = cam.username
    if cam.password:
        item["password"] = cam.password
    if cam.kind:
        item["type"] = cam.kind
    if not cam.reboot:
        item["reboot"] = False
    if cam.ssh_port != 22:
        item["ssh_port"] = int(cam.ssh_port)
    if cam.http_port != 80:
        item["http_port"] = int(cam.http_port)
    if cam.rotate:
        item["rotate"] = int(cam.rotate)
    if cam.detect_people:
        item["detect_people"] = True
    if cam.detect_objects:
        item["detect_objects"] = True
    if cam.detect_encroachment:
        item["detect_encroachment"] = True
    if cam.encroach_zones:
        from security_monitor.encroachment import zone_to_dict

        zones = []
        for zone in cam.encroach_zones:
            if len(zone.points) < 2:
                continue
            item_z = zone_to_dict(zone)
            item_z["points"] = _flow(
                [_flow([float(x), float(y)]) for x, y in zone.points]
            )
            zones.append(item_z)
        if zones:
            item["encroach_zones"] = zones
    if cam.encroach_line is not None:
        item["encroach_line"] = _flow([float(v) for v in cam.encroach_line])
    if cam.encroach_side and cam.encroach_side != "positive":
        item["encroach_side"] = str(cam.encroach_side)
    if cam.ha_door_entity:
        item["ha_door_entity"] = str(cam.ha_door_entity)
    if cam.ha_door_label:
        item["ha_door_label"] = str(cam.ha_door_label)
    return item


class _FlowList(list):
    """YAML list dumped as ``[a, b]`` so nested zone points round-trip."""


def _flow(values: list) -> _FlowList:
    seq = _FlowList(values)
    if not getattr(_FlowList, "_registered", False):
        def _represent(dumper, data: _FlowList):
            return dumper.represent_sequence(
                "tag:yaml.org,2002:seq", list(data), flow_style=True
            )

        yaml.add_representer(_FlowList, _represent, Dumper=yaml.SafeDumper)
        _FlowList._registered = True  # type: ignore[attr-defined]
    return seq


def unique_camera_name(cameras: list[CameraConfig], base: str) -> str:
    base = (base or "Camera").strip() or "Camera"
    taken = {cam.name.strip().lower() for cam in cameras}
    if base.lower() not in taken:
        return base
    for n in range(2, 10000):
        candidate = f"{base} {n}"
        if candidate.lower() not in taken:
            return candidate
    return f"{base} copy"


def layout_presets_for_count(enabled_count: int, *, max_extra: int = 2) -> list[tuple[int, int]]:
    """Grid sizes that fit ``enabled_count`` cameras without a huge empty grid."""
    n = max(1, int(enabled_count))
    extra = max(0, int(max_extra))
    max_slots = n + extra
    fitting = [
        preset
        for preset in LAYOUT_PRESETS
        if preset[0] * preset[1] >= n and preset[0] * preset[1] <= max_slots
    ]
    if not fitting:
        fitting = [preset for preset in LAYOUT_PRESETS if preset[0] * preset[1] >= n]
    if not fitting:
        fitting = [(n, 1)]
    return fitting


def next_layout_preset(
    columns: int,
    rows: int,
    step: int = 1,
    *,
    min_tiles: int | None = None,
    max_extra: int = 2,
) -> tuple[int, int]:
    current = (int(columns), int(rows))
    if min_tiles is None:
        presets = list(LAYOUT_PRESETS)
        if current not in presets:
            presets.append(current)
            presets.sort(key=lambda p: (p[0] * p[1], p[0], p[1]))
        index = presets.index(current)
        return presets[(index + int(step)) % len(presets)]

    presets = layout_presets_for_count(min_tiles, max_extra=max_extra)
    if current not in presets:
        return presets[0] if int(step) >= 0 else presets[-1]
    index = presets.index(current)
    return presets[(index + int(step)) % len(presets)]


def move_camera(cameras: list[CameraConfig], index: int, direction: int) -> int:
    """Swap camera at index with neighbor (direction -1 earlier, +1 later)."""
    if index < 0 or index >= len(cameras):
        return index
    target = index + (1 if direction > 0 else -1)
    if target < 0 or target >= len(cameras):
        return index
    cameras[index], cameras[target] = cameras[target], cameras[index]
    return target


def ensure_layout_fits(display: DisplayConfig, enabled_count: int) -> bool:
    """Grow columns/rows to the next preset that fits enabled_count. Return True if changed."""
    if enabled_count <= display.tile_count:
        return False
    for cols, rows in LAYOUT_PRESETS:
        if cols * rows >= enabled_count:
            display.columns = cols
            display.rows = rows
            return True
    # Fall back to a wide strip.
    display.columns = max(1, enabled_count)
    display.rows = 1
    return True


def save_display_settings(config: AppConfig) -> Path | None:
    """
    Persist display + camera list settings back into config.yaml.

    Writes layout, buffers, capture, detection, focus cycling, and the full
    cameras list (order, enabled, urls, …). Returns the path written, or None
    if there is no config file (e.g. pure demo mode).
    """
    path = config.path
    if path is None:
        return None
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    display = raw.get("display")
    if not isinstance(display, dict):
        display = {}
        raw["display"] = display
    d = config.display
    display["columns"] = int(d.columns)
    display["rows"] = int(d.rows)
    display["smooth_buffer"] = bool(d.smooth_buffer)
    display["smooth_buffer_seconds"] = float(d.smooth_buffer_seconds)
    display["rewind_buffer"] = bool(d.rewind_buffer)
    display["rewind_buffer_seconds"] = float(d.rewind_buffer_seconds)
    display["save_directory"] = str(d.save_directory or "")
    display["snapshot_format"] = str(d.snapshot_format)
    display["clip_seconds"] = float(d.clip_seconds)
    display["people_detection"] = bool(d.people_detection)
    display["object_detection"] = bool(d.object_detection)
    display["encroachment_detection"] = bool(d.encroachment_detection)
    display["encroachment_autofocus"] = bool(d.encroachment_autofocus)
    display["encroachment_alarm"] = bool(d.encroachment_alarm)
    display["encroachment_alarm_sound"] = bool(d.encroachment_alarm_sound)
    display["auto_person_capture"] = bool(d.auto_person_capture)
    display["person_pre_roll_seconds"] = float(d.person_pre_roll_seconds)
    display["person_post_roll_seconds"] = float(d.person_post_roll_seconds)
    display["person_max_event_seconds"] = float(d.person_max_event_seconds)
    display["capture_retention_days"] = float(d.capture_retention_days)
    display["capture_max_gb"] = float(d.capture_max_gb)
    display["decode_mode"] = str(d.decode_mode)
    display["hwaccel"] = str(d.hwaccel)
    display["hwaccel_device"] = str(d.hwaccel_device or "")
    display["cycle_focus"] = bool(d.cycle_focus)
    display["cycle_focus_seconds"] = float(d.cycle_focus_seconds)
    display["weather_enabled"] = bool(d.weather_enabled)
    display["weather_slot"] = str(d.weather_slot)
    display["weather_x"] = float(d.weather_x)
    display["weather_y"] = float(d.weather_y)
    display["weather_w"] = float(d.weather_w)
    display["weather_h"] = float(d.weather_h)
    display["weather_units"] = str(d.weather_units)
    display["weather_show_temp"] = bool(d.weather_show_temp)
    display["weather_show_conditions"] = bool(d.weather_show_conditions)
    display["weather_show_storm"] = bool(d.weather_show_storm)
    display["weather_show_lightning"] = bool(d.weather_show_lightning)
    display["weather_show_forecast"] = bool(d.weather_show_forecast)
    display["weather_lightning_miles"] = float(d.weather_lightning_miles)
    display["weather_opacity"] = float(d.weather_opacity)
    display["weather_overlay"] = bool(d.weather_overlay)
    display["hud_opacity"] = float(d.hud_opacity)
    if d.weather_latitude is not None:
        display["weather_latitude"] = float(d.weather_latitude)
    if d.weather_longitude is not None:
        display["weather_longitude"] = float(d.weather_longitude)
    display["weather_place"] = str(d.weather_place or "")
    display["weather_refresh_seconds"] = float(d.weather_refresh_seconds)
    display["ha_enabled"] = bool(d.ha_enabled)
    display["ha_url"] = str(d.ha_url or "")
    display["ha_token"] = str(d.ha_token or "")
    display["ha_poll_seconds"] = float(d.ha_poll_seconds)
    display["ha_show_hud"] = bool(d.ha_show_hud)
    display["ha_highlight"] = bool(d.ha_highlight)
    display["ha_autofocus"] = bool(d.ha_autofocus)
    display["ha_alarm_sound"] = bool(d.ha_alarm_sound)
    display["ha_hold_seconds"] = float(d.ha_hold_seconds)
    display["ha_popup_seconds"] = float(d.ha_popup_seconds)
    display["ha_panel_enabled"] = bool(d.ha_panel_enabled)
    display["ha_doors"] = [door_mapping_to_dict(door) for door in d.ha_doors]
    display["ha_lights"] = [light_control_to_dict(light) for light in d.ha_lights]
    display["webhook_enabled"] = bool(d.webhook_enabled)
    display["webhook_listen_host"] = str(d.webhook_listen_host or "0.0.0.0")
    display["webhook_listen_port"] = int(d.webhook_listen_port)
    display["webhook_secret"] = str(d.webhook_secret or "")
    display["webhook_pulse_seconds"] = float(d.webhook_pulse_seconds)
    display["webhook_incoming"] = [webhook_mapping_to_dict(item) for item in d.webhook_incoming]
    display["webhook_outgoing"] = [outgoing_webhook_to_dict(item) for item in d.webhook_outgoing]
    display["power_mode"] = str(d.power_mode)
    display["low_power_fps"] = float(d.low_power_fps)
    display["safe_mode_on_crash"] = bool(d.safe_mode_on_crash)

    raw["cameras"] = [camera_to_dict(cam) for cam in config.cameras]

    path.write_text(
        yaml.safe_dump(raw, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return path


def demo_config(columns: int = 2, rows: int = 2) -> AppConfig:
    display = DisplayConfig(columns=columns, rows=rows, show_fps=True)
    cameras = [
        CameraConfig(name=f"Demo Cam {i + 1}", url=f"demo://{i}")
        for i in range(display.tile_count)
    ]
    return AppConfig(display=display, cameras=cameras)


def _parse_display(raw: Any) -> DisplayConfig:
    if not isinstance(raw, dict):
        raise ConfigError("display: must be a mapping")
    data = DisplayConfig()
    data.columns = _positive_int(raw, "columns", data.columns)
    data.rows = _positive_int(raw, "rows", data.rows)
    data.cell_width = _positive_int(raw, "cell_width", data.cell_width, minimum=160)
    data.cell_height = _positive_int(raw, "cell_height", data.cell_height, minimum=90)
    data.scale_mode = str(raw.get("scale_mode", data.scale_mode)).lower()
    if data.scale_mode not in VALID_SCALE_MODES:
        raise ConfigError(f"display.scale_mode must be one of {VALID_SCALE_MODES}")
    data.window_title = str(raw.get("window_title", data.window_title))
    data.fullscreen = _bool(raw, "fullscreen", data.fullscreen)
    data.show_labels = _bool(raw, "show_labels", data.show_labels)
    data.show_fps = _bool(raw, "show_fps", data.show_fps)
    data.fps = _positive_int(raw, "fps", data.fps, minimum=1, maximum=60)
    data.reconnect_seconds = _positive_float(raw, "reconnect_seconds", data.reconnect_seconds)
    data.default_transport = str(raw.get("default_transport", data.default_transport)).lower()
    if data.default_transport not in VALID_TRANSPORTS:
        raise ConfigError(f"display.default_transport must be one of {VALID_TRANSPORTS}")
    data.open_timeout_ms = _positive_int(raw, "open_timeout_ms", data.open_timeout_ms, minimum=500)
    data.read_timeout_ms = _positive_int(raw, "read_timeout_ms", data.read_timeout_ms, minimum=500)
    data.screen_rotate = str(raw.get("screen_rotate", data.screen_rotate)).strip().lower()
    if data.screen_rotate not in VALID_SCREEN_ROTATIONS:
        raise ConfigError(
            f"display.screen_rotate must be one of {VALID_SCREEN_ROTATIONS}"
        )
    data.screen_output = str(raw.get("screen_output", data.screen_output)).strip() or "auto"
    data.smooth_buffer = _bool(raw, "smooth_buffer", data.smooth_buffer)
    data.smooth_buffer_seconds = _positive_float(
        raw, "smooth_buffer_seconds", data.smooth_buffer_seconds
    )
    if data.smooth_buffer_seconds > 10:
        raise ConfigError("display.smooth_buffer_seconds must be <= 10")
    data.rewind_buffer = _bool(raw, "rewind_buffer", data.rewind_buffer)
    data.rewind_buffer_seconds = _positive_float(
        raw, "rewind_buffer_seconds", data.rewind_buffer_seconds
    )
    if data.rewind_buffer_seconds > 300:
        raise ConfigError("display.rewind_buffer_seconds must be <= 300")
    data.save_directory = str(raw.get("save_directory", data.save_directory) or "").strip()
    data.snapshot_format = str(raw.get("snapshot_format", data.snapshot_format)).strip().lower()
    if data.snapshot_format == "jpeg":
        data.snapshot_format = "jpg"
    if data.snapshot_format not in {"jpg", "png"}:
        raise ConfigError("display.snapshot_format must be jpg or png")
    data.clip_seconds = _positive_float(raw, "clip_seconds", data.clip_seconds)
    if data.clip_seconds > 300:
        raise ConfigError("display.clip_seconds must be <= 300")
    data.people_detection = _bool(raw, "people_detection", data.people_detection)
    data.object_detection = _bool(raw, "object_detection", data.object_detection)
    data.encroachment_detection = _bool(
        raw, "encroachment_detection", data.encroachment_detection
    )
    data.encroachment_autofocus = _bool(
        raw, "encroachment_autofocus", data.encroachment_autofocus
    )
    data.encroachment_alarm = _bool(raw, "encroachment_alarm", data.encroachment_alarm)
    data.encroachment_alarm_sound = _bool(
        raw, "encroachment_alarm_sound", data.encroachment_alarm_sound
    )
    data.auto_person_capture = _bool(raw, "auto_person_capture", data.auto_person_capture)
    data.person_pre_roll_seconds = _positive_float(
        raw, "person_pre_roll_seconds", data.person_pre_roll_seconds
    )
    if data.person_pre_roll_seconds > 60:
        raise ConfigError("display.person_pre_roll_seconds must be <= 60")
    data.person_post_roll_seconds = _positive_float(
        raw, "person_post_roll_seconds", data.person_post_roll_seconds
    )
    if data.person_post_roll_seconds > 60:
        raise ConfigError("display.person_post_roll_seconds must be <= 60")
    data.person_max_event_seconds = _positive_float(
        raw, "person_max_event_seconds", data.person_max_event_seconds
    )
    if data.person_max_event_seconds > 600:
        raise ConfigError("display.person_max_event_seconds must be <= 600")
    data.capture_retention_days = float(
        raw.get("capture_retention_days", data.capture_retention_days) or 0
    )
    if data.capture_retention_days < 0 or data.capture_retention_days > 3650:
        raise ConfigError("display.capture_retention_days must be between 0 and 3650")
    data.capture_max_gb = float(raw.get("capture_max_gb", data.capture_max_gb) or 0)
    if data.capture_max_gb < 0 or data.capture_max_gb > 10_000:
        raise ConfigError("display.capture_max_gb must be between 0 and 10000")
    data.decode_mode = str(raw.get("decode_mode", data.decode_mode)).strip().lower() or "auto"
    if data.decode_mode not in VALID_DECODE_MODES:
        raise ConfigError(f"display.decode_mode must be one of {VALID_DECODE_MODES}")
    data.hwaccel = str(raw.get("hwaccel", data.hwaccel)).strip().lower() or "auto"
    if data.hwaccel not in VALID_HWACCELS:
        raise ConfigError(f"display.hwaccel must be one of {VALID_HWACCELS}")
    data.hwaccel_device = str(raw.get("hwaccel_device", data.hwaccel_device) or "").strip()
    data.cycle_focus = _bool(raw, "cycle_focus", data.cycle_focus)
    data.cycle_focus_seconds = _positive_float(
        raw, "cycle_focus_seconds", data.cycle_focus_seconds
    )
    if data.cycle_focus_seconds > 600:
        raise ConfigError("display.cycle_focus_seconds must be <= 600")
    data.weather_enabled = _bool(raw, "weather_enabled", data.weather_enabled)
    data.weather_slot = str(raw.get("weather_slot", data.weather_slot)).strip().lower() or "bottom_right"
    if data.weather_slot not in VALID_WEATHER_SLOTS:
        raise ConfigError(f"display.weather_slot must be one of {VALID_WEATHER_SLOTS}")
    data.weather_x = float(raw.get("weather_x", data.weather_x) or 0.0)
    data.weather_y = float(raw.get("weather_y", data.weather_y) or 0.0)
    data.weather_w = float(raw.get("weather_w", data.weather_w) or data.weather_w)
    data.weather_h = float(raw.get("weather_h", data.weather_h) or data.weather_h)
    if not (0.08 <= data.weather_w <= 0.6):
        raise ConfigError("display.weather_w must be between 0.08 and 0.6")
    if not (0.08 <= data.weather_h <= 0.5):
        raise ConfigError("display.weather_h must be between 0.08 and 0.5")
    data.weather_units = str(raw.get("weather_units", data.weather_units)).strip().lower() or "f"
    if data.weather_units in {"fahrenheit", "f°", "°f"}:
        data.weather_units = "f"
    if data.weather_units in {"celsius", "c°", "°c"}:
        data.weather_units = "c"
    if data.weather_units not in VALID_WEATHER_UNITS:
        raise ConfigError(f"display.weather_units must be one of {VALID_WEATHER_UNITS}")
    data.weather_show_temp = _bool(raw, "weather_show_temp", data.weather_show_temp)
    data.weather_show_conditions = _bool(
        raw, "weather_show_conditions", data.weather_show_conditions
    )
    data.weather_show_storm = _bool(raw, "weather_show_storm", data.weather_show_storm)
    data.weather_show_lightning = _bool(
        raw, "weather_show_lightning", data.weather_show_lightning
    )
    data.weather_show_forecast = _bool(
        raw, "weather_show_forecast", data.weather_show_forecast
    )
    if "weather_lightning_miles" in raw and raw.get("weather_lightning_miles") is not None:
        try:
            data.weather_lightning_miles = float(raw.get("weather_lightning_miles"))
        except (TypeError, ValueError) as exc:
            raise ConfigError("display.weather_lightning_miles must be a number") from exc
    if "weather_lightning_km" in raw and raw.get("weather_lightning_km") is not None:
        try:
            data.weather_lightning_miles = float(raw.get("weather_lightning_km")) / 1.609344
        except (TypeError, ValueError) as exc:
            raise ConfigError("display.weather_lightning_km must be a number") from exc
    if not (1.0 <= data.weather_lightning_miles <= 200.0):
        raise ConfigError("display.weather_lightning_miles must be between 1 and 200")
    data.weather_opacity = float(raw.get("weather_opacity", data.weather_opacity) or data.weather_opacity)
    if data.weather_opacity > 1.0 and data.weather_opacity <= 100.0:
        data.weather_opacity = data.weather_opacity / 100.0
    if not (0.12 <= data.weather_opacity <= 1.0):
        raise ConfigError("display.weather_opacity must be between 0.12 and 1.0")
    data.weather_overlay = _bool(raw, "weather_overlay", data.weather_overlay)
    data.hud_opacity = float(raw.get("hud_opacity", data.hud_opacity) or data.hud_opacity)
    if data.hud_opacity > 1.0 and data.hud_opacity <= 100.0:
        data.hud_opacity = data.hud_opacity / 100.0
    if not (0.12 <= data.hud_opacity <= 1.0):
        raise ConfigError("display.hud_opacity must be between 0.12 and 1.0")
    if "weather_latitude" in raw and raw.get("weather_latitude") is not None:
        try:
            data.weather_latitude = float(raw.get("weather_latitude"))
        except (TypeError, ValueError) as exc:
            raise ConfigError("display.weather_latitude must be a number") from exc
    if "weather_longitude" in raw and raw.get("weather_longitude") is not None:
        try:
            data.weather_longitude = float(raw.get("weather_longitude"))
        except (TypeError, ValueError) as exc:
            raise ConfigError("display.weather_longitude must be a number") from exc
    data.weather_place = str(raw.get("weather_place", data.weather_place) or "").strip()
    data.weather_refresh_seconds = _positive_float(
        raw, "weather_refresh_seconds", data.weather_refresh_seconds
    )
    if data.weather_refresh_seconds < 60 or data.weather_refresh_seconds > 7200:
        raise ConfigError("display.weather_refresh_seconds must be between 60 and 7200")
    data.ha_enabled = _bool(raw, "ha_enabled", data.ha_enabled)
    data.ha_url = normalize_ha_url(str(raw.get("ha_url", data.ha_url) or data.ha_url))
    data.ha_token = str(raw.get("ha_token", data.ha_token) or "").strip()
    data.ha_poll_seconds = float(raw.get("ha_poll_seconds", data.ha_poll_seconds) or data.ha_poll_seconds)
    if not (0.5 <= data.ha_poll_seconds <= 60.0):
        raise ConfigError("display.ha_poll_seconds must be between 0.5 and 60")
    data.ha_show_hud = _bool(raw, "ha_show_hud", data.ha_show_hud)
    data.ha_highlight = _bool(raw, "ha_highlight", data.ha_highlight)
    data.ha_autofocus = _bool(raw, "ha_autofocus", data.ha_autofocus)
    data.ha_alarm_sound = _bool(raw, "ha_alarm_sound", data.ha_alarm_sound)
    data.ha_hold_seconds = float(raw.get("ha_hold_seconds", data.ha_hold_seconds) or data.ha_hold_seconds)
    if not (0.0 <= data.ha_hold_seconds <= 300.0):
        raise ConfigError("display.ha_hold_seconds must be between 0 and 300")
    data.ha_popup_seconds = float(
        raw.get("ha_popup_seconds", data.ha_popup_seconds) or data.ha_popup_seconds
    )
    if not (1.0 <= data.ha_popup_seconds <= 60.0):
        raise ConfigError("display.ha_popup_seconds must be between 1 and 60")
    data.ha_panel_enabled = _bool(raw, "ha_panel_enabled", data.ha_panel_enabled)
    try:
        data.ha_doors = parse_door_mappings(raw.get("ha_doors", data.ha_doors))
    except ValueError as exc:
        raise ConfigError(f"display.{exc}") from exc
    try:
        data.ha_lights = parse_light_controls(raw.get("ha_lights", data.ha_lights))
    except ValueError as exc:
        raise ConfigError(f"display.{exc}") from exc
    data.webhook_enabled = _bool(raw, "webhook_enabled", data.webhook_enabled)
    data.webhook_listen_host = str(
        raw.get("webhook_listen_host", data.webhook_listen_host) or "0.0.0.0"
    ).strip() or "0.0.0.0"
    data.webhook_listen_port = _positive_int(
        raw, "webhook_listen_port", data.webhook_listen_port, minimum=1
    )
    if data.webhook_listen_port > 65535:
        raise ConfigError("display.webhook_listen_port must be between 1 and 65535")
    data.webhook_secret = str(raw.get("webhook_secret", data.webhook_secret) or "").strip()
    data.webhook_pulse_seconds = float(
        raw.get("webhook_pulse_seconds", data.webhook_pulse_seconds) or data.webhook_pulse_seconds
    )
    if not (1.0 <= data.webhook_pulse_seconds <= 120.0):
        raise ConfigError("display.webhook_pulse_seconds must be between 1 and 120")
    try:
        data.webhook_incoming = parse_incoming_webhooks(
            raw.get("webhook_incoming", data.webhook_incoming)
        )
    except ValueError as exc:
        raise ConfigError(f"display.{exc}") from exc
    try:
        data.webhook_outgoing = parse_outgoing_webhooks(
            raw.get("webhook_outgoing", data.webhook_outgoing)
        )
    except ValueError as exc:
        raise ConfigError(f"display.{exc}") from exc
    data.power_mode = str(raw.get("power_mode", data.power_mode)).strip().lower() or "auto"
    if data.power_mode not in VALID_POWER_MODES:
        raise ConfigError(f"display.power_mode must be one of {VALID_POWER_MODES}")
    data.low_power_fps = float(raw.get("low_power_fps", data.low_power_fps) or data.low_power_fps)
    if not (4.0 <= data.low_power_fps <= 30.0):
        raise ConfigError("display.low_power_fps must be between 4 and 30")
    data.safe_mode_on_crash = _bool(raw, "safe_mode_on_crash", data.safe_mode_on_crash)
    return data


def _parse_camera(raw: Any, index: int) -> CameraConfig:
    if not isinstance(raw, dict):
        raise ConfigError(f"cameras[{index}] must be a mapping")
    name = str(raw.get("name") or f"Camera {index + 1}").strip()
    if not name:
        raise ConfigError(f"cameras[{index}].name cannot be empty")
    url = raw.get("url")
    device = raw.get("device")
    if url is not None:
        url = str(url).strip() or None
    if device is not None:
        try:
            device = int(device)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"cameras[{index}].device must be an integer") from exc
        if device < 0:
            raise ConfigError(f"cameras[{index}].device must be >= 0")
    if url is None and device is None:
        raise ConfigError(f"cameras[{index}] ({name}) needs a url or device")
    if url and device is None:
        scheme = urlparse(url).scheme.lower()
        looks_like_path = Path(url).suffix.lower() in {".mp4", ".mkv", ".avi", ".mov", ".m4v"}
        if scheme and scheme not in URL_SCHEMES and not looks_like_path:
            raise ConfigError(
                f"cameras[{index}] ({name}) url scheme {scheme!r} is not supported "
                f"(use {', '.join(URL_SCHEMES)}, a file path, or device)"
            )
    transport = raw.get("transport")
    if transport is not None:
        transport = str(transport).lower()
        if transport not in VALID_TRANSPORTS:
            raise ConfigError(
                f"cameras[{index}].transport must be one of {VALID_TRANSPORTS}"
            )
    username = raw.get("username")
    password = raw.get("password")
    kind = raw.get("type")
    if kind is not None:
        kind = str(kind).strip().lower()
        if kind not in VALID_CAMERA_KINDS:
            raise ConfigError(
                f"cameras[{index}].type must be one of {VALID_CAMERA_KINDS}"
            )
    ssh_port = 22
    if "ssh_port" in raw:
        try:
            ssh_port = int(raw["ssh_port"])
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"cameras[{index}].ssh_port must be an integer") from exc
        if ssh_port < 1:
            raise ConfigError(f"cameras[{index}].ssh_port must be >= 1")
    http_port = 80
    if "http_port" in raw:
        try:
            http_port = int(raw["http_port"])
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"cameras[{index}].http_port must be an integer") from exc
        if http_port < 1:
            raise ConfigError(f"cameras[{index}].http_port must be >= 1")
    rotate = 0
    if "rotate" in raw:
        try:
            rotate = int(raw["rotate"])
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"cameras[{index}].rotate must be an integer") from exc
        if rotate not in VALID_CAMERA_ROTATIONS:
            raise ConfigError(
                f"cameras[{index}].rotate must be one of {VALID_CAMERA_ROTATIONS}"
            )
    encroach_line = None
    line_raw = raw.get("encroach_line")
    if line_raw not in (None, [], ()):
        try:
            from security_monitor.encroachment import parse_encroach_line

            encroach_line = parse_encroach_line(line_raw)
        except ValueError as exc:
            raise ConfigError(f"cameras[{index}].{exc}") from exc
    encroach_side = str(raw.get("encroach_side", "positive") or "positive").strip().lower()
    if encroach_side not in VALID_ENCROACH_SIDES:
        raise ConfigError(
            f"cameras[{index}].encroach_side must be one of {VALID_ENCROACH_SIDES}"
        )
    encroach_zones: list[EncroachZone] = []
    if "encroach_zones" in raw and raw.get("encroach_zones") is not None:
        try:
            from security_monitor.encroachment import parse_encroach_zones

            encroach_zones = parse_encroach_zones(raw.get("encroach_zones"))
        except ValueError as exc:
            raise ConfigError(f"cameras[{index}].{exc}") from exc
    ha_door_entity = str(raw.get("ha_door_entity", "") or "").strip()
    ha_door_label = str(raw.get("ha_door_label", "") or "").strip()
    return CameraConfig(
        name=name,
        url=url,
        device=device,
        enabled=_bool(raw, "enabled", True),
        transport=transport,
        username=None if username is None else str(username),
        password=None if password is None else str(password),
        kind=kind,
        reboot=_bool(raw, "reboot", True),
        ssh_port=ssh_port,
        http_port=http_port,
        rotate=rotate,
        detect_people=_bool(raw, "detect_people", False),
        detect_objects=_bool(raw, "detect_objects", False),
        detect_encroachment=_bool(raw, "detect_encroachment", False),
        encroach_zones=encroach_zones,
        encroach_line=encroach_line,
        encroach_side=encroach_side,
        ha_door_entity=ha_door_entity,
        ha_door_label=ha_door_label,
    )


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ConfigError("cameras: must be a list")
    return value


def _bool(raw: dict[str, Any], key: str, default: bool) -> bool:
    if key not in raw:
        return default
    value = raw[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "yes", "1", "on"}:
        return True
    if isinstance(value, str) and value.lower() in {"false", "no", "0", "off"}:
        return False
    raise ConfigError(f"{key} must be a boolean")


def _positive_int(
    raw: dict[str, Any],
    key: str,
    default: int,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    if key not in raw:
        return default
    try:
        value = int(raw[key])
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"display.{key} must be an integer") from exc
    if value < minimum or (maximum is not None and value > maximum):
        bound = f">= {minimum}" if maximum is None else f"{minimum}..{maximum}"
        raise ConfigError(f"display.{key} must be {bound}")
    return value


def _positive_float(raw: dict[str, Any], key: str, default: float) -> float:
    if key not in raw:
        return default
    try:
        value = float(raw[key])
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"display.{key} must be a number") from exc
    if value <= 0:
        raise ConfigError(f"display.{key} must be > 0")
    return value
