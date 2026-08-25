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

VALID_SCALE_MODES = ("fit", "fill", "stretch")
VALID_TRANSPORTS = ("tcp", "udp", "auto")
VALID_CAMERA_KINDS = ("ubiquiti", "reolink", "amcrest", "dahua")
URL_SCHEMES = ("rtsp", "rtp", "http", "https", "file", "rtmp")

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

    @property
    def tile_count(self) -> int:
        return self.columns * self.rows

    @property
    def canvas_size(self) -> tuple[int, int]:
        return self.columns * self.cell_width, self.rows * self.cell_height


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
