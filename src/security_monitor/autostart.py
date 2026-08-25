"""Install / remove login autostart entries (Ubuntu desktop + Windows)."""

from __future__ import annotations

import os
import shlex
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


DESKTOP_FILENAME = "security-monitor.desktop"
WINDOWS_STARTUP_BAT = "security-monitor.bat"
SYSTEMD_UNIT = "security-monitor.service"


class AutostartError(RuntimeError):
    """Raised when autostart cannot be installed or removed."""


@dataclass(frozen=True)
class AutostartOptions:
    config: Path | None = None
    fullscreen: bool = False
    delay: float = 0.0
    working_directory: Path | None = None
    method: str = "auto"  # auto | desktop | systemd


@dataclass(frozen=True)
class AutostartStatus:
    platform: str
    method: str
    installed: bool
    path: Path | None
    detail: str = ""


def is_windows() -> bool:
    return os.name == "nt"


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def xdg_config_home() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))


def desktop_autostart_path() -> Path:
    return xdg_config_home() / "autostart" / DESKTOP_FILENAME


def systemd_user_unit_path() -> Path:
    return xdg_config_home() / "systemd" / "user" / SYSTEMD_UNIT


def windows_startup_bat_path() -> Path:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        raise AutostartError("APPDATA is not set; cannot locate the Windows Startup folder")
    return (
        Path(appdata)
        / "Microsoft"
        / "Windows"
        / "Start Menu"
        / "Programs"
        / "Startup"
        / WINDOWS_STARTUP_BAT
    )


def resolve_launcher() -> list[str]:
    """Return an absolute argv that launches security-monitor reliably."""
    candidates: list[Path] = []
    which = shutil.which("security-monitor")
    if which:
        candidates.append(Path(which))
    script_dir = Path(sys.executable).resolve().parent
    if is_windows():
        candidates.append(script_dir / "security-monitor.exe")
        candidates.append(script_dir / "security-monitor")
    else:
        candidates.append(script_dir / "security-monitor")

    for path in candidates:
        if path.is_file():
            return [str(path.resolve())]

    # Fallback: same interpreter, module entry (works for editable installs).
    return [str(Path(sys.executable).resolve()), "-m", "security_monitor"]


def build_run_argv(options: AutostartOptions) -> list[str]:
    argv = resolve_launcher()
    if options.config is not None:
        argv.extend(["--config", str(options.config.expanduser().resolve())])
    if options.fullscreen:
        argv.append("--fullscreen")
    if options.delay and options.delay > 0:
        argv.extend(["--delay", str(options.delay)])
    return argv


def _quote_desktop_exec(argv: list[str]) -> str:
    # Desktop Entry Exec uses shell-like quoting.
    return " ".join(shlex.quote(part) for part in argv)


def build_desktop_entry(options: AutostartOptions) -> str:
    argv = build_run_argv(options)
    # Prefer X11 (XWayland) for OpenCV windows on Ubuntu Wayland sessions,
    # and disable Qt HiDPI scaling so the mosaic is not stretched.
    if argv and argv[0] != "env":
        argv = [
            "env",
            "QT_QPA_PLATFORM=xcb",
            "QT_AUTO_SCREEN_SCALE_FACTOR=0",
            "QT_ENABLE_HIGHDPI_SCALING=0",
            "QT_SCALE_FACTOR=1",
            *argv,
        ]
    exec_line = _quote_desktop_exec(argv)
    workdir = options.working_directory
    if workdir is None and options.config is not None:
        workdir = options.config.expanduser().resolve().parent
    if workdir is None:
        workdir = Path.home()

    lines = [
        "[Desktop Entry]",
        "Type=Application",
        "Version=1.0",
        "Name=Security Monitor",
        "Comment=Multi-camera RTSP/RTP mosaic viewer",
        f"Exec={exec_line}",
        f"Path={workdir.expanduser().resolve()}",
        "Terminal=false",
        "Categories=AudioVideo;Monitor;",
        "StartupNotify=false",
        "X-GNOME-Autostart-enabled=true",
        "X-GNOME-Autostart-Phase=Application",
    ]
    return "\n".join(lines) + "\n"


def build_systemd_unit(options: AutostartOptions) -> str:
    argv = build_run_argv(options)
    exec_line = " ".join(shlex.quote(part) for part in argv)
    workdir = options.working_directory
    if workdir is None and options.config is not None:
        workdir = options.config.expanduser().resolve().parent
    if workdir is None:
        workdir = Path.home()

    # graphical-session.target is reached after the user logs into a desktop.
    return (
        "[Unit]\n"
        "Description=Security Monitor camera mosaic\n"
        "PartOf=graphical-session.target\n"
        "After=graphical-session.target\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        f"WorkingDirectory={workdir.expanduser().resolve()}\n"
        f"ExecStart={exec_line}\n"
        "Restart=on-failure\n"
        "RestartSec=5\n"
        # OpenCV GUI backends are more reliable on XWayland; disable HiDPI stretch.
        "Environment=QT_QPA_PLATFORM=xcb\n"
        "Environment=QT_AUTO_SCREEN_SCALE_FACTOR=0\n"
        "Environment=QT_ENABLE_HIGHDPI_SCALING=0\n"
        "Environment=QT_SCALE_FACTOR=1\n"
        "\n"
        "[Install]\n"
        "WantedBy=graphical-session.target\n"
    )


def build_windows_bat(options: AutostartOptions) -> str:
    argv = build_run_argv(options)
    workdir = options.working_directory
    if workdir is None and options.config is not None:
        workdir = options.config.expanduser().resolve().parent
    quoted = " ".join(f'"{part}"' if " " in part else part for part in argv)
    lines = ["@echo off"]
    if workdir is not None:
        lines.append(f'cd /d "{workdir.expanduser().resolve()}"')
    lines.append(quoted)
    return "\r\n".join(lines) + "\r\n"


def _choose_linux_method(method: str) -> str:
    if method == "auto":
        return "desktop"
    if method in {"desktop", "systemd"}:
        return method
    raise AutostartError(f"Unknown autostart method: {method!r} (use desktop or systemd)")


def install_autostart(options: AutostartOptions) -> AutostartStatus:
    if options.config is not None and not options.config.expanduser().is_file():
        raise AutostartError(f"Config file not found: {options.config}")

    if is_windows():
        path = windows_startup_bat_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(build_windows_bat(options), encoding="utf-8")
        return AutostartStatus(
            platform="windows",
            method="startup-folder",
            installed=True,
            path=path,
            detail="Starts with Windows after you sign in.",
        )

    if not is_linux():
        raise AutostartError(
            f"Autostart install is only supported on Linux and Windows (got {sys.platform})"
        )

    method = _choose_linux_method(options.method)
    if method == "desktop":
        path = desktop_autostart_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(build_desktop_entry(options), encoding="utf-8")
        return AutostartStatus(
            platform="linux",
            method="desktop",
            installed=True,
            path=path,
            detail="Starts after you log into your Ubuntu desktop session.",
        )

    path = systemd_user_unit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_systemd_unit(options), encoding="utf-8")
    enabled = _systemd_user_enable()
    detail = "systemd user unit installed"
    if enabled is True:
        detail += " and enabled for graphical-session.target"
    elif enabled is False:
        detail += (
            "; run: systemctl --user daemon-reload && "
            "systemctl --user enable --now security-monitor.service"
        )
    return AutostartStatus(
        platform="linux",
        method="systemd",
        installed=True,
        path=path,
        detail=detail,
    )


def uninstall_autostart(*, method: str = "auto") -> list[AutostartStatus]:
    removed: list[AutostartStatus] = []
    if is_windows():
        path = windows_startup_bat_path()
        existed = path.is_file()
        if existed:
            path.unlink()
        removed.append(
            AutostartStatus(
                platform="windows",
                method="startup-folder",
                installed=False,
                path=path,
                detail="removed" if existed else "was not installed",
            )
        )
        return removed

    if not is_linux():
        raise AutostartError(
            f"Autostart uninstall is only supported on Linux and Windows (got {sys.platform})"
        )

    targets: list[tuple[str, Path]] = []
    chosen = method if method != "auto" else None
    if chosen in (None, "desktop"):
        targets.append(("desktop", desktop_autostart_path()))
    if chosen in (None, "systemd"):
        targets.append(("systemd", systemd_user_unit_path()))
    if chosen not in (None, "desktop", "systemd"):
        raise AutostartError(f"Unknown autostart method: {method!r}")

    for name, path in targets:
        existed = path.is_file()
        if name == "systemd" and existed:
            _systemd_user_disable()
        if existed:
            path.unlink()
        removed.append(
            AutostartStatus(
                platform="linux",
                method=name,
                installed=False,
                path=path,
                detail="removed" if existed else "was not installed",
            )
        )
    return removed


def status_autostart() -> list[AutostartStatus]:
    if is_windows():
        path = windows_startup_bat_path()
        return [
            AutostartStatus(
                platform="windows",
                method="startup-folder",
                installed=path.is_file(),
                path=path,
                detail=path.read_text(encoding="utf-8")[:200] if path.is_file() else "",
            )
        ]

    results = [
        AutostartStatus(
            platform="linux",
            method="desktop",
            installed=desktop_autostart_path().is_file(),
            path=desktop_autostart_path(),
        ),
        AutostartStatus(
            platform="linux",
            method="systemd",
            installed=systemd_user_unit_path().is_file(),
            path=systemd_user_unit_path(),
        ),
    ]
    return results


def _systemd_user_enable() -> bool | None:
    """Best-effort enable. Returns True/False, or None if systemctl is unavailable."""
    systemctl = shutil.which("systemctl")
    if not systemctl:
        return None
    import subprocess

    try:
        subprocess.run(
            [systemctl, "--user", "daemon-reload"],
            check=False,
            capture_output=True,
            text=True,
        )
        result = subprocess.run(
            [systemctl, "--user", "enable", SYSTEMD_UNIT],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except OSError:
        return False


def _systemd_user_disable() -> None:
    systemctl = shutil.which("systemctl")
    if not systemctl:
        return
    import subprocess

    try:
        subprocess.run(
            [systemctl, "--user", "disable", "--now", SYSTEMD_UNIT],
            check=False,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [systemctl, "--user", "daemon-reload"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return
