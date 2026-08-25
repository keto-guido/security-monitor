from __future__ import annotations

from pathlib import Path

import pytest

from security_monitor.autostart import (
    AutostartOptions,
    build_desktop_entry,
    build_run_argv,
    build_systemd_unit,
    build_windows_bat,
    desktop_autostart_path,
    install_autostart,
    status_autostart,
    uninstall_autostart,
)
from security_monitor.cli import main


def test_build_run_argv_includes_flags(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("cameras: []\n", encoding="utf-8")
    argv = build_run_argv(
        AutostartOptions(config=cfg, fullscreen=True, delay=12.5)
    )
    assert "--fullscreen" in argv
    assert argv[argv.index("--delay") + 1] == "12.5"
    assert argv[argv.index("--config") + 1] == str(cfg.resolve())


def test_desktop_entry_uses_x11_and_paths(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("ok\n", encoding="utf-8")
    text = build_desktop_entry(AutostartOptions(config=cfg, fullscreen=True, delay=5))
    assert text.startswith("[Desktop Entry]\n")
    assert "QT_QPA_PLATFORM=xcb" in text
    assert "--fullscreen" in text
    assert str(cfg.resolve()) in text
    assert f"Path={tmp_path.resolve()}" in text
    assert "X-GNOME-Autostart-enabled=true" in text


def test_systemd_unit_targets_graphical_session(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("ok\n", encoding="utf-8")
    text = build_systemd_unit(AutostartOptions(config=cfg, delay=10))
    assert "WantedBy=graphical-session.target" in text
    assert "QT_QPA_PLATFORM=xcb" in text
    assert "--delay" in text


def test_windows_bat_changes_directory(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("ok\n", encoding="utf-8")
    text = build_windows_bat(AutostartOptions(config=cfg, fullscreen=True))
    assert text.startswith("@echo off")
    assert f'cd /d "{tmp_path.resolve()}"' in text
    assert "--fullscreen" in text


def test_install_and_uninstall_desktop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr("security_monitor.autostart.is_windows", lambda: False)
    monkeypatch.setattr("security_monitor.autostart.is_linux", lambda: True)

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "display:\n  columns: 1\n  rows: 1\ncameras:\n  - name: A\n    url: rtsp://1.2.3.4/x\n",
        encoding="utf-8",
    )
    status = install_autostart(
        AutostartOptions(config=cfg, fullscreen=True, delay=3, method="desktop")
    )
    assert status.installed
    assert status.path == desktop_autostart_path()
    assert status.path is not None and status.path.is_file()
    body = status.path.read_text(encoding="utf-8")
    assert "Security Monitor" in body
    assert "--fullscreen" in body

    results = status_autostart()
    assert any(item.installed and item.method == "desktop" for item in results)

    removed = uninstall_autostart(method="desktop")
    assert removed[0].detail == "removed"
    assert not desktop_autostart_path().is_file()


def test_cli_autostart_status_and_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr("security_monitor.autostart.is_windows", lambda: False)
    monkeypatch.setattr("security_monitor.autostart.is_linux", lambda: True)

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "display:\n  columns: 1\n  rows: 1\ncameras:\n  - name: A\n    url: rtsp://1.2.3.4/x\n",
        encoding="utf-8",
    )

    assert main(["autostart", "status"]) == 1
    assert main(["autostart", "install", "--config", str(cfg), "--fullscreen", "--delay", "1"]) == 0
    out = capsys.readouterr().out
    assert "Installed desktop autostart" in out
    assert main(["autostart", "status"]) == 0
    assert main(["autostart", "uninstall"]) == 0


def test_init_user_writes_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr("security_monitor.config.os.name", "posix")
    assert main(["init", "--user"]) == 0
    dest = tmp_path / "xdg" / "security-monitor" / "config.yaml"
    assert dest.is_file()
    assert "cameras:" in dest.read_text(encoding="utf-8")
