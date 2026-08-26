"""Crash marker, safe-mode decision, and low-power FPS hysteresis."""

from __future__ import annotations

from pathlib import Path

import pytest

from security_monitor.runtime import (
    CrashGuard,
    PowerPolicy,
    PowerTracker,
    clear_crash_marker,
    crash_marker_path,
    power_mode_label,
    should_start_safe_mode,
    write_crash_marker,
)


def test_crash_marker_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.delenv("APPDATA", raising=False)
    assert crash_marker_path().parent == tmp_path / "cfg" / "security-monitor"
    assert not should_start_safe_mode()
    write_crash_marker()
    assert crash_marker_path().is_file()
    assert should_start_safe_mode(on_crash=True)
    assert not should_start_safe_mode(ignore=True)
    assert should_start_safe_mode(force=True, ignore=False, on_crash=False)
    clear_crash_marker()
    assert not crash_marker_path().is_file()
    assert not should_start_safe_mode()


def test_crash_guard_clears_on_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    with CrashGuard() as guard:
        assert crash_marker_path().is_file()
        guard.disarm()
    assert not crash_marker_path().is_file()


def test_crash_guard_keeps_marker_on_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))

    class Boom(RuntimeError):
        pass

    try:
        with CrashGuard():
            raise Boom("compose failed")
    except Boom:
        pass
    assert crash_marker_path().is_file()


def test_crash_guard_clears_on_keyboard_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    try:
        with CrashGuard():
            raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    assert not crash_marker_path().is_file()


def test_power_tracker_auto_enter_and_exit() -> None:
    tracker = PowerTracker(PowerPolicy(mode="auto", fps_threshold=12, enter_seconds=2, exit_seconds=3))
    assert not tracker.update(20.0, 0.0)
    assert not tracker.low_power
    assert not tracker.update(8.0, 1.0)
    assert not tracker.low_power
    assert tracker.update(8.0, 3.1)
    assert tracker.low_power
    assert not tracker.update(10.0, 4.0)
    assert tracker.low_power
    assert not tracker.update(16.0, 5.0)
    assert tracker.update(16.0, 8.1)
    assert not tracker.low_power


def test_power_tracker_force_on_off() -> None:
    tracker = PowerTracker()
    assert tracker.set_mode("on")
    assert tracker.low_power
    assert not tracker.update(60.0, 10.0)
    assert tracker.low_power
    assert tracker.set_mode("off")
    assert not tracker.low_power
    assert not tracker.update(2.0, 20.0)
    assert not tracker.low_power


def test_power_mode_label() -> None:
    assert "Auto" in power_mode_label("auto", threshold=12)
    assert "ON now" in power_mode_label("auto", active=True, threshold=12)
    assert "Off" in power_mode_label("off")
    assert "detection" in power_mode_label("off").lower()
    assert "HUD" in power_mode_label("on")
