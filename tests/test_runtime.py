"""Crash marker, safe-mode decision, and low-power FPS hysteresis."""

from __future__ import annotations

from pathlib import Path

import pytest

from security_monitor.runtime import (
    CrashGuard,
    FramePacer,
    PowerPolicy,
    PowerTracker,
    RateMeter,
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


def test_frame_pacer_absorbs_compose_time() -> None:
    """A 25 fps target with a 30 ms compose must still land on ~40 ms frames."""
    pacer = FramePacer(target_fps=25.0)
    now = 100.0
    periods = []
    previous = None
    for _ in range(20):
        wait = pacer.wait_seconds(now)
        if previous is not None:
            periods.append(now - previous)
        previous = now
        now += wait + 0.030  # compose cost
    settled = periods[-8:]
    assert all(abs(p - 0.040) < 0.002 for p in settled), settled


def test_frame_pacer_cannot_keep_up_runs_flat_out() -> None:
    pacer = FramePacer(target_fps=25.0)
    now = 0.0
    for _ in range(5):
        now += pacer.wait_seconds(now) + 0.120  # compose slower than the period
    assert pacer.wait_seconds(now) == 0.0


def test_frame_pacer_resyncs_after_a_stall() -> None:
    """A long freeze must not leave a backlog the loop sprints through."""
    pacer = FramePacer(target_fps=25.0, max_catch_up=2.0)
    pacer.wait_seconds(0.0)
    assert pacer.wait_seconds(10.0) == 0.0  # gave up on the missed deadlines
    assert pacer.wait_seconds(10.0) == pytest.approx(0.040, abs=1e-6)


def test_frame_pacer_wait_ms_never_blocks_forever() -> None:
    # cv2.waitKey(0) waits for a keypress instead of returning — never emit 0.
    pacer = FramePacer(target_fps=25.0)
    now = 0.0
    for _ in range(6):
        assert pacer.wait_ms(now) >= 1
        now += 0.200


def test_frame_pacer_retargets_on_fps_change() -> None:
    pacer = FramePacer(target_fps=25.0)
    pacer.set_target_fps(10.0)
    assert pacer.period == pytest.approx(0.1)
    assert pacer.wait_seconds(5.0) == pytest.approx(0.1)


def test_rate_meter_counts_events_per_second() -> None:
    meter = RateMeter(window=2.0)
    now = 0.0
    for _ in range(24):
        meter.mark(now)
        now += 1 / 12
    assert meter.rate(now) == pytest.approx(12.0, abs=1.0)


def test_rate_meter_falls_to_zero_when_frames_stop() -> None:
    """The displayed-FPS readout has to admit when nothing is being painted."""
    meter = RateMeter(window=2.0)
    now = 0.0
    for _ in range(50):
        meter.mark(now)
        now += 1 / 25
    assert meter.rate(now) > 20.0
    assert meter.rate(now + 3.0) == 0.0


def test_rate_meter_is_zero_before_it_has_a_sample() -> None:
    assert RateMeter().rate(1.0) == 0.0
