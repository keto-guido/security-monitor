"""Tests for Linux display orientation helpers."""

from __future__ import annotations

import pytest

from security_monitor.display_setup import (
    DisplaySetupError,
    apply_screen_rotate,
    list_outputs,
    maybe_apply_config_rotation,
    normalize_rotation,
    pick_output,
    sanitize_window_size,
)


SAMPLE_XRANDR = """\
Screen 0: minimum 8 x 8, current 1920 x 1080, maximum 32767 x 32767
HDMI-1 connected primary 1920x1080+0+0 (normal left inverted right x axis y axis) 600mm x 340mm
   1920x1080     60.00*+  59.94
   1280x720      60.00
DP-1 connected 1280x1024+1920+0 left (normal left inverted right x axis y axis) 380mm x 300mm
   1280x1024     60.02*+
eDP-1 disconnected (normal left inverted right x axis y axis)
"""


def test_normalize_rotation_aliases() -> None:
    assert normalize_rotation("LEFT") == "left"
    assert normalize_rotation("none") == "none"
    with pytest.raises(DisplaySetupError):
        normalize_rotation("clockwise")


def test_list_outputs_parses_rotation_and_primary() -> None:
    outputs = list_outputs(SAMPLE_XRANDR)
    by_name = {out.name: out for out in outputs}
    assert by_name["HDMI-1"].connected
    assert by_name["HDMI-1"].primary
    assert by_name["HDMI-1"].width == 1920
    assert by_name["HDMI-1"].height == 1080
    assert by_name["HDMI-1"].rotation == "normal"
    assert by_name["HDMI-1"].preferred == "1920x1080"
    assert by_name["DP-1"].rotation == "left"
    assert by_name["DP-1"].primary is False
    assert by_name["eDP-1"].connected is False


def test_pick_output_prefers_primary_then_named() -> None:
    outputs = list_outputs(SAMPLE_XRANDR)
    assert pick_output(outputs).name == "HDMI-1"
    assert pick_output(outputs, "DP-1").name == "DP-1"
    with pytest.raises(DisplaySetupError, match="not found"):
        pick_output(outputs, "VGA-1")


def test_apply_none_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_args: list[str]) -> str:
        raise AssertionError("xrandr should not run for none")

    monkeypatch.setattr("security_monitor.display_setup._run_xrandr", boom)
    assert "unchanged" in apply_screen_rotate("none")


def test_apply_rotate_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "security_monitor.display_setup.list_outputs",
        lambda: list_outputs(SAMPLE_XRANDR),
    )
    message = apply_screen_rotate("left", dry_run=True)
    assert "dry-run" in message
    assert "HDMI-1" in message
    assert "--rotate left" in message


def test_maybe_apply_skips_none() -> None:
    assert maybe_apply_config_rotation("none") is None


def test_maybe_apply_invalid_rotation_does_not_raise() -> None:
    message = maybe_apply_config_rotation("clockwise")
    assert message is not None
    assert "failed" in message


def test_maybe_apply_swallows_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "security_monitor.display_setup.apply_screen_rotate",
        lambda *_a, **_k: (_ for _ in ()).throw(DisplaySetupError("no display")),
    )
    monkeypatch.setattr("security_monitor.display_setup.is_linux", lambda: True)
    message = maybe_apply_config_rotation("left")
    assert message is not None
    assert "failed" in message


def test_sanitize_rejects_squished_fullscreen_rect() -> None:
    # Reproduces the OpenCV Qt bug: 1920x547 reported on a 1920x1200 screen.
    assert sanitize_window_size(
        (1920, 547),
        fullscreen=True,
        screen=(1920, 1200),
        fallback=(1280, 720),
    ) == (1920, 1200)


def test_sanitize_rejects_wrong_fullscreen_mode() -> None:
    assert sanitize_window_size(
        (1920, 1080),
        fullscreen=True,
        screen=(1920, 1200),
        fallback=(1280, 720),
    ) == (1920, 1200)


def test_sanitize_keeps_sane_windowed_size() -> None:
    assert sanitize_window_size(
        (1280, 720),
        fullscreen=False,
        screen=(1920, 1200),
        fallback=(640, 360),
    ) == (1280, 720)


def test_sanitize_falls_back_when_too_small() -> None:
    assert sanitize_window_size(
        (100, 80),
        fullscreen=False,
        screen=None,
        fallback=(1280, 720),
    ) == (1280, 720)
