from __future__ import annotations

import numpy as np
import pytest

from security_monitor.mosaic import (
    _menu_action_cycles,
    clamp_center,
    escape_action,
    fallback_canvas,
    first_selectable_index,
    is_menu_header,
    magnify,
    menu_section,
    menu_should_pause_decode,
    placeholder,
    resolve_zone_target,
    root_menu_items,
    scale_frame,
    selectable_menu_entries,
    step_menu_index,
    visible_menu_range,
    zoom_toward,
)
from security_monitor.overlay import draw_text
from security_monitor.stream import rotate_frame


def test_scale_modes_output_cell_size() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for mode in ("fit", "fill", "stretch"):
        out = scale_frame(frame, 320, 180, mode)
        assert out.shape == (180, 320, 3)


def test_scale_frame_swallows_malformed_input() -> None:
    out = scale_frame(np.zeros((0, 0, 3), dtype=np.uint8), 80, 45, "fit")
    assert out.shape == (45, 80, 3)


def test_placeholder_size() -> None:
    tile = placeholder(400, 300, "Cam", "NO SIGNAL")
    assert tile.shape == (300, 400, 3)
    assert tile.sum() > 0


def test_draw_text_is_anti_aliased() -> None:
    tile = np.zeros((80, 240, 3), dtype=np.uint8)
    draw_text(tile, "Front Door", (12, 50), size=16, valign="bottom")
    assert tile.max() > 0
    # Soft edges from 2x raster + shadow should use intermediate values.
    assert ((tile > 8) & (tile < 247)).any()


def test_clamp_center_at_1x() -> None:
    assert clamp_center(1.0, 0.2, 0.9) == (0.5, 0.5)


def test_clamp_center_keeps_crop_in_bounds() -> None:
    cx, cy = clamp_center(4.0, 0.0, 1.0)
    half = 0.5 / 4.0
    assert cx == half
    assert cy == 1.0 - half


def test_zoom_toward_cursor_keeps_focus_point() -> None:
    cx, cy = zoom_toward(1.0, 0.5, 0.5, 2.0, 0.0, 0.0)
    vis = 0.5
    src_x = cx - vis / 2
    assert src_x == pytest.approx(0.0, abs=1e-6)
    assert cy == pytest.approx(0.25)


def test_magnify_preserves_shape_and_zooms() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[:, :20] = (0, 0, 255)
    out = magnify(frame, 2.0, 0.5, 0.5)
    assert out.shape == frame.shape
    # Center crop at 2x should not be dominated by the left red strip.
    assert int(out[:, 0].mean()) < int(frame[:, 0].mean())


def test_escape_from_nested_view_returns_to_main() -> None:
    assert escape_action(menu_open=False, on_main_layout=False) == "main_layout"


def test_escape_on_main_opens_menu() -> None:
    assert escape_action(menu_open=False, on_main_layout=True) == "open_menu"


def test_escape_in_menu_closes_menu() -> None:
    assert escape_action(menu_open=True, on_main_layout=True) == "close_menu"


def test_rotate_frame_swaps_dimensions_for_90() -> None:
    frame = np.zeros((40, 80, 3), dtype=np.uint8)
    frame[0, 0] = (1, 2, 3)
    out = rotate_frame(frame, 90)
    assert out.shape == (80, 40, 3)
    assert tuple(out[0, 39]) == (1, 2, 3)
    assert rotate_frame(frame, 0) is frame
    upside = rotate_frame(frame, 180)
    assert upside.shape == frame.shape
    assert tuple(upside[-1, -1]) == (1, 2, 3)


def test_rotate_frame_returns_original_if_opencv_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = np.zeros((40, 80, 3), dtype=np.uint8)
    frame[0, 0] = (1, 2, 3)

    def boom(*_args, **_kwargs):
        raise RuntimeError("Unknown C++ exception")

    monkeypatch.setattr("security_monitor.stream.cv2.rotate", boom)
    assert rotate_frame(frame, 90) is frame


def test_fallback_canvas_reuses_last_frame() -> None:
    previous = np.ones((10, 20, 3), dtype=np.uint8)
    assert fallback_canvas(previous, (640, 360)) is previous
    blank = fallback_canvas(None, (64, 36))
    assert blank.shape == (36, 64, 3)


def test_menu_sections_are_headers() -> None:
    action, label = menu_section("Cameras")
    assert is_menu_header(action)
    assert label == "Cameras"
    assert not is_menu_header("cameras")
    assert _menu_action_cycles("layout")
    assert _menu_action_cycles("clip_length")
    assert not _menu_action_cycles("cameras")
    assert not _menu_action_cycles("resume")


def test_menu_navigation_skips_headers() -> None:
    items = [
        menu_section("Cameras"),
        ("cameras", "Cameras…"),
        menu_section("Media"),
        ("capture", "Capture…"),
        ("exit", "Exit"),
    ]
    assert first_selectable_index(items) == 1
    assert first_selectable_index(items, start=2) == 3
    assert step_menu_index(items, 1, 1) == 3
    assert step_menu_index(items, 3, 1) == 4
    assert step_menu_index(items, 4, 1) == 1
    assert step_menu_index(items, 1, -1) == 4
    assert selectable_menu_entries(items) == [
        (1, "cameras"),
        (3, "capture"),
        (4, "exit"),
    ]


def test_visible_menu_range_keeps_selection() -> None:
    items = [menu_section("A"), ("one", "One"), ("two", "Two"), ("three", "Three")]
    start, end = visible_menu_range(items, selected=3, max_height=80, row_h=40, header_h=20)
    assert start <= 3 < end
    assert end - start < len(items)


def test_zone_target_uses_named_camera_not_first() -> None:
    from security_monitor.config import CameraConfig

    cams = [
        CameraConfig(name="Front Yard", url="rtsp://a"),
        CameraConfig(name="Front Door", url="rtsp://b"),
        CameraConfig(name="Back Door", url="rtsp://c"),
    ]
    picked = resolve_zone_target(cams, "Front Door", zoom_index=None)
    assert picked is not None
    assert picked.name == "Front Door"
    assert resolve_zone_target(cams, None, zoom_index=2).name == "Back Door"
    assert resolve_zone_target(cams, None, zoom_index=None) is None


def test_menu_pauses_decode_except_live_preview_pages() -> None:
    assert menu_should_pause_decode(menu_open=True, page="root")
    assert menu_should_pause_decode(menu_open=True, page="video")
    assert menu_should_pause_decode(menu_open=True, page="detection")
    assert not menu_should_pause_decode(menu_open=False, page="root")
    assert not menu_should_pause_decode(menu_open=True, page="captures_view")
    assert not menu_should_pause_decode(menu_open=True, page="events_play")
    assert not menu_should_pause_decode(menu_open=True, page="root", zone_editing=True)
    assert not menu_should_pause_decode(menu_open=True, page="root", recording=True)
    assert not menu_should_pause_decode(menu_open=True, page="root", prompt_open=True)
    assert not menu_should_pause_decode(menu_open=True, page="root", overlay_blocking=True)


def test_root_menu_groups_and_keeps_actions() -> None:
    items = root_menu_items(fullscreen=False, safe_mode=False)
    actions = [action for action, _label in items]
    headers = [label for action, label in items if is_menu_header(action)]
    assert headers == ["Live", "Library", "Alerts", "Display", "System"]
    for action in (
        "resume",
        "fullscreen",
        "cameras",
        "capture",
        "captures_root",
        "events_root",
        "detection",
        "weather",
        "ha",
        "video",
        "reconnect",
        "reboot",
        "exit",
    ):
        assert action in actions
    assert first_selectable_index(items) == 0
    assert items[0] == ("resume", "Resume live view")
    safe = root_menu_items(fullscreen=True, safe_mode=True)
    assert ("fullscreen", "Windowed mode") in safe
    assert ("exit_safe_mode", "Exit safe mode (restore extras)") in safe
