from __future__ import annotations

import numpy as np
import pytest

from security_monitor.mosaic import (
    clamp_center,
    escape_action,
    magnify,
    placeholder,
    scale_frame,
    zoom_toward,
)
from security_monitor.overlay import draw_text
from security_monitor.stream import rotate_frame


def test_scale_modes_output_cell_size() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for mode in ("fit", "fill", "stretch"):
        out = scale_frame(frame, 320, 180, mode)
        assert out.shape == (180, 320, 3)


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
