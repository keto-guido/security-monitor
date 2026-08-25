from __future__ import annotations

import numpy as np

from security_monitor.mosaic import placeholder, scale_frame
from security_monitor.overlay import draw_text


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
    # Soft edges should use intermediate values, not only 0/255.
    assert ((tile > 20) & (tile < 235)).any()
