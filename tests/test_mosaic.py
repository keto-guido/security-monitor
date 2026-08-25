from __future__ import annotations

import numpy as np

from security_monitor.mosaic import placeholder, scale_frame


def test_scale_modes_output_cell_size() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for mode in ("fit", "fill", "stretch"):
        out = scale_frame(frame, 320, 180, mode)
        assert out.shape == (180, 320, 3)


def test_placeholder_size() -> None:
    tile = placeholder(400, 300, "Cam", "NO SIGNAL")
    assert tile.shape == (300, 400, 3)
