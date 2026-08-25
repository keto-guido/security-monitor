"""Tests for per-camera encroachment / tripwire helpers."""

from __future__ import annotations

import numpy as np
import pytest

from security_monitor.config import ConfigError, parse_config, save_display_settings
from security_monitor.detection import Box
from security_monitor.encroachment import (
    DEFAULT_ENCROACH_LINE,
    EncroachLine,
    any_person_in_zone,
    line_preset_label,
    map_tile_click_to_frame_norm,
    next_line_preset,
    parse_encroach_line,
    point_in_zone,
    resolve_line,
)


def test_horizontal_line_zone_sides() -> None:
    line = EncroachLine(0.0, 0.5, 1.0, 0.5, side="positive")
    # Image y grows downward → positive side is below the midline.
    assert point_in_zone(line, 0.5, 0.8) is True
    assert point_in_zone(line, 0.5, 0.2) is False
    flipped = line.flip_side()
    assert point_in_zone(flipped, 0.5, 0.2) is True
    assert point_in_zone(flipped, 0.5, 0.8) is False


def test_person_feet_in_zone() -> None:
    line = resolve_line(DEFAULT_ENCROACH_LINE, "positive")
    # Feet at bottom of box below midline.
    person = Box(40, 40, 80, 180, "person")
    assert any_person_in_zone(line, [person], 200, 200) is True
    # Entirely above the line.
    above = Box(40, 10, 80, 60, "person")
    assert any_person_in_zone(line, [above], 200, 200) is False


def test_parse_and_presets() -> None:
    assert parse_encroach_line([0, 0.5, 1, 0.5]) == (0.0, 0.5, 1.0, 0.5)
    with pytest.raises(ValueError):
        parse_encroach_line([0.5, 0.5, 0.5, 0.5])
    nxt = next_line_preset(DEFAULT_ENCROACH_LINE, 1)
    assert line_preset_label(nxt) == "Horizontal low"


def test_map_tile_click_fit_letterbox() -> None:
    # 200x100 frame into 200x200 tile → letterbox top/bottom of 50px each.
    mapped = map_tile_click_to_frame_norm(
        100,
        100,
        tile_w=200,
        tile_h=200,
        frame_w=200,
        frame_h=100,
        mode="fit",
    )
    assert mapped is not None
    assert mapped[0] == pytest.approx(0.5, abs=0.02)
    assert mapped[1] == pytest.approx(0.5, abs=0.02)
    assert (
        map_tile_click_to_frame_norm(
            10, 10, tile_w=200, tile_h=200, frame_w=200, frame_h=100, mode="fit"
        )
        is None
    )


def test_config_encroachment_roundtrip(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "display:\n  columns: 1\n  rows: 1\ncameras:\n  - name: Gate\n    url: rtsp://1.2.3.4/x\n",
        encoding="utf-8",
    )
    cfg = parse_config(
        {
            "display": {
                "columns": 1,
                "rows": 1,
                "encroachment_detection": True,
                "encroachment_autofocus": True,
            },
            "cameras": [
                {
                    "name": "Gate",
                    "url": "rtsp://1.2.3.4/x",
                    "detect_encroachment": True,
                    "encroach_line": [0.1, 0.2, 0.9, 0.8],
                    "encroach_side": "negative",
                }
            ],
        },
        path=path,
    )
    assert cfg.display.encroachment_detection is True
    assert cfg.display.encroachment_autofocus is True
    cam = cfg.cameras[0]
    assert cam.detect_encroachment is True
    assert cam.encroach_line == (0.1, 0.2, 0.9, 0.8)
    assert cam.encroach_side == "negative"
    assert save_display_settings(cfg) == path
    text = path.read_text(encoding="utf-8")
    assert "encroachment_detection: true" in text
    assert "encroachment_autofocus: true" in text
    assert "detect_encroachment: true" in text
    assert "encroach_side: negative" in text


def test_config_rejects_bad_encroach_side() -> None:
    with pytest.raises(ConfigError, match="encroach_side"):
        parse_config(
            {
                "cameras": [
                    {
                        "name": "A",
                        "url": "rtsp://1.2.3.4/x",
                        "encroach_side": "inside",
                    }
                ]
            }
        )


def test_draw_helpers_smoke() -> None:
    from security_monitor.encroachment import draw_encroach_highlight, draw_encroach_line

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    line = resolve_line(None, "positive")
    draw_encroach_line(frame, line, active=True)
    tile = np.zeros((90, 160, 3), dtype=np.uint8)
    draw_encroach_highlight(tile, active=True)
    assert tile[2, 2, 2] > 0  # red channel of border
