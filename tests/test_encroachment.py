"""Tests for per-camera encroachment zones (lines + polygons) and alarms."""

from __future__ import annotations

import numpy as np
import pytest

from security_monitor.config import ConfigError, parse_config, save_display_settings
from security_monitor.detection import Box
from security_monitor.encroachment import (
    DEFAULT_ENCROACH_LINE,
    EncroachLine,
    EncroachZone,
    any_person_in_zone,
    draw_alarm_banner,
    draw_encroach_highlight,
    draw_zones,
    effective_zones,
    evaluate_zones,
    line_preset_label,
    map_tile_click_to_frame_norm,
    next_line_preset,
    next_polygon_preset,
    parse_encroach_line,
    parse_encroach_zones,
    point_in_polygon,
    point_in_zone,
    resolve_line,
)


def test_horizontal_line_zone_sides() -> None:
    line = EncroachLine(0.0, 0.5, 1.0, 0.5, side="positive")
    assert point_in_zone(line, 0.5, 0.8) is True
    assert point_in_zone(line, 0.5, 0.2) is False
    flipped = line.flip_side()
    assert point_in_zone(flipped, 0.5, 0.2) is True
    assert point_in_zone(flipped, 0.5, 0.8) is False


def test_person_feet_in_zone() -> None:
    line = resolve_line(DEFAULT_ENCROACH_LINE, "positive")
    person = Box(40, 40, 80, 180, "person")
    assert any_person_in_zone(line, [person], 200, 200) is True
    above = Box(40, 10, 80, 60, "person")
    assert any_person_in_zone(line, [above], 200, 200) is False


def test_polygon_contains_feet() -> None:
    porch = EncroachZone(
        "Porch",
        ((0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)),
    )
    assert point_in_polygon(porch.points, 0.5, 0.8) is True
    assert point_in_polygon(porch.points, 0.5, 0.2) is False
    person = Box(40, 100, 80, 180, "person")
    ok, names = evaluate_zones([porch], [person], 200, 200)
    assert ok is True
    assert names == ("Porch",)


def test_multi_zone_hits_any() -> None:
    left = EncroachZone("Left", ((0.0, 0.0), (0.4, 0.0), (0.4, 1.0), (0.0, 1.0)))
    right = EncroachZone("Right", ((0.6, 0.0), (1.0, 0.0), (1.0, 1.0), (0.6, 1.0)))
    # Feet near right side of frame.
    person = Box(150, 40, 190, 180, "person")
    ok, names = evaluate_zones([left, right], [person], 200, 200)
    assert ok is True
    assert names == ("Right",)


def test_effective_zones_legacy_fallback() -> None:
    zones = effective_zones([], legacy_line=(0.0, 0.5, 1.0, 0.5), legacy_side="negative")
    assert len(zones) == 1
    assert zones[0].is_line
    assert zones[0].side == "negative"


def test_parse_and_presets() -> None:
    assert parse_encroach_line([0, 0.5, 1, 0.5]) == (0.0, 0.5, 1.0, 0.5)
    with pytest.raises(ValueError):
        parse_encroach_line([0.5, 0.5, 0.5, 0.5])
    nxt = next_line_preset(DEFAULT_ENCROACH_LINE, 1)
    assert line_preset_label(nxt) == "Horizontal low"
    name, pts = next_polygon_preset(None, 1)
    assert name == "Bottom half"
    assert len(pts) == 4
    name2, _ = next_polygon_preset(pts, 1)
    assert name2 == "Bottom third"


def test_parse_encroach_zones_yaml() -> None:
    zones = parse_encroach_zones(
        [
            {"name": "Wire", "kind": "line", "points": [[0, 0.5], [1, 0.5]], "side": "negative"},
            {
                "name": "Porch",
                "kind": "polygon",
                "points": [[0.1, 0.6], [0.9, 0.6], [0.9, 0.95], [0.1, 0.95]],
            },
        ]
    )
    assert len(zones) == 2
    assert zones[0].is_line and zones[0].side == "negative"
    assert zones[1].is_polygon and zones[1].name == "Porch"


def test_map_tile_click_fit_letterbox() -> None:
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
                "encroachment_alarm": True,
                "encroachment_alarm_sound": False,
            },
            "cameras": [
                {
                    "name": "Gate",
                    "url": "rtsp://1.2.3.4/x",
                    "detect_encroachment": True,
                    "encroach_line": [0.1, 0.2, 0.9, 0.8],
                    "encroach_side": "negative",
                    "encroach_zones": [
                        {
                            "name": "Porch",
                            "kind": "polygon",
                            "points": [[0.1, 0.6], [0.9, 0.6], [0.9, 0.95], [0.1, 0.95]],
                        }
                    ],
                }
            ],
        },
        path=path,
    )
    assert cfg.display.encroachment_detection is True
    assert cfg.display.encroachment_alarm is True
    assert cfg.display.encroachment_alarm_sound is False
    cam = cfg.cameras[0]
    assert cam.detect_encroachment is True
    assert cam.encroach_line == (0.1, 0.2, 0.9, 0.8)
    assert cam.encroach_side == "negative"
    assert len(cam.encroach_zones) == 1
    assert cam.encroach_zones[0].name == "Porch"
    assert save_display_settings(cfg) == path
    text = path.read_text(encoding="utf-8")
    assert "encroachment_detection: true" in text
    assert "encroachment_alarm: true" in text
    assert "encroachment_alarm_sound: false" in text
    assert "detect_encroachment: true" in text
    assert "encroach_zones:" in text
    assert "Porch" in text


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
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    zones = [
        EncroachZone("Wire", ((0.0, 0.5), (1.0, 0.5))),
        EncroachZone("Box", ((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8))),
    ]
    draw_zones(frame, zones, active_names=("Box",))
    tile = np.zeros((90, 160, 3), dtype=np.uint8)
    draw_encroach_highlight(tile, active=True, pulse=0.9, strong=True)
    assert tile[2, 2, 2] > 0
    canvas = np.zeros((240, 320, 3), dtype=np.uint8)
    draw_alarm_banner(canvas, ["Gate", "Driveway"], pulse=1.0)
    assert canvas[20, 20, 2] > 0


def test_alarm_beep_smoke() -> None:
    from security_monitor.alarm import play_alert_beep

    play_alert_beep(double=False)
