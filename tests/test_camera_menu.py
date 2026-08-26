"""Tests for camera layout / order helpers used by the Cameras menu."""

from __future__ import annotations

from pathlib import Path

from security_monitor.config import (
    CameraConfig,
    DisplayConfig,
    ensure_layout_fits,
    layout_presets_for_count,
    move_camera,
    next_layout_preset,
    parse_config,
    save_display_settings,
    unique_camera_name,
)


def test_next_layout_preset_cycles() -> None:
    assert next_layout_preset(2, 2, 1) == (2, 3)
    assert next_layout_preset(1, 1, -1)[0] * next_layout_preset(1, 1, -1)[1] >= 1


def test_next_layout_preset_stays_sized_for_cameras() -> None:
    assert layout_presets_for_count(4) == [(2, 2), (2, 3), (3, 2)]
    assert next_layout_preset(2, 2, 1, min_tiles=4) == (2, 3)
    assert next_layout_preset(2, 3, 1, min_tiles=4) == (3, 2)
    assert next_layout_preset(3, 2, 1, min_tiles=4) == (2, 2)
    assert next_layout_preset(4, 4, 1, min_tiles=4) == (2, 2)
    assert next_layout_preset(2, 2, -1, min_tiles=4) == (3, 2)


def test_move_camera_reorders() -> None:
    cams = [
        CameraConfig(name="A", url="rtsp://1"),
        CameraConfig(name="B", url="rtsp://2"),
        CameraConfig(name="C", url="rtsp://3"),
    ]
    assert move_camera(cams, 1, -1) == 0
    assert [c.name for c in cams] == ["B", "A", "C"]
    assert move_camera(cams, 0, -1) == 0  # clamped


def test_unique_camera_name() -> None:
    cams = [CameraConfig(name="Porch", url="rtsp://1")]
    assert unique_camera_name(cams, "Porch") == "Porch 2"
    assert unique_camera_name(cams, "Gate") == "Gate"


def test_ensure_layout_fits_grows_grid() -> None:
    display = DisplayConfig(columns=2, rows=2)
    assert ensure_layout_fits(display, 4) is False
    assert ensure_layout_fits(display, 5) is True
    assert display.tile_count >= 5


def test_save_persists_layout_order_and_cycle(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "display:\n  columns: 2\n  rows: 2\n"
        "cameras:\n  - name: A\n    url: rtsp://1.2.3.4/a\n"
        "  - name: B\n    url: rtsp://1.2.3.4/b\n",
        encoding="utf-8",
    )
    cfg = parse_config(
        {
            "display": {"columns": 2, "rows": 2},
            "cameras": [
                {"name": "A", "url": "rtsp://1.2.3.4/a"},
                {"name": "B", "url": "rtsp://1.2.3.4/b"},
            ],
        },
        path=path,
    )
    cfg.display.columns = 3
    cfg.display.rows = 2
    cfg.display.cycle_focus = True
    cfg.display.cycle_focus_seconds = 30
    cfg.cameras[0].enabled = False
    move_camera(cfg.cameras, 1, -1)
    cfg.cameras.append(CameraConfig(name="C", url="rtsp://1.2.3.4/c", enabled=True))
    assert save_display_settings(cfg) == path
    text = path.read_text(encoding="utf-8")
    assert "columns: 3" in text
    assert "rows: 2" in text
    assert "cycle_focus: true" in text
    assert "cycle_focus_seconds: 30" in text
    # Order should be B, A, C with A hidden.
    assert text.index("name: B") < text.index("name: A") < text.index("name: C")
    assert "enabled: false" in text
