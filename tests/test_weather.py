"""Tests for weather HUD layout reservation and config."""

from __future__ import annotations

import numpy as np
import pytest

from security_monitor.config import ConfigError, parse_config, save_display_settings
from security_monitor.weather import (
    WeatherRect,
    WeatherSnapshot,
    compute_tile_rects,
    draw_weather_widget,
    next_opacity,
    next_weather_slot,
    opacity_label,
    resolve_weather_rect,
    shrink_tile_from_reserved,
    slot_label,
    wmo_label,
    WEATHER_OPACITY_CHOICES,
)


def test_between_columns_splits_both_tiles() -> None:
    reserved = resolve_weather_rect(
        slot="between_h",
        norm_x=0.0,
        norm_y=0.0,
        norm_w=0.2,
        norm_h=0.25,
        grid_x=0,
        grid_y=0,
        grid_w=800,
        grid_h=400,
        columns=2,
        rows=1,
        canvas_w=800,
        canvas_h=400,
    )
    left, right = compute_tile_rects(
        columns=2,
        rows=1,
        cell_w=400,
        cell_h=400,
        grid_x=0,
        grid_y=0,
        reserved=reserved,
    )
    assert left[2] < 400
    assert right[2] < 400
    assert left[0] + left[2] <= reserved.x + 1
    assert right[0] >= reserved.x + reserved.w - 1


def test_bottom_right_only_touches_corner_tile() -> None:
    reserved = resolve_weather_rect(
        slot="bottom_right",
        norm_x=0.0,
        norm_y=0.0,
        norm_w=0.25,
        norm_h=0.25,
        grid_x=0,
        grid_y=0,
        grid_w=800,
        grid_h=400,
        columns=2,
        rows=1,
        canvas_w=800,
        canvas_h=400,
    )
    left, right = compute_tile_rects(
        columns=2,
        rows=1,
        cell_w=400,
        cell_h=400,
        grid_x=0,
        grid_y=0,
        reserved=reserved,
    )
    assert left == (0, 0, 400, 400)
    assert right[2] < 400 or right[3] < 400


def test_shrink_no_overlap_unchanged() -> None:
    tile = (0, 0, 100, 100)
    reserved = WeatherRect(200, 200, 50, 50)
    assert shrink_tile_from_reserved(tile, reserved) == tile


def test_slot_cycle_and_labels() -> None:
    assert next_weather_slot("bottom_left", 1) == "bottom_right"
    assert "Between" in slot_label("between_h")
    assert wmo_label(95) == "Thunderstorm"


def test_draw_weather_widget_smoke() -> None:
    canvas = np.zeros((360, 640, 3), dtype=np.uint8)
    snap = WeatherSnapshot(
        temperature_c=22.0,
        weather_code=95,
        condition="Thunderstorm",
        storm_warning="Severe Thunderstorm Warning",
        lightning_risk="High",
        lightning_detail="Thunderstorm",
        place="Testville",
        updated_at=1.0,
    )
    draw_weather_widget(
        canvas,
        WeatherRect(400, 240, 220, 110),
        snap,
        show_temp=True,
        show_conditions=True,
        show_storm=True,
        show_lightning=True,
    )
    assert canvas[250, 420].sum() > 0


def test_draw_weather_opacity_blends_underlay() -> None:
    canvas = np.full((200, 300, 3), 200, dtype=np.uint8)
    snap = WeatherSnapshot(
        temperature_c=10.0,
        weather_code=0,
        condition="Clear",
        place="Blend",
        updated_at=1.0,
    )
    solid = canvas.copy()
    faded = canvas.copy()
    rect = WeatherRect(20, 20, 160, 100)
    draw_weather_widget(solid, rect, snap, opacity=1.0)
    draw_weather_widget(faded, rect, snap, opacity=0.25)
    # Low opacity should stay closer to the bright underlay than full opacity.
    assert int(faded[50, 50].mean()) > int(solid[50, 50].mean())
    assert opacity_label(0.85) == "85%"
    assert next_opacity(WEATHER_OPACITY_CHOICES, 0.85, 1) == 1.0


def test_config_weather_roundtrip(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "display:\n  columns: 2\ncameras:\n  - name: A\n    url: rtsp://1.2.3.4/x\n",
        encoding="utf-8",
    )
    cfg = parse_config(
        {
            "display": {
                "weather_enabled": True,
                "weather_slot": "between_v",
                "weather_x": 0.05,
                "weather_y": -0.02,
                "weather_w": 0.3,
                "weather_h": 0.22,
                "weather_units": "c",
                "weather_show_temp": True,
                "weather_show_conditions": False,
                "weather_show_storm": True,
                "weather_show_lightning": False,
                "weather_opacity": 0.55,
                "weather_overlay": True,
                "hud_opacity": 0.40,
                "weather_latitude": 30.27,
                "weather_longitude": -97.74,
                "weather_place": "Austin",
            },
            "cameras": [{"name": "A", "url": "rtsp://1.2.3.4/x"}],
        },
        path=path,
    )
    assert cfg.display.weather_enabled is True
    assert cfg.display.weather_slot == "between_v"
    assert cfg.display.weather_units == "c"
    assert cfg.display.weather_show_conditions is False
    assert cfg.display.weather_show_lightning is False
    assert cfg.display.weather_opacity == pytest.approx(0.55)
    assert cfg.display.weather_overlay is True
    assert cfg.display.hud_opacity == pytest.approx(0.40)
    assert cfg.display.weather_latitude == pytest.approx(30.27)
    assert save_display_settings(cfg) == path
    text = path.read_text(encoding="utf-8")
    assert "weather_enabled: true" in text
    assert "weather_slot: between_v" in text
    assert "weather_show_lightning: false" in text
    assert "weather_opacity: 0.55" in text
    assert "weather_overlay: true" in text
    assert "hud_opacity: 0.4" in text or "hud_opacity: 0.40" in text
    assert "Austin" in text


def test_config_accepts_percent_opacity() -> None:
    cfg = parse_config(
        {
            "display": {"weather_opacity": 70, "hud_opacity": 50},
            "cameras": [{"name": "A", "url": "rtsp://1.2.3.4/x"}],
        }
    )
    assert cfg.display.weather_opacity == pytest.approx(0.70)
    assert cfg.display.hud_opacity == pytest.approx(0.50)


def test_config_rejects_bad_weather_slot() -> None:
    with pytest.raises(ConfigError, match="weather_slot"):
        parse_config(
            {
                "display": {"weather_slot": "top"},
                "cameras": [{"name": "A", "url": "rtsp://1.2.3.4/x"}],
            }
        )


def test_config_rejects_bad_opacity() -> None:
    with pytest.raises(ConfigError, match="weather_opacity"):
        parse_config(
            {
                "display": {"weather_opacity": 0.05},
                "cameras": [{"name": "A", "url": "rtsp://1.2.3.4/x"}],
            }
        )
