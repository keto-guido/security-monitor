"""Tests for Home Assistant door-sensor helpers and config."""

from __future__ import annotations

import json
import time
from unittest.mock import patch

import numpy as np
import pytest

from security_monitor.config import ConfigError, parse_config, save_display_settings
from security_monitor.home_assistant import (
    DoorState,
    HADoorMapping,
    HALightControl,
    HAPopup,
    HASnapshot,
    call_ha_service,
    cameras_highlighted_by_doors,
    default_trigger_states,
    domain_counts,
    draw_door_hud,
    draw_ha_light_panel,
    draw_ha_popups,
    fetch_door_states,
    filter_entities,
    mask_token,
    merge_camera_door_entities,
    normalize_ha_url,
    parse_door_mappings,
    parse_entity_catalog,
    parse_light_controls,
    prune_popups,
    suggested_states_for_entity,
    toggle_open_state,
)


def test_normalize_and_mask() -> None:
    assert normalize_ha_url("homeassistant.local:8123") == "http://homeassistant.local:8123"
    assert normalize_ha_url("http://ha.local:8123/") == "http://ha.local:8123"
    assert mask_token("") == "(not set)"
    assert "…" in mask_token("abcdefghijklmnopqrstuvwxyz")


def test_parse_door_mappings() -> None:
    doors = parse_door_mappings(
        [
            {"entity_id": "binary_sensor.front", "label": "Front", "camera": "Cam A"},
            {"entity_id": "binary_sensor.front"},  # duplicate ignored
            {"entity": "binary_sensor.side", "name": "Side"},
        ]
    )
    assert len(doors) == 2
    assert doors[0].camera == "Cam A"
    assert doors[1].display_label == "Side"


def test_merge_camera_door_entities() -> None:
    class Cam:
        def __init__(self, name: str, entity: str = "", label: str = "") -> None:
            self.name = name
            self.ha_door_entity = entity
            self.ha_door_label = label

    doors = [HADoorMapping(entity_id="binary_sensor.front", camera="")]
    merged = merge_camera_door_entities(
        doors,
        [Cam("Porch", "binary_sensor.front", "Front"), Cam("Gate", "binary_sensor.gate")],
    )
    assert len(merged) == 2
    assert merged[0].camera == "Porch"
    assert merged[1].camera == "Gate"


def test_cameras_highlighted_hold() -> None:
    now = 1000.0
    doors = [
        DoorState(
            entity_id="binary_sensor.a",
            label="A",
            camera="Cam A",
            open=False,
            last_opened_at=990.0,
        ),
        DoorState(
            entity_id="binary_sensor.b",
            label="B",
            camera="Cam B",
            open=True,
            last_opened_at=995.0,
        ),
    ]
    active = cameras_highlighted_by_doors(doors, hold_seconds=20.0, now=now)
    assert active["Cam A"] == "A"
    assert active["Cam B"] == "B"
    expired = cameras_highlighted_by_doors(doors, hold_seconds=5.0, now=now)
    assert "Cam A" not in expired
    assert expired["Cam B"] == "B"


def test_fetch_door_states_from_bulk() -> None:
    payload = [
        {
            "entity_id": "binary_sensor.front_door",
            "state": "on",
            "attributes": {"friendly_name": "Front Door"},
        },
        {"entity_id": "binary_sensor.other", "state": "off", "attributes": {}},
    ]

    class _Resp:
        def read(self) -> bytes:
            return json.dumps(payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    with patch("security_monitor.home_assistant.urlopen", return_value=_Resp()):
        snap = fetch_door_states(
            "http://ha.local:8123",
            "token",
            [HADoorMapping(entity_id="binary_sensor.front_door", camera="Front")],
        )
    assert snap.ok is True
    assert snap.connected is True
    assert snap.doors[0].open is True
    assert snap.doors[0].label == "Front Door"


def test_draw_door_hud_smoke() -> None:
    canvas = np.full((240, 480, 3), 40, dtype=np.uint8)
    snap = HASnapshot(
        ok=True,
        connected=True,
        doors=[
            DoorState(
                entity_id="binary_sensor.front",
                label="Front",
                camera="Cam",
                open=True,
            )
        ],
    )
    draw_door_hud(canvas, snap)
    assert canvas[20, 30].sum() != 40 * 3


def test_config_ha_roundtrip(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "display:\n  columns: 2\ncameras:\n  - name: A\n    url: rtsp://1.2.3.4/x\n",
        encoding="utf-8",
    )
    cfg = parse_config(
        {
            "display": {
                "ha_enabled": True,
                "ha_url": "http://192.168.1.50:8123",
                "ha_token": "secret-token-value",
                "ha_poll_seconds": 3,
                "ha_show_hud": True,
                "ha_highlight": False,
                "ha_autofocus": True,
                "ha_alarm_sound": False,
                "ha_hold_seconds": 15,
                "ha_doors": [
                    {
                        "entity_id": "binary_sensor.front_door",
                        "label": "Front",
                        "camera": "A",
                        "open_states": ["on"],
                        "notify_hud": True,
                        "notify_popup": True,
                        "notify_highlight": False,
                        "notify_autofocus": True,
                        "notify_sound": False,
                    }
                ],
                "ha_lights": [
                    {"entity_id": "light.kitchen", "label": "Kitchen"},
                    "switch.porch",
                ],
                "ha_popup_seconds": 8,
                "ha_panel_enabled": True,
            },
            "cameras": [
                {
                    "name": "A",
                    "url": "rtsp://1.2.3.4/x",
                    "ha_door_entity": "binary_sensor.side",
                    "ha_door_label": "Side",
                }
            ],
        },
        path=path,
    )
    assert cfg.display.ha_enabled is True
    assert cfg.display.ha_url == "http://192.168.1.50:8123"
    assert cfg.display.ha_token == "secret-token-value"
    assert cfg.display.ha_highlight is False
    assert cfg.display.ha_doors[0].entity_id == "binary_sensor.front_door"
    assert cfg.display.ha_doors[0].notify_highlight is False
    assert cfg.display.ha_doors[0].notify_popup is True
    assert cfg.display.ha_doors[0].notify_sound is False
    assert cfg.display.ha_popup_seconds == pytest.approx(8.0)
    assert cfg.display.ha_panel_enabled is True
    assert len(cfg.display.ha_lights) == 2
    assert cfg.display.ha_lights[0].entity_id == "light.kitchen"
    assert cfg.display.ha_lights[1].entity_id == "switch.porch"
    assert cfg.cameras[0].ha_door_entity == "binary_sensor.side"
    assert save_display_settings(cfg) == path
    text = path.read_text(encoding="utf-8")
    assert "ha_enabled: true" in text
    assert "binary_sensor.front_door" in text
    assert "notify_sound: false" in text
    assert "notify_popup: true" in text
    assert "light.kitchen" in text
    assert "ha_door_entity: binary_sensor.side" in text


def test_popup_and_panel_draw_smoke() -> None:
    canvas = np.full((360, 640, 3), 30, dtype=np.uint8)
    popups = [HAPopup(message="Front door", until=time.monotonic() + 5)]
    draw_ha_popups(canvas, popups)
    assert canvas[30, 320].sum() != 30 * 3
    assert prune_popups(popups, now=time.monotonic() + 10) == []
    lights = [HALightControl(entity_id="light.kitchen", label="Kitchen")]
    hits = draw_ha_light_panel(
        canvas, lights, {"light.kitchen": "on"}, open_amount=1.0, enabled=True
    )
    assert any(a == "ha_panel_toggle" for a, *_ in hits)
    assert any(a.startswith("ha_light:") for a, *_ in hits)


def test_call_ha_service_posts(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _fake_request(base, token, path, *, timeout=4.0, method="GET", payload=None):
        seen["path"] = path
        seen["method"] = method
        seen["payload"] = payload
        return []

    monkeypatch.setattr("security_monitor.home_assistant._ha_request", _fake_request)
    err = call_ha_service("http://ha.local:8123", "tok", "light", "toggle", "light.kitchen")
    assert err == ""
    assert seen["method"] == "POST"
    assert seen["path"] == "/api/services/light/toggle"
    assert seen["payload"] == {"entity_id": "light.kitchen"}


def test_entity_catalog_helpers() -> None:
    rows = [
        {
            "entity_id": "binary_sensor.front_door",
            "state": "on",
            "attributes": {"friendly_name": "Front Door", "device_class": "door"},
        },
        {
            "entity_id": "cover.garage",
            "state": "closed",
            "attributes": {"friendly_name": "Garage"},
        },
        {
            "entity_id": "sensor.temp",
            "state": "21.5",
            "attributes": {"friendly_name": "Temp", "unit_of_measurement": "°C"},
        },
    ]
    entities = parse_entity_catalog(rows)
    assert len(entities) == 3
    doors = filter_entities(entities, domain="binary_sensor")
    assert len(doors) == 1
    assert doors[0].display_name == "Front Door"
    counts = dict(domain_counts(entities))
    assert counts["binary_sensor"] == 1
    assert counts["cover"] == 1
    assert default_trigger_states(doors[0]) == ("on",)
    states = suggested_states_for_entity(doors[0])
    assert "on" in states and "off" in states
    assert toggle_open_state(("on",), "off") == ("on", "off")
    assert toggle_open_state(("on", "off"), "on") == ("off",)


def test_cameras_highlighted_respects_notify_flags() -> None:
    doors = [
        DoorState(
            entity_id="a",
            label="A",
            camera="Cam A",
            open=True,
            last_opened_at=1000.0,
            notify_highlight=False,
            notify_autofocus=True,
        ),
        DoorState(
            entity_id="b",
            label="B",
            camera="Cam B",
            open=True,
            last_opened_at=1000.0,
            notify_highlight=True,
            notify_autofocus=False,
        ),
    ]
    hl = cameras_highlighted_by_doors(doors, hold_seconds=20, now=1001.0, require_highlight=True)
    af = cameras_highlighted_by_doors(doors, hold_seconds=20, now=1001.0, require_autofocus=True)
    assert "Cam A" not in hl
    assert hl["Cam B"] == "B"
    assert af["Cam A"] == "A"
    assert "Cam B" not in af


def test_config_rejects_bad_ha_poll() -> None:
    with pytest.raises(ConfigError, match="ha_poll_seconds"):
        parse_config(
            {
                "display": {"ha_poll_seconds": 0.1},
                "cameras": [{"name": "A", "url": "rtsp://1.2.3.4/x"}],
            }
        )
