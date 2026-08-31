"""Incoming / outgoing webhook helpers."""

from __future__ import annotations

import json
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from security_monitor.config import ConfigError, parse_config, save_display_settings
from security_monitor.webhooks import (
    OutgoingWebhook,
    WebhookMapping,
    WebhookService,
    fire_outgoing_webhooks,
    format_webhook_receive_url,
    format_webhook_sendto_copy,
    resolve_webhook_public_host,
    webhook_bind_blocks_remote,
    parse_incoming_webhooks,
    parse_outgoing_webhooks,
    parse_webhook_action,
    slugify_path,
    toggle_outgoing_event,
    unique_webhook_path,
    webhook_curl_example,
    webhook_entity_id,
    webhook_json_body,
    webhook_open_edges,
    webhook_setup_hint,
)
from security_monitor.home_assistant import DoorState


def test_slugify_and_unique_path() -> None:
    assert slugify_path("Front Door") == "front_door"
    assert unique_webhook_path(["front_door"], "Front Door") == "front_door-2"
    assert webhook_entity_id("gate") == "webhook.gate"


def test_parse_incoming_and_outgoing() -> None:
    incoming = parse_incoming_webhooks(
        [
            {"path": "front_door", "label": "Front", "camera": "Porch"},
            {"path": "front_door"},
            {"id": "gate", "notify_sound": False},
        ]
    )
    assert len(incoming) == 2
    assert incoming[0].camera == "Porch"
    assert incoming[1].notify_sound is False
    outgoing = parse_outgoing_webhooks(
        [
            "http://example.local/hook",
            {"url": "http://example.local/hook"},
            {"url": "http://other.local/x", "events": ["person", "nope"], "secret": "s"},
        ]
    )
    assert len(outgoing) == 2
    assert outgoing[1].secret == "s"
    assert outgoing[1].events == ("person",)


def test_parse_webhook_action_defaults_to_pulse() -> None:
    state, is_open, is_pulse = parse_webhook_action(None)
    assert is_open is True and is_pulse is True
    assert parse_webhook_action({"state": "closed"}) == ("closed", False, False)
    assert parse_webhook_action({"action": "open"})[1:] == (True, False)
    assert parse_webhook_action({"event": "trigger"})[2] is True


def test_webhook_open_edges_first_post_is_rising() -> None:
    door = DoorState(
        entity_id="webhook.front",
        label="Front",
        camera="Porch",
        open=True,
        state="trigger",
    )
    opened, closed, nxt = webhook_open_edges({}, [door])
    assert [d.entity_id for d in opened] == ["webhook.front"]
    assert closed == []
    opened2, closed2, _nxt = webhook_open_edges(nxt, [door])
    assert opened2 == []
    closed_door = DoorState(
        entity_id="webhook.front", label="Front", camera="Porch", open=False, state="idle"
    )
    opened3, closed3, _ = webhook_open_edges(nxt, [closed_door])
    assert opened3 == []
    assert [d.entity_id for d in closed3] == ["webhook.front"]


def test_toggle_outgoing_event() -> None:
    events = toggle_outgoing_event(("person",), "encroachment")
    assert "encroachment" in events and "person" in events
    events = toggle_outgoing_event(events, "person")
    assert "person" not in events


def test_config_webhook_roundtrip(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    cfg = parse_config(
        {
            "display": {
                "webhook_enabled": True,
                "webhook_listen_port": 9123,
                "webhook_secret": "abc",
                "webhook_incoming": [
                    {"path": "bell", "label": "Doorbell", "camera": "Front Door"}
                ],
                "webhook_outgoing": [
                    {"url": "http://127.0.0.1:9/hook", "events": ["person", "capture"]}
                ],
            },
            "cameras": [{"name": "Front Door", "url": "rtsp://1.2.3.4/x"}],
        },
        path=path,
    )
    assert cfg.display.webhook_enabled is True
    assert cfg.display.webhook_listen_port == 9123
    assert cfg.display.webhook_incoming[0].path == "bell"
    assert cfg.display.webhook_outgoing[0].events == ("person", "capture")
    assert save_display_settings(cfg) == path
    text = path.read_text(encoding="utf-8")
    assert "webhook_enabled: true" in text
    assert "bell" in text
    with pytest.raises(ConfigError):
        parse_config({"display": {"webhook_incoming": "nope"}, "cameras": []})


def test_format_webhook_receive_url_without_listener(monkeypatch) -> None:
    monkeypatch.setattr("security_monitor.webhooks.guess_lan_ip", lambda: "192.168.10.116")
    url = format_webhook_receive_url("0.0.0.0", 8765, "front_door", secret="tok")
    assert url == "http://192.168.10.116:8765/webhook/front_door?token=tok"
    url_local_bind = format_webhook_receive_url("127.0.0.1", 9123, "porch", secret="s")
    assert url_local_bind.startswith("http://192.168.10.116:9123/webhook/porch")
    assert format_webhook_receive_url("192.168.1.10", 9123, "") == "http://192.168.10.116:9123/webhook"


def test_resolve_webhook_public_host_prefers_lan(monkeypatch) -> None:
    monkeypatch.setattr("security_monitor.webhooks.guess_lan_ip", lambda: "192.168.10.116")
    assert resolve_webhook_public_host("127.0.0.1") == "192.168.10.116"
    assert resolve_webhook_public_host("0.0.0.0") == "192.168.10.116"
    assert webhook_bind_blocks_remote("127.0.0.1") is True
    assert webhook_bind_blocks_remote("0.0.0.0") is False


def test_webhook_json_body() -> None:
    assert webhook_json_body() == '{"state": "open"}'
    assert webhook_json_body(state="closed") == '{"state": "closed"}'
    assert "-d" not in webhook_json_body()


def test_format_webhook_sendto_copy() -> None:
    text = format_webhook_sendto_copy(
        "http://192.168.1.5:9123/webhook/porch?token=abc"
    )
    lines = text.splitlines()
    assert lines[0] == "http://192.168.1.5:9123/webhook/porch?token=abc"
    assert lines[1] == '{"state": "open"}'


def test_receive_url_includes_secret_token() -> None:
    service = WebhookService()
    service.configure(enabled=True, host="127.0.0.1", port=0, secret="s3cret")
    try:
        snap = service.snapshot
        assert snap.listening
        url = snap.receive_url("front_door", secret="s3cret")
        assert f":{snap.port}/webhook/front_door?" in url
        assert "token=s3cret" in url
        assert not url.startswith("http://127.0.0.1:")
    finally:
        service.stop()


def test_webhook_curl_example_and_setup_hint() -> None:
    url = "http://127.0.0.1:8765/webhook/front?token=abc"
    curl = webhook_curl_example(url)
    assert "curl -X POST" in curl
    assert url in curl
    assert '"state"' in curl and "open" in curl
    assert webhook_setup_hint(enabled=False, listening=False, mapping_count=0).startswith("Next:")
    assert "alert route" in webhook_setup_hint(enabled=True, listening=True, mapping_count=0).lower()
    assert "copy" in webhook_setup_hint(enabled=True, listening=True, mapping_count=2).lower()


def test_webhook_service_http_roundtrip() -> None:
    service = WebhookService()
    mapping = WebhookMapping(path="front", label="Front", camera="Porch")
    service.configure(
        enabled=True,
        host="127.0.0.1",
        port=0,
        secret="s3cret",
        pulse_seconds=8.0,
        mappings=[mapping],
    )
    try:
        snap = service.snapshot
        assert snap.listening
        port = snap.port
        assert port > 0
        req = Request(
            f"http://127.0.0.1:{port}/webhook/front",
            data=b'{"state":"open"}',
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer s3cret",
            },
            method="POST",
        )
        with urlopen(req, timeout=2) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["ok"] is True
        assert body["open"] is True
        doors = service.snapshot.doors
        assert doors and doors[0].open is True
        assert doors[0].label == "Front"
        bad = Request(
            f"http://127.0.0.1:{port}/webhook/front",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as exc:
            urlopen(bad, timeout=2)
        assert exc.value.code == 401
    finally:
        service.stop()


def test_webhook_pulse_expires() -> None:
    service = WebhookService()
    service.configure(
        enabled=False,
        mappings=[WebhookMapping(path="bell", label="Bell")],
        pulse_seconds=0.05,
    )
    ok, _msg, door = service.apply("bell", payload={"state": "trigger"})
    assert ok and door is not None and door.open is True
    time.sleep(0.08)
    later = service.snapshot.doors
    assert later and later[0].open is False


def test_fire_outgoing_posts(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class _Resp:
        def read(self, _n=None):
            return b"ok"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def _urlopen(req, timeout=3.0):
        seen["url"] = req.full_url
        seen["auth"] = req.get_header("Authorization")
        seen["body"] = json.loads(req.data.decode("utf-8"))
        return _Resp()

    monkeypatch.setattr("security_monitor.webhooks.urlopen", _urlopen)
    target = OutgoingWebhook(url="http://127.0.0.1:9/hook", events=("person",), secret="tok")
    fire_outgoing_webhooks([target], "person", {"camera": "Gate"})
    deadline = time.monotonic() + 2.0
    while "url" not in seen and time.monotonic() < deadline:
        time.sleep(0.01)
    assert seen["url"] == "http://127.0.0.1:9/hook"
    assert seen["auth"] == "Bearer tok"
    assert seen["body"]["event"] == "person"
    assert seen["body"]["camera"] == "Gate"
