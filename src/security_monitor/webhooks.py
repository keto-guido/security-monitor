"""Incoming HTTP webhooks (HA-style alerts) and outgoing event POSTs."""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from security_monitor.home_assistant import DoorState

WEBHOOK_PORT_CHOICES: tuple[int, ...] = (8765, 8088, 9000, 9123, 18765)
WEBHOOK_PULSE_CHOICES: tuple[float, ...] = (3.0, 5.0, 8.0, 12.0, 20.0, 30.0)
WEBHOOK_EVENT_CHOICES: tuple[str, ...] = (
    "person",
    "encroachment",
    "ha_door",
    "webhook",
    "capture",
)
DEFAULT_OUTGOING_EVENTS: tuple[str, ...] = (
    "person",
    "encroachment",
    "ha_door",
    "capture",
)
WEBHOOK_ACCENT = (48, 150, 220)
_PATH_RE = re.compile(r"[^a-z0-9._-]+")
_OPEN_WORDS = frozenset(
    {
        "on",
        "open",
        "unlocked",
        "true",
        "1",
        "active",
        "triggered",
        "trigger",
        "pulse",
        "alarm",
        "alert",
    }
)
_CLOSED_WORDS = frozenset(
    {
        "off",
        "closed",
        "locked",
        "false",
        "0",
        "inactive",
        "clear",
        "idle",
        "ok",
        "close",
    }
)
_PULSE_WORDS = frozenset({"trigger", "pulse", "alarm", "alert"})


def slugify_path(value: str) -> str:
    text = (value or "").strip().lower().replace(" ", "_")
    text = _PATH_RE.sub("-", text).strip("-._")
    return text[:64]


def webhook_entity_id(path: str) -> str:
    return f"webhook.{slugify_path(path)}"


def unique_webhook_path(existing: list[str], desired: str) -> str:
    base = slugify_path(desired) or "hook"
    taken = {slugify_path(p) for p in existing}
    if base not in taken:
        return base
    for index in range(2, 100):
        candidate = f"{base}-{index}"
        if candidate not in taken:
            return candidate
    return f"{base}-{int(time.time())}"


@dataclass
class WebhookMapping:
    """Incoming path → the same notify options as an HA door mapping."""

    path: str
    label: str = ""
    camera: str = ""
    notify_hud: bool = True
    notify_popup: bool = True
    notify_highlight: bool = True
    notify_autofocus: bool = True
    notify_sound: bool = True

    @property
    def slug(self) -> str:
        return slugify_path(self.path)

    @property
    def entity_id(self) -> str:
        return webhook_entity_id(self.slug)

    @property
    def display_label(self) -> str:
        return (self.label or self.slug or self.path).strip()


@dataclass
class OutgoingWebhook:
    url: str
    events: tuple[str, ...] = DEFAULT_OUTGOING_EVENTS
    secret: str = ""
    enabled: bool = True

    def accepts(self, event: str) -> bool:
        if not self.enabled:
            return False
        wanted = {e.strip().lower() for e in self.events}
        return event.strip().lower() in wanted


@dataclass
class WebhookSnapshot:
    ok: bool = False
    listening: bool = False
    error: str = ""
    host: str = ""
    port: int = 0
    doors: list[DoorState] = field(default_factory=list)
    last_path: str = ""
    last_at: float = 0.0

    def status_line(self) -> str:
        if self.error and not self.listening:
            return f"Webhooks: {self.error[:48]}"
        if self.listening:
            n = sum(1 for d in self.doors if d.open)
            extra = f" · {n} active" if n else ""
            return f"Listening :{self.port}{extra}"
        return "Webhooks: off"

    def receive_url(self, path: str = "") -> str:
        if not self.listening:
            return ""
        host = self.host if self.host not in {"0.0.0.0", "::", ""} else "127.0.0.1"
        slug = slugify_path(path)
        suffix = f"/{slug}" if slug else ""
        return f"http://{host}:{self.port}/webhook{suffix}"


def webhook_mapping_to_dict(item: WebhookMapping) -> dict[str, Any]:
    data: dict[str, Any] = {"path": item.slug}
    if item.label:
        data["label"] = item.label
    if item.camera:
        data["camera"] = item.camera
    data["notify_hud"] = bool(item.notify_hud)
    data["notify_popup"] = bool(item.notify_popup)
    data["notify_highlight"] = bool(item.notify_highlight)
    data["notify_autofocus"] = bool(item.notify_autofocus)
    data["notify_sound"] = bool(item.notify_sound)
    return data


def outgoing_webhook_to_dict(item: OutgoingWebhook) -> dict[str, Any]:
    data: dict[str, Any] = {"url": item.url}
    data["events"] = list(item.events)
    if item.secret:
        data["secret"] = item.secret
    data["enabled"] = bool(item.enabled)
    return data


def _mapping_bool(item: dict[str, Any], key: str, default: bool) -> bool:
    if key not in item:
        return default
    value = item[key]
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def parse_incoming_webhooks(raw: Any) -> list[WebhookMapping]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("webhook_incoming must be a list")
    out: list[WebhookMapping] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"webhook_incoming[{index}] must be a mapping")
        path = slugify_path(str(item.get("path") or item.get("id") or item.get("name") or ""))
        if not path:
            raise ValueError(f"webhook_incoming[{index}].path is required")
        if path in seen:
            continue
        seen.add(path)
        out.append(
            WebhookMapping(
                path=path,
                label=str(item.get("label") or item.get("name") or "").strip(),
                camera=str(item.get("camera") or "").strip(),
                notify_hud=_mapping_bool(item, "notify_hud", True),
                notify_popup=_mapping_bool(item, "notify_popup", True),
                notify_highlight=_mapping_bool(item, "notify_highlight", True),
                notify_autofocus=_mapping_bool(item, "notify_autofocus", True),
                notify_sound=_mapping_bool(item, "notify_sound", True),
            )
        )
    return out


def parse_outgoing_webhooks(raw: Any) -> list[OutgoingWebhook]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("webhook_outgoing must be a list")
    out: list[OutgoingWebhook] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if isinstance(item, str):
            url = item.strip()
            events = DEFAULT_OUTGOING_EVENTS
            secret = ""
            enabled = True
        elif isinstance(item, dict):
            url = str(item.get("url") or item.get("endpoint") or "").strip()
            events_raw = item.get("events", DEFAULT_OUTGOING_EVENTS)
            if isinstance(events_raw, str):
                parts = [p.strip().lower() for p in events_raw.split(",") if p.strip()]
            elif isinstance(events_raw, (list, tuple)):
                parts = [str(p).strip().lower() for p in events_raw if str(p).strip()]
            else:
                raise ValueError(f"webhook_outgoing[{index}].events must be a list or string")
            allowed = set(WEBHOOK_EVENT_CHOICES)
            events = tuple(p for p in parts if p in allowed) or DEFAULT_OUTGOING_EVENTS
            secret = str(item.get("secret") or item.get("token") or "").strip()
            enabled = _mapping_bool(item, "enabled", True)
        else:
            raise ValueError(f"webhook_outgoing[{index}] must be a URL or mapping")
        if not url:
            raise ValueError(f"webhook_outgoing[{index}].url is required")
        key = url.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(OutgoingWebhook(url=url, events=tuple(events), secret=secret, enabled=enabled))
    return out


def parse_webhook_action(payload: Any, query: dict[str, list[str]] | None = None) -> tuple[str, bool, bool]:
    """Return (state_label, is_open, is_pulse) from JSON body and query string."""
    query = query or {}
    text = ""
    if isinstance(payload, dict):
        for key in ("state", "action", "event", "status", "value"):
            if payload.get(key) is not None:
                text = str(payload.get(key)).strip().lower()
                break
        if not text and payload.get("open") is not None:
            text = "open" if bool(payload.get("open")) else "closed"
    elif payload not in (None, ""):
        text = str(payload).strip().lower()
    if not text:
        for key in ("state", "action", "event"):
            values = query.get(key) or []
            if values:
                text = str(values[0]).strip().lower()
                break
    if not text:
        return "trigger", True, True
    if text in _PULSE_WORDS:
        return text, True, True
    if text in _CLOSED_WORDS:
        return text, False, False
    if text in _OPEN_WORDS:
        return text, True, False
    return text, True, True


def mapping_to_door(
    mapping: WebhookMapping,
    *,
    open: bool,
    state: str,
    now: float,
) -> DoorState:
    return DoorState(
        entity_id=mapping.entity_id,
        label=mapping.display_label,
        camera=mapping.camera,
        open=bool(open),
        state=state,
        changed_at=time.time(),
        local_changed_at=now,
        last_opened_at=now if open else 0.0,
        notify_hud=mapping.notify_hud,
        notify_popup=mapping.notify_popup,
        notify_highlight=mapping.notify_highlight,
        notify_autofocus=mapping.notify_autofocus,
        notify_sound=mapping.notify_sound,
    )


def webhook_open_edges(
    previous: dict[str, bool],
    doors: list[DoorState],
) -> tuple[list[DoorState], list[DoorState], dict[str, bool]]:
    """Rising edges include the first explicit open (a POST is the event)."""
    opened: list[DoorState] = []
    closed: list[DoorState] = []
    nxt = dict(previous)
    for door in doors:
        key = door.entity_id.lower()
        was = previous.get(key, False)
        if door.open and not was:
            opened.append(door)
        elif (not door.open) and was:
            closed.append(door)
        nxt[key] = bool(door.open)
    return opened, closed, nxt


def toggle_outgoing_event(events: tuple[str, ...], event: str) -> tuple[str, ...]:
    event = event.strip().lower()
    if event not in WEBHOOK_EVENT_CHOICES:
        return events
    current = [e for e in events if e in WEBHOOK_EVENT_CHOICES]
    if event in current:
        current = [e for e in current if e != event]
    else:
        current.append(event)
    order = {name: i for i, name in enumerate(WEBHOOK_EVENT_CHOICES)}
    current.sort(key=lambda name: order.get(name, 99))
    return tuple(current)


def fire_outgoing_webhooks(
    targets: list[OutgoingWebhook],
    event: str,
    payload: dict[str, Any],
    *,
    timeout: float = 3.0,
) -> None:
    """POST JSON to matching targets on a daemon thread (best-effort)."""
    event = (event or "").strip().lower()
    matching = [t for t in targets if t.accepts(event) and t.url]
    if not matching:
        return
    body = dict(payload)
    body.setdefault("source", "security-monitor")
    body["event"] = event
    body.setdefault("ts", time.time())

    def _run() -> None:
        data = json.dumps(body).encode("utf-8")
        for target in matching:
            try:
                headers = {"Content-Type": "application/json", "User-Agent": "security-monitor"}
                if target.secret:
                    headers["Authorization"] = f"Bearer {target.secret}"
                    headers["X-Webhook-Secret"] = target.secret
                req = Request(target.url, data=data, headers=headers, method="POST")
                with urlopen(req, timeout=max(0.5, float(timeout))) as resp:
                    resp.read(256)
            except (HTTPError, URLError, TimeoutError, OSError, ValueError):
                continue

    threading.Thread(target=_run, name="webhook-out", daemon=True).start()


class WebhookService:
    """Background HTTP listener that exposes HA-style DoorState snapshots."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._enabled = False
        self._host = "0.0.0.0"
        self._port = 8765
        self._secret = ""
        self._pulse_seconds = 8.0
        self._mappings: dict[str, WebhookMapping] = {}
        self._states: dict[str, DoorState] = {}
        self._pulse_until: dict[str, float] = {}
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._error = ""
        self._bound_host = ""
        self._bound_port = 0
        self._last_path = ""
        self._last_at = 0.0

    def configure(
        self,
        *,
        enabled: bool,
        host: str = "0.0.0.0",
        port: int = 8765,
        secret: str = "",
        pulse_seconds: float = 8.0,
        mappings: list[WebhookMapping] | None = None,
    ) -> None:
        host = (host or "0.0.0.0").strip() or "0.0.0.0"
        port = int(port or 8765)
        secret = (secret or "").strip()
        pulse = max(0.05, min(120.0, float(pulse_seconds or 8.0)))
        maps = {m.slug: m for m in (mappings or []) if m.slug}
        with self._lock:
            bind_changed = (
                enabled != self._enabled
                or host != self._host
                or port != self._port
            )
            self._enabled = bool(enabled)
            self._host = host
            self._port = port
            self._secret = secret
            self._pulse_seconds = pulse
            self._mappings = maps
            keep = {m.entity_id.lower() for m in maps.values()}
            self._states = {k: v for k, v in self._states.items() if k in keep}
            self._pulse_until = {k: v for k, v in self._pulse_until.items() if k in keep}
        if not enabled:
            self._stop_server()
            with self._lock:
                self._error = ""
            return
        if bind_changed or self._httpd is None:
            self._restart_server()

    def stop(self) -> None:
        self._stop_server()

    @property
    def snapshot(self) -> WebhookSnapshot:
        now = time.monotonic()
        with self._lock:
            self._expire_pulses_locked(now)
            doors = list(self._states.values())
            listening = self._httpd is not None
            return WebhookSnapshot(
                ok=listening or not self._enabled,
                listening=listening,
                error=self._error,
                host=self._bound_host or self._host,
                port=self._bound_port or (self._port if listening else 0),
                doors=doors,
                last_path=self._last_path,
                last_at=self._last_at,
            )

    def apply(
        self,
        path: str,
        *,
        payload: Any = None,
        query: dict[str, list[str]] | None = None,
    ) -> tuple[bool, str, DoorState | None]:
        slug = slugify_path(path)
        with self._lock:
            mapping = self._mappings.get(slug)
            if mapping is None:
                return False, f"unknown webhook {slug!r}", None
            state, is_open, is_pulse = parse_webhook_action(payload, query)
            now = time.monotonic()
            door = mapping_to_door(mapping, open=is_open, state=state, now=now)
            key = door.entity_id.lower()
            previous = self._states.get(key)
            if previous is not None and is_open:
                door.last_opened_at = previous.last_opened_at or now
            self._states[key] = door
            if is_pulse and is_open:
                self._pulse_until[key] = now + self._pulse_seconds
            elif not is_open:
                self._pulse_until.pop(key, None)
            self._last_path = slug
            self._last_at = now
        return True, "ok", door

    def _expire_pulses_locked(self, now: float) -> None:
        expired = [key for key, until in self._pulse_until.items() if until <= now]
        for key in expired:
            self._pulse_until.pop(key, None)
            door = self._states.get(key)
            if door is None or not door.open:
                continue
            self._states[key] = DoorState(
                entity_id=door.entity_id,
                label=door.label,
                camera=door.camera,
                open=False,
                state="idle",
                changed_at=time.time(),
                local_changed_at=now,
                last_opened_at=door.last_opened_at,
                notify_hud=door.notify_hud,
                notify_popup=door.notify_popup,
                notify_highlight=door.notify_highlight,
                notify_autofocus=door.notify_autofocus,
                notify_sound=door.notify_sound,
            )

    def _authorized(self, headers: Any, query: dict[str, list[str]]) -> bool:
        with self._lock:
            secret = self._secret
        if not secret:
            return True
        auth = ""
        header_secret = ""
        try:
            auth = str(headers.get("Authorization") or "")
            header_secret = str(headers.get("X-Webhook-Secret") or "")
        except Exception:  # noqa: BLE001
            auth = ""
        token = ""
        if auth.lower().startswith("bearer "):
            token = auth[7:].strip()
        if header_secret and header_secret == secret:
            return True
        if token and token == secret:
            return True
        qtok = (query.get("token") or query.get("secret") or [""])[0]
        return bool(qtok) and qtok == secret

    def _restart_server(self) -> None:
        self._stop_server()
        with self._lock:
            host, port = self._host, self._port
        service = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, _fmt: str, *_args: object) -> None:
                return

            def _query(self) -> dict[str, list[str]]:
                return parse_qs(urlparse(self.path).query)

            def _json(self, code: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _read_payload(self) -> Any:
                length = int(self.headers.get("Content-Length") or 0)
                if length <= 0:
                    return None
                raw = self.rfile.read(min(length, 65536))
                if not raw:
                    return None
                try:
                    return json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    return raw.decode("utf-8", errors="replace")

            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path in {"/", "/health", "/webhook"}:
                    snap = service.snapshot
                    self._json(
                        200,
                        {
                            "ok": True,
                            "listening": snap.listening,
                            "port": snap.port,
                        },
                    )
                    return
                self._json(404, {"ok": False, "error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                query = parse_qs(parsed.query)
                if not service._authorized(self.headers, query):
                    self._json(401, {"ok": False, "error": "unauthorized"})
                    return
                parts = [p for p in parsed.path.split("/") if p]
                slug = ""
                if len(parts) >= 2 and parts[0] in {"webhook", "hooks", "hook"}:
                    slug = slugify_path(parts[1])
                elif len(parts) == 1 and parts[0] not in {"webhook", "hooks", "health"}:
                    slug = slugify_path(parts[0])
                if not slug:
                    self._json(400, {"ok": False, "error": "missing webhook path"})
                    return
                payload = self._read_payload()
                ok, message, door = service.apply(slug, payload=payload, query=query)
                if not ok:
                    self._json(404, {"ok": False, "error": message})
                    return
                self._json(
                    200,
                    {
                        "ok": True,
                        "path": slug,
                        "state": door.state if door else "",
                        "open": bool(door.open) if door else False,
                    },
                )

        try:
            httpd = ThreadingHTTPServer((host, port), _Handler)
            httpd.daemon_threads = True
        except OSError as exc:
            with self._lock:
                self._error = f"bind {host}:{port} failed ({exc})"
                self._httpd = None
                self._bound_host = ""
                self._bound_port = 0
            print(f"Webhooks: {self._error}")
            return
        thread = threading.Thread(target=httpd.serve_forever, name="webhooks", daemon=True)
        bound_host, bound_port = httpd.server_address[:2]
        with self._lock:
            self._httpd = httpd
            self._thread = thread
            self._error = ""
            self._bound_host = str(bound_host)
            self._bound_port = int(bound_port)
        thread.start()
        print(f"Webhooks listening on http://{bound_host}:{bound_port}/webhook/<path>")

    def _stop_server(self) -> None:
        with self._lock:
            httpd = self._httpd
            thread = self._thread
            self._httpd = None
            self._thread = None
            self._bound_host = ""
            self._bound_port = 0
        if httpd is not None:
            try:
                httpd.shutdown()
            except Exception:  # noqa: BLE001
                pass
            try:
                httpd.server_close()
            except Exception:  # noqa: BLE001
                pass
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.5)
