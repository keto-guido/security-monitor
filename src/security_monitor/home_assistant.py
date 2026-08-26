"""Local Home Assistant door / window binary-sensor integration."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import cv2
import numpy as np

from security_monitor.overlay import draw_text, shade_round_rect

HA_POLL_CHOICES: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0, 10.0)
HA_HOLD_CHOICES: tuple[float, ...] = (0.0, 5.0, 8.0, 12.0, 20.0)
HA_POPUP_CHOICES: tuple[float, ...] = (2.0, 3.0, 5.0, 8.0)
DEFAULT_OPEN_STATES: tuple[str, ...] = ("on", "open", "unlocked")
HA_TOAST_MAX_SECONDS = 8.0
# Domains that commonly map to doors / openings / occupancy alerts.
HA_BROWSE_DOMAINS: tuple[str, ...] = (
    "binary_sensor",
    "cover",
    "lock",
    "switch",
    "input_boolean",
    "sensor",
    "light",
)
DOMAIN_STATE_HINTS: dict[str, tuple[str, ...]] = {
    "binary_sensor": ("on", "off"),
    "cover": ("open", "opening", "closed", "closing", "stopped"),
    "lock": ("unlocked", "locked", "locking", "unlocking", "jammed"),
    "switch": ("on", "off"),
    "input_boolean": ("on", "off"),
    "light": ("on", "off"),
    "sensor": (),
}
PANEL_WIDTH = 220
PANEL_TAB_WIDTH = 28


@dataclass
class HADoorMapping:
    """Maps a Home Assistant entity to optional camera + notification options."""

    entity_id: str
    label: str = ""
    camera: str = ""
    open_states: tuple[str, ...] = DEFAULT_OPEN_STATES
    notify_hud: bool = True
    notify_popup: bool = True
    notify_highlight: bool = True
    notify_autofocus: bool = True
    notify_sound: bool = True

    @property
    def display_label(self) -> str:
        return (self.label or self.entity_id.split(".", 1)[-1] or self.entity_id).strip()

    @property
    def domain(self) -> str:
        return self.entity_id.split(".", 1)[0].lower() if "." in self.entity_id else ""

    def trigger_label(self) -> str:
        return "/".join(self.open_states) if self.open_states else "—"


@dataclass
class HALightControl:
    """A light (or switch) shown on the HA slide-out panel."""

    entity_id: str
    label: str = ""

    @property
    def display_label(self) -> str:
        return (self.label or self.entity_id.split(".", 1)[-1] or self.entity_id).strip()

    @property
    def domain(self) -> str:
        return self.entity_id.split(".", 1)[0].lower() if "." in self.entity_id else "light"


@dataclass
class HAPopup:
    """Temporary on-mosaic toast (not tied to a camera tile)."""

    message: str
    until: float
    accent: tuple[int, int, int] = (40, 120, 255)
    entity_id: str = ""


@dataclass
class HAEntityInfo:
    """One entity from HA `/api/states` for the browse UI."""

    entity_id: str
    state: str = ""
    friendly_name: str = ""
    domain: str = ""
    device_class: str = ""
    unit: str = ""

    @property
    def display_name(self) -> str:
        return (self.friendly_name or self.entity_id.split(".", 1)[-1] or self.entity_id).strip()

    def menu_label(self) -> str:
        bits = [self.display_name[:34], f"[{self.state or '?'}]"]
        if self.device_class:
            bits.append(self.device_class)
        return "  ".join(bits)[:64]


@dataclass
class DoorState:
    entity_id: str
    label: str
    camera: str
    open: bool
    state: str = ""
    changed_at: float = 0.0  # wall time of last poll that saw this entity
    local_changed_at: float = 0.0  # monotonic of last open/closed flip
    last_opened_at: float = 0.0  # monotonic of most recent closed→open edge
    notify_hud: bool = True
    notify_popup: bool = True
    notify_highlight: bool = True
    notify_autofocus: bool = True
    notify_sound: bool = True


@dataclass
class HASnapshot:
    ok: bool = False
    connected: bool = False
    error: str = ""
    doors: list[DoorState] = field(default_factory=list)
    updated_at: float = 0.0

    @property
    def open_doors(self) -> list[DoorState]:
        return [d for d in self.doors if d.open]

    def status_line(self) -> str:
        if not self.connected and self.error:
            return f"HA: {self.error[:48]}"
        if not self.ok and self.error:
            return f"HA: {self.error[:48]}"
        opens = self.open_doors
        if opens:
            names = ", ".join(d.label for d in opens[:3])
            extra = f" +{len(opens) - 3}" if len(opens) > 3 else ""
            return f"DOOR OPEN: {names}{extra}"
        if self.connected:
            return "HA: connected · all doors closed"
        return "HA: idle"


def normalize_ha_url(url: str) -> str:
    text = (url or "").strip().rstrip("/")
    if not text:
        return ""
    parsed = urlparse(text)
    # urlparse("host:8123") treats "host" as the scheme — require a real scheme.
    if not parsed.scheme or parsed.scheme.lower() not in {"http", "https"}:
        text = "http://" + text.lstrip("/")
    return text.rstrip("/")


def mask_token(token: str) -> str:
    value = (token or "").strip()
    if not value:
        return "(not set)"
    if len(value) <= 8:
        return "••••"
    return f"{value[:4]}…{value[-4:]}"


def door_mapping_to_dict(door: HADoorMapping) -> dict[str, Any]:
    item: dict[str, Any] = {"entity_id": door.entity_id}
    if door.label:
        item["label"] = door.label
    if door.camera:
        item["camera"] = door.camera
    if door.open_states and tuple(door.open_states) != DEFAULT_OPEN_STATES:
        item["open_states"] = list(door.open_states)
    item["notify_hud"] = bool(door.notify_hud)
    item["notify_popup"] = bool(door.notify_popup)
    item["notify_highlight"] = bool(door.notify_highlight)
    item["notify_autofocus"] = bool(door.notify_autofocus)
    item["notify_sound"] = bool(door.notify_sound)
    return item


def light_control_to_dict(light: HALightControl) -> dict[str, Any]:
    item: dict[str, Any] = {"entity_id": light.entity_id}
    if light.label:
        item["label"] = light.label
    return item


def parse_door_mappings(raw: Any) -> list[HADoorMapping]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("ha_doors must be a list")
    out: list[HADoorMapping] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"ha_doors[{index}] must be a mapping")
        entity = str(item.get("entity_id") or item.get("entity") or "").strip()
        if not entity:
            raise ValueError(f"ha_doors[{index}].entity_id is required")
        key = entity.lower()
        if key in seen:
            continue
        seen.add(key)
        label = str(item.get("label") or item.get("name") or "").strip()
        camera = str(item.get("camera") or "").strip()
        states_raw = item.get("open_states")
        if states_raw is None:
            open_states = DEFAULT_OPEN_STATES
        else:
            if isinstance(states_raw, str):
                parts = [p.strip() for p in states_raw.split(",") if p.strip()]
            elif isinstance(states_raw, (list, tuple)):
                parts = [str(p).strip() for p in states_raw if str(p).strip()]
            else:
                raise ValueError(f"ha_doors[{index}].open_states must be a list or string")
            if not parts:
                raise ValueError(f"ha_doors[{index}].open_states cannot be empty")
            open_states = tuple(parts)
        out.append(
            HADoorMapping(
                entity_id=entity,
                label=label,
                camera=camera,
                open_states=open_states,
                notify_hud=_mapping_bool(item, "notify_hud", True),
                notify_popup=_mapping_bool(item, "notify_popup", True),
                notify_highlight=_mapping_bool(item, "notify_highlight", True),
                notify_autofocus=_mapping_bool(item, "notify_autofocus", True),
                notify_sound=_mapping_bool(item, "notify_sound", True),
            )
        )
    return out


def parse_light_controls(raw: Any) -> list[HALightControl]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("ha_lights must be a list")
    out: list[HALightControl] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if isinstance(item, str):
            entity = item.strip()
            label = ""
        elif isinstance(item, dict):
            entity = str(item.get("entity_id") or item.get("entity") or "").strip()
            label = str(item.get("label") or item.get("name") or "").strip()
        else:
            raise ValueError(f"ha_lights[{index}] must be a mapping or entity_id string")
        if not entity:
            raise ValueError(f"ha_lights[{index}].entity_id is required")
        key = entity.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(HALightControl(entity_id=entity, label=label))
    return out


def _mapping_bool(raw: dict[str, Any], key: str, default: bool) -> bool:
    if key not in raw:
        return default
    value = raw.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def default_trigger_states(entity: HAEntityInfo | None, domain: str = "") -> tuple[str, ...]:
    """Pick sensible alert states for a newly selected entity."""
    dom = (entity.domain if entity else domain) or ""
    device = (entity.device_class if entity else "") or ""
    if dom == "binary_sensor":
        if device in {"door", "window", "garage_door", "opening", "lock"}:
            return ("on",)
        if device in {"motion", "occupancy", "presence", "moving"}:
            return ("on",)
        return ("on",)
    if dom == "cover":
        return ("open", "opening")
    if dom == "lock":
        return ("unlocked",)
    if dom in {"switch", "input_boolean"}:
        return ("on",)
    if entity and entity.state:
        return (entity.state.lower(),)
    hints = DOMAIN_STATE_HINTS.get(dom, ())
    return hints[:1] if hints else DEFAULT_OPEN_STATES


def suggested_states_for_entity(entity: HAEntityInfo | None, domain: str = "") -> list[str]:
    """States offered in the trigger-state picker."""
    dom = (entity.domain if entity else domain) or ""
    ordered: list[str] = []
    seen: set[str] = set()

    def _add(state: str) -> None:
        key = state.lower().strip()
        if not key or key in seen:
            return
        seen.add(key)
        ordered.append(key)

    for hint in DOMAIN_STATE_HINTS.get(dom, ()):
        _add(hint)
    if entity and entity.state:
        _add(entity.state)
    for extra in DEFAULT_OPEN_STATES:
        _add(extra)
    # Always allow common closed/idle counterparts so users can invert.
    for extra in ("off", "closed", "locked", "unavailable", "unknown"):
        _add(extra)
    return ordered


def toggle_open_state(current: tuple[str, ...], state: str) -> tuple[str, ...]:
    key = state.lower().strip()
    if not key:
        return current
    values = [s for s in current if s.lower() != key]
    if len(values) == len(current):
        values.append(key)
    if not values:
        values = [key]
    return tuple(values)


def merge_camera_door_entities(
    doors: list[HADoorMapping],
    cameras: list[Any],
) -> list[HADoorMapping]:
    """Fold per-camera ``ha_door_entity`` shortcuts into the mapping list."""
    by_id = {d.entity_id.lower(): d for d in doors}
    merged = list(doors)
    for cam in cameras:
        entity = str(getattr(cam, "ha_door_entity", "") or "").strip()
        if not entity:
            continue
        key = entity.lower()
        label = str(getattr(cam, "ha_door_label", "") or "").strip()
        name = str(getattr(cam, "name", "") or "").strip()
        if key in by_id:
            existing = by_id[key]
            if not existing.camera and name:
                existing.camera = name
            if not existing.label and label:
                existing.label = label
            continue
        mapping = HADoorMapping(entity_id=entity, label=label, camera=name)
        by_id[key] = mapping
        merged.append(mapping)
    return merged


def _ha_request(
    base_url: str,
    token: str,
    path: str,
    *,
    timeout: float = 4.0,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> Any:
    url = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    data = None
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers=headers, method=method.upper())
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310
        body = resp.read().decode("utf-8", errors="replace")
    if not body:
        return None
    return json.loads(body)


def call_ha_service(
    base_url: str,
    token: str,
    domain: str,
    service: str,
    entity_id: str,
    *,
    timeout: float = 4.0,
) -> str:
    """
    Call a Home Assistant service (e.g. light.turn_on).

    Returns empty string on success, or an error message.
    """
    base = normalize_ha_url(base_url)
    token = (token or "").strip()
    domain = (domain or "").strip().lower()
    service = (service or "").strip().lower()
    entity_id = (entity_id or "").strip()
    if not base:
        return "Home Assistant URL not set"
    if not token:
        return "Long-lived access token not set"
    if not domain or not service or not entity_id:
        return "Invalid service call"
    try:
        _ha_request(
            base,
            token,
            f"/api/services/{domain}/{service}",
            timeout=timeout,
            method="POST",
            payload={"entity_id": entity_id},
        )
        return ""
    except HTTPError as exc:
        if exc.code == 401:
            return "Unauthorized (check token)"
        return f"HTTP {exc.code}"
    except (URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError) as exc:
        return str(exc)[:120]


def fetch_door_states(
    base_url: str,
    token: str,
    mappings: list[HADoorMapping],
    *,
    previous: dict[str, DoorState] | None = None,
    timeout: float = 4.0,
) -> HASnapshot:
    """Poll HA REST API for configured door entities."""
    base = normalize_ha_url(base_url)
    token = (token or "").strip()
    if not base:
        return HASnapshot(error="Home Assistant URL not set", updated_at=time.time())
    if not token:
        return HASnapshot(error="Long-lived access token not set", updated_at=time.time())
    if not mappings:
        return HASnapshot(
            ok=True,
            connected=True,
            error="No door sensors configured",
            updated_at=time.time(),
        )

    previous = previous or {}
    now_mono = time.monotonic()
    now_wall = time.time()
    doors: list[DoorState] = []
    try:
        # Prefer one bulk fetch — fewer round-trips on typical LAN installs.
        all_states = _ha_request(base, token, "/api/states", timeout=timeout)
        by_id: dict[str, dict[str, Any]] = {}
        if isinstance(all_states, list):
            for row in all_states:
                if isinstance(row, dict) and row.get("entity_id"):
                    by_id[str(row["entity_id"]).lower()] = row
        else:
            by_id = {}

        for mapping in mappings:
            row = by_id.get(mapping.entity_id.lower())
            if row is None:
                # Fall back to single-entity lookup when entity missing from bulk.
                try:
                    row = _ha_request(
                        base, token, f"/api/states/{mapping.entity_id}", timeout=timeout
                    )
                except HTTPError as exc:
                    if exc.code == 404:
                        doors.append(
                            DoorState(
                                entity_id=mapping.entity_id,
                                label=mapping.display_label,
                                camera=mapping.camera,
                                open=False,
                                state="missing",
                                local_changed_at=now_mono,
                            )
                        )
                        continue
                    raise
            if not isinstance(row, dict):
                continue
            state = str(row.get("state") or "").strip().lower()
            attrs = row.get("attributes") if isinstance(row.get("attributes"), dict) else {}
            friendly = str(attrs.get("friendly_name") or "").strip()
            label = mapping.label or friendly or mapping.display_label
            is_open = state in {s.lower() for s in mapping.open_states}
            prev = previous.get(mapping.entity_id.lower())
            if prev is not None and prev.open == is_open:
                local_changed = prev.local_changed_at or now_mono
                last_opened = prev.last_opened_at
            else:
                local_changed = now_mono
                if is_open:
                    last_opened = now_mono
                else:
                    last_opened = prev.last_opened_at if prev is not None else 0.0
            doors.append(
                DoorState(
                    entity_id=mapping.entity_id,
                    label=label,
                    camera=mapping.camera,
                    open=is_open,
                    state=state,
                    changed_at=now_wall,
                    local_changed_at=local_changed,
                    last_opened_at=last_opened,
                    notify_hud=bool(mapping.notify_hud),
                    notify_popup=bool(mapping.notify_popup),
                    notify_highlight=bool(mapping.notify_highlight),
                    notify_autofocus=bool(mapping.notify_autofocus),
                    notify_sound=bool(mapping.notify_sound),
                )
            )
        return HASnapshot(
            ok=True,
            connected=True,
            doors=doors,
            updated_at=now_wall,
        )
    except HTTPError as exc:
        detail = f"HTTP {exc.code}"
        if exc.code == 401:
            detail = "Unauthorized (check token)"
        elif exc.code == 404:
            detail = "API not found (check URL)"
        return HASnapshot(error=detail, updated_at=now_wall)
    except (URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError) as exc:
        return HASnapshot(error=str(exc)[:120], updated_at=now_wall)


def parse_entity_catalog(rows: Any) -> list[HAEntityInfo]:
    """Parse `/api/states` payload into browseable entity infos."""
    if not isinstance(rows, list):
        return []
    out: list[HAEntityInfo] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        entity_id = str(row.get("entity_id") or "").strip()
        if not entity_id or "." not in entity_id:
            continue
        domain = entity_id.split(".", 1)[0].lower()
        attrs = row.get("attributes") if isinstance(row.get("attributes"), dict) else {}
        out.append(
            HAEntityInfo(
                entity_id=entity_id,
                state=str(row.get("state") or "").strip().lower(),
                friendly_name=str(attrs.get("friendly_name") or "").strip(),
                domain=domain,
                device_class=str(attrs.get("device_class") or "").strip().lower(),
                unit=str(attrs.get("unit_of_measurement") or "").strip(),
            )
        )
    out.sort(key=lambda e: (e.domain, e.display_name.lower(), e.entity_id.lower()))
    return out


def filter_entities(
    entities: list[HAEntityInfo],
    *,
    domain: str | None = None,
    query: str = "",
) -> list[HAEntityInfo]:
    domain_key = (domain or "").strip().lower()
    q = (query or "").strip().lower()
    out: list[HAEntityInfo] = []
    for entity in entities:
        if domain_key and domain_key != "all" and entity.domain != domain_key:
            continue
        if q:
            blob = f"{entity.entity_id} {entity.friendly_name} {entity.device_class}".lower()
            if q not in blob:
                continue
        out.append(entity)
    return out


def domain_counts(entities: list[HAEntityInfo]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for entity in entities:
        counts[entity.domain] = counts.get(entity.domain, 0) + 1
    preferred = [d for d in HA_BROWSE_DOMAINS if counts.get(d)]
    others = sorted(d for d in counts if d not in HA_BROWSE_DOMAINS)
    ordered = preferred + others
    return [(domain, counts[domain]) for domain in ordered]


def fetch_entity_catalog(
    base_url: str,
    token: str,
    *,
    timeout: float = 8.0,
) -> tuple[list[HAEntityInfo], str]:
    """
    Fetch all accessible HA entities.

    Returns (entities, error). error is empty on success.
    """
    base = normalize_ha_url(base_url)
    token = (token or "").strip()
    if not base:
        return [], "Home Assistant URL not set"
    if not token:
        return [], "Long-lived access token not set"
    try:
        rows = _ha_request(base, token, "/api/states", timeout=timeout)
        entities = parse_entity_catalog(rows)
        if not entities:
            return [], "No entities returned (check token permissions)"
        return entities, ""
    except HTTPError as exc:
        if exc.code == 401:
            return [], "Unauthorized (check token)"
        if exc.code == 404:
            return [], "API not found (check URL)"
        return [], f"HTTP {exc.code}"
    except (URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError) as exc:
        return [], str(exc)[:120]


class HomeAssistantService:
    """Background poller for Home Assistant door sensors."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = HASnapshot(error="Waiting for Home Assistant…")
        self._prev_states: dict[str, DoorState] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._enabled = False
        self._url = ""
        self._token = ""
        self._interval = 2.0
        self._mappings: list[HADoorMapping] = []
        self._entities: list[HAEntityInfo] = []
        self._entities_error = ""
        self._entities_updated_at = 0.0

    @property
    def snapshot(self) -> HASnapshot:
        with self._lock:
            return self._snapshot

    @property
    def entities(self) -> list[HAEntityInfo]:
        with self._lock:
            return list(self._entities)

    @property
    def entities_error(self) -> str:
        with self._lock:
            return self._entities_error

    @property
    def entities_updated_at(self) -> float:
        with self._lock:
            return self._entities_updated_at

    def configure(
        self,
        *,
        enabled: bool,
        url: str,
        token: str,
        poll_seconds: float = 2.0,
        doors: list[HADoorMapping] | None = None,
    ) -> None:
        with self._lock:
            self._enabled = bool(enabled)
            self._url = normalize_ha_url(url)
            self._token = (token or "").strip()
            self._interval = max(0.5, float(poll_seconds))
            self._mappings = list(doors or [])
        if enabled and (self._thread is None or not self._thread.is_alive()):
            self.start()
        if enabled:
            threading.Thread(target=self._refresh_once, name="ha-kick", daemon=True).start()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ha-service", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def refresh_now(self) -> HASnapshot:
        self._refresh_once()
        return self.snapshot

    def refresh_entities(self) -> tuple[list[HAEntityInfo], str]:
        with self._lock:
            url = self._url
            token = self._token
        entities, error = fetch_entity_catalog(url, token)
        with self._lock:
            self._entities = entities
            self._entities_error = error
            self._entities_updated_at = time.time()
        return entities, error

    def entity_state(self, entity_id: str) -> str:
        key = (entity_id or "").lower()
        with self._lock:
            for entity in self._entities:
                if entity.entity_id.lower() == key:
                    return entity.state
        return ""

    def call_service(self, domain: str, service: str, entity_id: str) -> str:
        with self._lock:
            url = self._url
            token = self._token
        error = call_ha_service(url, token, domain, service, entity_id)
        if not error:
            # Optimistic local state flip for snappy panel UI.
            desired = ""
            if service == "turn_on":
                desired = "on"
            elif service == "turn_off":
                desired = "off"
            elif service == "toggle":
                current = self.entity_state(entity_id)
                desired = "off" if current == "on" else "on"
            if desired:
                with self._lock:
                    for entity in self._entities:
                        if entity.entity_id.lower() == entity_id.lower():
                            entity.state = desired
                            break
            threading.Thread(
                target=self.refresh_entities, name="ha-entities-after-call", daemon=True
            ).start()
        return error

    def toggle_light(self, light: HALightControl) -> str:
        domain = light.domain or "light"
        if domain not in {"light", "switch", "input_boolean"}:
            domain = "light"
        return self.call_service(domain, "toggle", light.entity_id)

    def _refresh_once(self) -> None:
        with self._lock:
            if not self._enabled:
                return
            url = self._url
            token = self._token
            mappings = list(self._mappings)
            previous = dict(self._prev_states)
        snap = fetch_door_states(url, token, mappings, previous=previous)
        with self._lock:
            if snap.ok:
                self._prev_states = {d.entity_id.lower(): d for d in snap.doors}
            self._snapshot = snap

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._enabled:
                self._refresh_once()
            with self._lock:
                wait = self._interval
            self._stop.wait(wait)


def cameras_highlighted_by_doors(
    doors: list[DoorState],
    *,
    hold_seconds: float,
    now: float | None = None,
    require_highlight: bool = False,
    require_autofocus: bool = False,
) -> dict[str, str]:
    """
    Return camera_name → door label for cameras that should highlight / focus.

    Active while the door is open, or within ``hold_seconds`` after the most
    recent open edge (so a quick open/close still gets attention).
    """
    now = time.monotonic() if now is None else float(now)
    hold = max(0.0, float(hold_seconds))
    active: dict[str, str] = {}
    for door in doors:
        if require_highlight and not door.notify_highlight:
            continue
        if require_autofocus and not door.notify_autofocus:
            continue
        camera = (door.camera or "").strip()
        if not camera:
            continue
        recently_opened = (
            door.last_opened_at > 0 and (now - door.last_opened_at) <= hold
        )
        if door.open or recently_opened:
            if door.open or camera not in active:
                active[camera] = door.label
    return active


def open_sensor_labels(doors: list[DoorState]) -> dict[str, str]:
    """Camera name → friendly label for sensors that are still tripped."""
    active: dict[str, str] = {}
    for door in doors:
        if not door.open:
            continue
        if not door.notify_hud:
            continue
        camera = (door.camera or "").strip()
        if not camera:
            continue
        active[camera] = door.label or door.entity_id
    return active


def draw_sensor_chip(
    tile: np.ndarray,
    label: str,
    *,
    opacity: float = 0.82,
) -> None:
    """Quiet per-tile note that a linked sensor is still open."""
    text = (label or "").strip()
    if tile is None or tile.size == 0 or not text:
        return
    text = text[:36]
    h, w = tile.shape[:2]
    if h < 48 or w < 80:
        return
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    box_w = min(w - 12, size[0] + 16)
    box_h = 22
    x0, y0 = 8, 8
    opacity = max(0.35, min(1.0, float(opacity)))
    layer = tile[y0 : y0 + box_h, x0 : x0 + box_w].copy()
    shade_round_rect(
        layer,
        (0, 0, box_w, box_h),
        color=(22, 24, 32),
        alpha=0.88,
        radius=8,
    )
    draw_text(
        layer,
        text,
        (8, box_h // 2),
        size=13,
        color=(200, 210, 230),
        valign="center",
    )
    base = tile[y0 : y0 + box_h, x0 : x0 + box_w].astype(np.float32)
    top = layer.astype(np.float32)
    tile[y0 : y0 + box_h, x0 : x0 + box_w] = (
        base * (1.0 - opacity) + top * opacity
    ).astype(np.uint8)


def draw_door_hud(
    canvas: np.ndarray,
    snap: HASnapshot,
    *,
    opacity: float = 0.88,
) -> None:
    """Paint HA connection errors. Open sensors use the per-tile chip instead."""
    if canvas is None or canvas.size == 0:
        return
    if snap.connected and snap.ok:
        return
    opens: list[DoorState] = []
    title = "Home Assistant"
    detail = snap.error or "Disconnected"
    accent = (70, 140, 220)
    panel = (22, 24, 32)

    h, w = canvas.shape[:2]
    box_w = min(w - 24, max(280, int(w * 0.42)))
    box_h = 54 if opens else 44
    x0, y0 = 12, 12
    opacity = max(0.25, min(1.0, float(opacity)))
    layer = canvas[y0 : y0 + box_h, x0 : x0 + box_w].copy()
    shade_round_rect(
        layer,
        (0, 0, box_w, box_h),
        color=panel,
        alpha=0.90,
        radius=10,
    )
    cv2.rectangle(layer, (6, 6), (10, box_h - 6), accent, -1)
    draw_text(
        layer,
        title,
        (18, 8),
        size=13,
        color=accent,
        valign="top",
    )
    draw_text(
        layer,
        detail[:64],
        (18, box_h - 10),
        size=15,
        color=(230, 230, 235),
        valign="bottom",
    )
    base = canvas[y0 : y0 + box_h, x0 : x0 + box_w].astype(np.float32)
    top = layer.astype(np.float32)
    canvas[y0 : y0 + box_h, x0 : x0 + box_w] = (
        base * (1.0 - opacity) + top * opacity
    ).astype(np.uint8)


def door_open_edges(
    previous: dict[str, bool],
    doors: list[DoorState],
    *,
    snapshot_ok: bool,
) -> tuple[list[DoorState], list[DoorState], dict[str, bool]]:
    """
    Rising/falling edges for door alerts.

    First observation of an entity is a baseline (no toast) so already-open
    sensors at connect, or after a failed poll, do not spam the mosaic.
    Failed snapshots leave ``previous`` unchanged.
    """
    if not snapshot_ok:
        return [], [], dict(previous)
    opened: list[DoorState] = []
    closed: list[DoorState] = []
    nxt = dict(previous)
    for door in doors:
        key = door.entity_id.lower()
        was = previous.get(key)
        if was is None:
            nxt[key] = bool(door.open)
            continue
        if door.open and not was:
            opened.append(door)
        elif (not door.open) and was:
            closed.append(door)
        nxt[key] = bool(door.open)
    return opened, closed, nxt


def toast_seconds(requested: float) -> float:
    return max(1.5, min(HA_TOAST_MAX_SECONDS, float(requested)))


def upsert_popup(popups: list[HAPopup], popup: HAPopup) -> list[HAPopup]:
    """Replace any existing toast for the same entity, then append."""
    entity = (popup.entity_id or "").strip().lower()
    if entity:
        popups = [p for p in popups if (p.entity_id or "").strip().lower() != entity]
    popups.append(popup)
    return popups


def prune_popups(
    popups: list[HAPopup],
    *,
    now: float | None = None,
    closed_entity_ids: set[str] | None = None,
) -> list[HAPopup]:
    now = time.monotonic() if now is None else float(now)
    closed = {e.strip().lower() for e in (closed_entity_ids or set()) if e}
    out: list[HAPopup] = []
    for popup in popups:
        if popup.until <= now:
            continue
        if closed and (popup.entity_id or "").strip().lower() in closed:
            continue
        out.append(popup)
    return out


def draw_ha_popups(
    canvas: np.ndarray,
    popups: list[HAPopup],
    *,
    opacity: float = 0.92,
) -> None:
    """Draw temporary toast popups near the top-center (camera-independent)."""
    if canvas is None or canvas.size == 0 or not popups:
        return
    h, w = canvas.shape[:2]
    opacity = max(0.35, min(1.0, float(opacity)))
    y = 16
    for popup in popups[-4:]:
        text = (popup.message or "").strip()
        if not text:
            continue
        box_w = min(w - 40, max(280, int(w * 0.48)))
        box_h = 46
        x0 = max(12, (w - box_w) // 2)
        y0 = y
        if y0 + box_h >= h - 8:
            break
        layer = canvas[y0 : y0 + box_h, x0 : x0 + box_w].copy()
        shade_round_rect(
            layer,
            (0, 0, box_w, box_h),
            color=(18, 20, 28),
            alpha=0.94,
            radius=12,
        )
        cv2.rectangle(layer, (8, 8), (12, box_h - 8), popup.accent, -1)
        draw_text(
            layer,
            "HA",
            (20, 8),
            size=12,
            color=popup.accent,
            valign="top",
        )
        draw_text(
            layer,
            text[:58],
            (20, box_h - 10),
            size=15,
            color=(235, 235, 240),
            valign="bottom",
        )
        base = canvas[y0 : y0 + box_h, x0 : x0 + box_w].astype(np.float32)
        top = layer.astype(np.float32)
        canvas[y0 : y0 + box_h, x0 : x0 + box_w] = (
            base * (1.0 - opacity) + top * opacity
        ).astype(np.uint8)
        y += box_h + 8


def draw_ha_light_panel(
    canvas: np.ndarray,
    lights: list[HALightControl],
    states: dict[str, str],
    *,
    open_amount: float,
    enabled: bool = True,
) -> list[tuple[str, int, int, int, int]]:
    """
    Draw a right-side slide-out panel for HA lights.

    Returns hitboxes as (action, x0, y0, x1, y1). Actions:
      ha_panel_toggle — edge tab
      ha_light:<entity_id> — toggle that light
    """
    hitboxes: list[tuple[str, int, int, int, int]] = []
    if canvas is None or canvas.size == 0 or not enabled:
        return hitboxes
    h, w = canvas.shape[:2]
    open_amount = max(0.0, min(1.0, float(open_amount)))
    panel_w = PANEL_WIDTH
    slide = int(round(panel_w * open_amount))
    tab_w = PANEL_TAB_WIDTH
    # Tab always visible on the right edge.
    tab_x0 = w - tab_w
    tab_y0 = max(40, h // 2 - 48)
    tab_y1 = min(h - 40, tab_y0 + 96)
    shade_round_rect(
        canvas,
        (tab_x0 - 2, tab_y0, w, tab_y1),
        color=(28, 30, 38),
        alpha=0.92,
        radius=10,
    )
    draw_text(
        canvas,
        "HA",
        (tab_x0 + tab_w // 2, (tab_y0 + tab_y1) // 2),
        size=14,
        color=(210, 215, 230),
        align="center",
        valign="center",
    )
    hitboxes.append(("ha_panel_toggle", tab_x0 - 2, tab_y0, w, tab_y1))

    if slide <= 2:
        return hitboxes

    x0 = w - slide
    shade_round_rect(
        canvas,
        (x0, 8, w - 4, h - 8),
        color=(16, 18, 24),
        alpha=0.94,
        radius=14,
    )
    draw_text(
        canvas,
        "Lights",
        (x0 + (slide // 2), 20),
        size=18,
        color=(230, 230, 235),
        align="center",
        valign="top",
    )
    draw_text(
        canvas,
        "] close",
        (x0 + (slide // 2), 44),
        size=12,
        color=(140, 145, 155),
        align="center",
        valign="top",
    )

    if not lights:
        draw_text(
            canvas,
            "Add lights in",
            (x0 + (slide // 2), h // 2 - 10),
            size=14,
            color=(160, 165, 175),
            align="center",
            valign="center",
        )
        draw_text(
            canvas,
            "HA → Light panel",
            (x0 + (slide // 2), h // 2 + 14),
            size=14,
            color=(160, 165, 175),
            align="center",
            valign="center",
        )
        return hitboxes

    row_h = 54
    y = 68
    for light in lights:
        if y + row_h > h - 20:
            break
        state = (states.get(light.entity_id.lower()) or "").lower()
        on = state == "on"
        btn_x0 = x0 + 12
        btn_x1 = w - 16
        btn_y0 = y
        btn_y1 = y + row_h - 8
        color = (50, 140, 90) if on else (40, 42, 52)
        shade_round_rect(
            canvas,
            (btn_x0, btn_y0, btn_x1, btn_y1),
            color=color,
            alpha=0.92,
            radius=10,
        )
        status = "ON" if on else "OFF"
        status_color = (230, 240, 230) if on else (170, 175, 185)
        draw_text(
            canvas,
            light.display_label[:22],
            (btn_x0 + 12, btn_y0 + 8),
            size=15,
            color=(235, 235, 240),
            valign="top",
        )
        draw_text(
            canvas,
            f"Tap to turn {'off' if on else 'on'}  ·  {status}",
            (btn_x0 + 12, btn_y1 - 8),
            size=12,
            color=status_color,
            valign="bottom",
        )
        hitboxes.append((f"ha_light:{light.entity_id}", btn_x0, btn_y0, btn_x1, btn_y1))
        y += row_h
    return hitboxes
