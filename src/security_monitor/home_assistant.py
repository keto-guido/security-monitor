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
HA_HOLD_CHOICES: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0, 30.0, 60.0)
DEFAULT_OPEN_STATES: tuple[str, ...] = ("on", "open", "unlocked")


@dataclass
class HADoorMapping:
    """Maps a Home Assistant entity to an optional camera and HUD label."""

    entity_id: str
    label: str = ""
    camera: str = ""
    open_states: tuple[str, ...] = DEFAULT_OPEN_STATES

    @property
    def display_label(self) -> str:
        return (self.label or self.entity_id.split(".", 1)[-1] or self.entity_id).strip()


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
            )
        )
    return out


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
) -> Any:
    url = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    req = Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="GET",
    )
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310
        body = resp.read().decode("utf-8", errors="replace")
    if not body:
        return None
    return json.loads(body)


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

    @property
    def snapshot(self) -> HASnapshot:
        with self._lock:
            return self._snapshot

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
) -> dict[str, str]:
    """
    Return camera_name → door label for cameras that should highlight.

    Active while the door is open, or within ``hold_seconds`` after the most
    recent open edge (so a quick open/close still gets attention).
    """
    now = time.monotonic() if now is None else float(now)
    hold = max(0.0, float(hold_seconds))
    active: dict[str, str] = {}
    for door in doors:
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


def draw_door_hud(
    canvas: np.ndarray,
    snap: HASnapshot,
    *,
    opacity: float = 0.88,
) -> None:
    """Paint a compact door-status strip near the top of the mosaic."""
    if canvas is None or canvas.size == 0:
        return
    opens = snap.open_doors
    if opens:
        title = "DOOR OPEN"
        detail = " · ".join(d.label for d in opens[:4])
        if len(opens) > 4:
            detail += f" +{len(opens) - 4}"
        accent = (40, 90, 255)
        panel = (24, 20, 18)
    elif snap.connected and snap.ok:
        # Quiet connected state — skip HUD clutter unless something is open
        # or there is an error worth showing.
        return
    else:
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
