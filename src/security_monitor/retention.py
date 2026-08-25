"""Lockable media retention — auto-erase old unlocked captures/events."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from security_monitor.capture import (
    CAPTURE_IMAGE_EXTS,
    CAPTURE_VIDEO_EXTS,
    CaptureError,
    delete_capture,
    resolve_save_directory,
)
from security_monitor.events import (
    delete_person_event,
    list_person_events,
    write_event_meta,
)

# 0 = retention disabled for that axis.
RETENTION_DAY_CHOICES = (0, 1, 3, 7, 14, 30, 60, 90)
RETENTION_GB_CHOICES = (0, 1, 2, 5, 10, 20, 50, 100)


def capture_lock_path(path: Path) -> Path:
    """Sidecar marker: ``photo.jpg`` → ``photo.jpg.lock``."""
    return Path(str(path) + ".lock")


def is_capture_locked(path: Path) -> bool:
    return capture_lock_path(path).is_file()


def set_capture_locked(path: Path, locked: bool) -> None:
    target = Path(path)
    if not target.is_file():
        raise CaptureError(f"File not found: {target}")
    marker = capture_lock_path(target)
    if locked:
        marker.write_text("locked\n", encoding="utf-8")
    elif marker.is_file():
        try:
            marker.unlink()
        except OSError as exc:
            raise CaptureError(f"Could not unlock {target.name}: {exc}") from exc


def _read_event_meta(event_dir: Path) -> dict:
    meta_path = event_dir / "meta.json"
    if not meta_path.is_file():
        return {}
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def is_event_locked(event_dir: Path) -> bool:
    return bool(_read_event_meta(event_dir).get("locked"))


def set_event_locked(event_dir: Path, locked: bool) -> None:
    target = Path(event_dir)
    if not target.is_dir():
        raise CaptureError(f"Event not found: {target}")
    meta = _read_event_meta(target)
    meta["locked"] = bool(locked)
    if "camera" not in meta:
        meta["camera"] = target.name
    write_event_meta(target, meta)


def _dir_size_bytes(path: Path) -> int:
    total = 0
    try:
        for child in path.rglob("*"):
            if child.is_file():
                try:
                    total += child.stat().st_size
                except OSError:
                    continue
    except OSError:
        return 0
    return total


@dataclass(frozen=True)
class PurgeResult:
    deleted_events: int = 0
    deleted_files: int = 0
    freed_bytes: int = 0
    kept_locked: int = 0

    @property
    def deleted(self) -> int:
        return self.deleted_events + self.deleted_files

    @property
    def summary(self) -> str:
        if self.deleted <= 0:
            return "Nothing to erase"
        mb = self.freed_bytes / (1024 * 1024)
        return f"Erased {self.deleted} item(s), freed {mb:.1f} MB"


def purge_old_media(
    save_directory: Path | str | None,
    *,
    max_age_days: float = 0.0,
    max_total_gb: float = 0.0,
    now: datetime | None = None,
) -> PurgeResult:
    """
    Delete unlocked person events and flat captures that are too old and/or
    push total usage over ``max_total_gb``. Locked items are never removed.
    """
    root = resolve_save_directory(str(save_directory) if save_directory is not None else "")
    now = now or datetime.now()
    age_cutoff = None
    if max_age_days and max_age_days > 0:
        age_cutoff = now - timedelta(days=float(max_age_days))

    deleted_events = 0
    deleted_files = 0
    freed = 0
    kept_locked = 0

    # --- Age-based purge -------------------------------------------------
    if age_cutoff is not None:
        for event in list_person_events(root, limit=10_000):
            if event.locked:
                kept_locked += 1
                continue
            if event.when >= age_cutoff:
                continue
            size = _dir_size_bytes(event.path)
            try:
                delete_person_event(event.path)
            except CaptureError:
                continue
            deleted_events += 1
            freed += size

        for path in _iter_flat_captures(root):
            if is_capture_locked(path):
                kept_locked += 1
                continue
            try:
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
                size = int(path.stat().st_size)
            except OSError:
                continue
            if mtime >= age_cutoff:
                continue
            try:
                delete_capture(path)
            except CaptureError:
                continue
            deleted_files += 1
            freed += size

    # --- Size-based purge (oldest unlocked first) ------------------------
    max_bytes = int(float(max_total_gb) * 1024 * 1024 * 1024) if max_total_gb and max_total_gb > 0 else 0
    if max_bytes > 0:
        entries: list[tuple[float, int, str, Path]] = []
        # (mtime, size, kind, path)
        for event in list_person_events(root, limit=10_000):
            if event.locked:
                kept_locked += 1
                continue
            size = _dir_size_bytes(event.path)
            entries.append((event.when.timestamp(), size, "event", event.path))
        for path in _iter_flat_captures(root):
            if is_capture_locked(path):
                kept_locked += 1
                continue
            try:
                st = path.stat()
            except OSError:
                continue
            entries.append((float(st.st_mtime), int(st.st_size), "file", path))

        total = sum(size for _m, size, _k, _p in entries) + _locked_bytes(root)
        entries.sort(key=lambda row: row[0])  # oldest first
        for _mtime, size, kind, path in entries:
            if total <= max_bytes:
                break
            try:
                if kind == "event":
                    delete_person_event(path)
                    deleted_events += 1
                else:
                    delete_capture(path)
                    deleted_files += 1
            except CaptureError:
                continue
            total -= size
            freed += size

    return PurgeResult(
        deleted_events=deleted_events,
        deleted_files=deleted_files,
        freed_bytes=freed,
        kept_locked=kept_locked,
    )


def _iter_flat_captures(root: Path) -> list[Path]:
    out: list[Path] = []
    try:
        for path in root.iterdir():
            if not path.is_file():
                continue
            if path.name.endswith(".lock"):
                continue
            if path.suffix.lower() in CAPTURE_IMAGE_EXTS | CAPTURE_VIDEO_EXTS:
                out.append(path)
    except OSError:
        return []
    return out


def _locked_bytes(root: Path) -> int:
    total = 0
    for event in list_person_events(root, limit=10_000):
        if event.locked:
            total += _dir_size_bytes(event.path)
    for path in _iter_flat_captures(root):
        if is_capture_locked(path):
            try:
                total += path.stat().st_size
            except OSError:
                continue
    return total


def retention_label_days(days: float) -> str:
    if not days or days <= 0:
        return "Off"
    if float(days).is_integer():
        return f"{int(days)}d"
    return f"{days:g}d"


def retention_label_gb(gb: float) -> str:
    if not gb or gb <= 0:
        return "Off"
    return f"{gb:g} GB"
