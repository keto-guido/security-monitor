"""Person-detection event capture: pre-roll + presence + post-roll clips."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from security_monitor.capture import CaptureError, resolve_save_directory, write_clip

PERSON_PRE_ROLL_CHOICES = (2, 3, 5, 10, 15)
PERSON_POST_ROLL_CHOICES = (2, 3, 5, 10, 15)


def events_root(save_directory: Path | str | None) -> Path:
    root = resolve_save_directory(str(save_directory) if save_directory is not None else "")
    path = root / "events"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _slug(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-._")
    return cleaned[:48] or "camera"


def new_event_dir(save_directory: Path | str | None, camera: str, when: datetime | None = None) -> Path:
    stamp = (when or datetime.now()).strftime("%Y%m%d-%H%M%S")
    root = events_root(save_directory)
    base = f"{stamp}_{_slug(camera)}"
    folder = root / base
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=False)
        return folder
    for n in range(2, 1000):
        folder = root / f"{base}-{n}"
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=False)
            return folder
    raise CaptureError("Could not allocate event folder")


def stamp_datetime_overlay(frame: np.ndarray, *, camera: str, when: datetime | None = None) -> np.ndarray:
    """Copy frame and burn camera + date/time into a bottom bar."""
    if frame is None or frame.size == 0:
        raise CaptureError("No frame to stamp")
    out = frame.copy()
    when = when or datetime.now()
    label = f"{camera}  ·  {when.strftime('%Y-%m-%d %H:%M:%S')}"
    h, w = out.shape[:2]
    bar_h = max(28, h // 18)
    y0 = h - bar_h
    overlay = out.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (16, 16, 20), -1)
    cv2.addWeighted(overlay, 0.72, out, 0.28, 0, out)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, min(0.9, w / 900))
    thickness = 1 if scale < 0.7 else 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    x = max(8, (w - tw) // 2)
    y = y0 + (bar_h + th) // 2 - 2
    cv2.putText(out, label, (x, y), font, scale, (235, 235, 235), thickness, cv2.LINE_AA)
    return out


def save_event_snapshot(frame: np.ndarray, event_dir: Path, *, fmt: str = "jpg") -> Path:
    event_dir.mkdir(parents=True, exist_ok=True)
    ext = "png" if fmt.lower().lstrip(".") == "png" else "jpg"
    path = event_dir / f"snapshot.{ext}"
    params: list[int] = []
    if ext == "jpg":
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 92]
    ok = cv2.imwrite(str(path), frame, params)
    if not ok:
        raise CaptureError(f"Failed to write event snapshot: {path}")
    return path


def write_event_meta(event_dir: Path, payload: dict) -> Path:
    path = event_dir / "meta.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


@dataclass(frozen=True)
class PersonEventItem:
    path: Path
    camera: str
    when: datetime
    snapshot: Path
    clip: Path | None
    has_clip: bool

    @property
    def label(self) -> str:
        stamp = self.when.strftime("%Y-%m-%d %H:%M:%S")
        clip = "clip" if self.has_clip else "no clip"
        return f"{stamp}  ·  {self.camera}  ·  {clip}"


def list_person_events(save_directory: Path | str | None, *, limit: int = 200) -> list[PersonEventItem]:
    root = events_root(save_directory)
    items: list[PersonEventItem] = []
    try:
        dirs = [p for p in root.iterdir() if p.is_dir()]
    except OSError:
        return []
    for folder in dirs:
        snap = folder / "snapshot.jpg"
        if not snap.is_file():
            snap = folder / "snapshot.png"
        if not snap.is_file():
            continue
        clip = folder / "clip.mp4"
        if not clip.is_file():
            alt = folder / "clip.avi"
            clip_path = alt if alt.is_file() else None
        else:
            clip_path = clip
        camera = folder.name
        when = datetime.fromtimestamp(snap.stat().st_mtime)
        meta_path = folder / "meta.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                camera = str(meta.get("camera") or camera)
                raw_when = meta.get("started_at")
                if raw_when:
                    when = datetime.fromisoformat(str(raw_when))
            except (OSError, ValueError, json.JSONDecodeError, TypeError):
                pass
        else:
            # folder name: YYYYMMDD-HHMMSS_Camera-Name
            parts = folder.name.split("_", 1)
            if len(parts) == 2:
                camera = parts[1].replace("-", " ")
                try:
                    when = datetime.strptime(parts[0], "%Y%m%d-%H%M%S")
                except ValueError:
                    pass
        items.append(
            PersonEventItem(
                path=folder,
                camera=camera,
                when=when,
                snapshot=snap,
                clip=clip_path,
                has_clip=clip_path is not None,
            )
        )
    items.sort(key=lambda item: item.when, reverse=True)
    return items[: max(1, int(limit))]


def delete_person_event(event_dir: Path) -> None:
    target = Path(event_dir)
    if not target.is_dir():
        raise CaptureError(f"Event not found: {target}")
    # Only delete known event files, then the directory.
    for name in ("snapshot.jpg", "snapshot.png", "clip.mp4", "clip.avi", "meta.json"):
        path = target / name
        if path.is_file():
            try:
                path.unlink()
            except OSError as exc:
                raise CaptureError(f"Could not delete {path.name}: {exc}") from exc
    try:
        target.rmdir()
    except OSError as exc:
        raise CaptureError(f"Could not remove event folder: {exc}") from exc


@dataclass
class PersonEventRecorder:
    """
    Open-ended clip writer: seeded with pre-roll frames, fed live until the
    person is gone for ``post_roll`` seconds (or ``max_seconds`` elapses).
    """

    camera: str
    event_dir: Path
    frames: list[np.ndarray] = field(default_factory=list)
    fps: float = 12.0
    post_roll: float = 5.0
    max_seconds: float = 120.0
    started_mono: float = field(default_factory=time.monotonic)
    started_at: datetime = field(default_factory=datetime.now)
    snapshot_path: Path | None = None
    person_present: bool = True
    lost_at: float | None = None
    finished: bool = False
    clip_path: Path | None = None
    error: str = ""
    _last_feed: float = 0.0

    @classmethod
    def start(
        cls,
        *,
        camera: str,
        save_directory: Path | str | None,
        pre_frames: list[np.ndarray],
        snapshot_frame: np.ndarray,
        fps: float,
        post_roll: float,
        max_seconds: float,
        snapshot_format: str = "jpg",
    ) -> PersonEventRecorder:
        when = datetime.now()
        event_dir = new_event_dir(save_directory, camera, when=when)
        stamped = stamp_datetime_overlay(snapshot_frame, camera=camera, when=when)
        snap_path = save_event_snapshot(stamped, event_dir, fmt=snapshot_format)
        write_event_meta(
            event_dir,
            {
                "camera": camera,
                "started_at": when.isoformat(timespec="seconds"),
                "pre_frames": len(pre_frames),
                "post_roll_seconds": float(post_roll),
                "max_seconds": float(max_seconds),
            },
        )
        frames = [f.copy() for f in pre_frames if f is not None and getattr(f, "size", 0) > 0]
        return cls(
            camera=camera,
            event_dir=event_dir,
            frames=frames,
            fps=max(5.0, float(fps)),
            post_roll=max(0.05, float(post_roll)),
            max_seconds=max(5.0, float(max_seconds)),
            started_mono=time.monotonic(),
            started_at=when,
            snapshot_path=snap_path,
            person_present=True,
        )

    @property
    def elapsed(self) -> float:
        return max(0.0, time.monotonic() - self.started_mono)

    def feed(self, frame: np.ndarray | None, *, person_present: bool) -> bool:
        """Append a live frame. Returns True when the event has finished writing."""
        if self.finished:
            return True
        now = time.monotonic()
        if frame is not None and getattr(frame, "size", 0) > 0:
            # Limit feed rate roughly to history fps.
            if self._last_feed <= 0 or (now - self._last_feed) >= (1.0 / max(self.fps, 5.0)):
                self.frames.append(frame.copy())
                self._last_feed = now
        self.person_present = bool(person_present)
        if person_present:
            self.lost_at = None
        elif self.lost_at is None:
            self.lost_at = now

        done = False
        if self.elapsed >= self.max_seconds:
            done = True
        elif self.lost_at is not None and (now - self.lost_at) >= self.post_roll:
            done = True
        if not done:
            return False
        self._finalize()
        return True

    def _finalize(self) -> None:
        self.finished = True
        try:
            if not self.frames:
                raise CaptureError("No frames captured for person event")
            path = write_clip(
                self.frames,
                self.event_dir,
                "clip",
                fps=self.fps,
            )
            dest = self.event_dir / f"clip{path.suffix.lower()}"
            if path.resolve() != dest.resolve():
                if dest.exists():
                    dest.unlink()
                path.replace(dest)
            self.clip_path = dest
            write_event_meta(
                self.event_dir,
                {
                    "camera": self.camera,
                    "started_at": self.started_at.isoformat(timespec="seconds"),
                    "ended_at": datetime.now().isoformat(timespec="seconds"),
                    "duration_seconds": round(self.elapsed, 2),
                    "frame_count": len(self.frames),
                    "clip": dest.name,
                    "snapshot": self.snapshot_path.name if self.snapshot_path else None,
                },
            )
        except CaptureError as exc:
            self.error = str(exc)
        self.frames.clear()
