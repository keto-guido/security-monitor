"""Save snapshots and short video clips from the live mosaic."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

CLIP_LENGTH_CHOICES = (5, 10, 15, 30, 60)
VALID_SNAPSHOT_FORMATS = ("jpg", "jpeg", "png")


class CaptureError(RuntimeError):
    """Raised when a snapshot or clip cannot be written."""


def default_save_directory() -> Path:
    return Path.home() / "security-monitor" / "captures"


def resolve_save_directory(raw: str | None) -> Path:
    text = (raw or "").strip() or str(default_save_directory())
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _slug(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-._")
    return cleaned[:48] or "capture"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def snapshot_path(directory: Path, label: str, fmt: str = "jpg") -> Path:
    ext = fmt.lower().lstrip(".")
    if ext == "jpeg":
        ext = "jpg"
    if ext not in {"jpg", "png"}:
        ext = "jpg"
    return directory / f"{_timestamp()}_{_slug(label)}.{ext}"


def clip_path(directory: Path, label: str) -> Path:
    return directory / f"{_timestamp()}_{_slug(label)}.mp4"


def save_snapshot(
    frame: np.ndarray,
    directory: Path,
    label: str,
    *,
    fmt: str = "jpg",
    quality: int = 92,
) -> Path:
    if frame is None or frame.size == 0:
        raise CaptureError("No frame to save")
    directory = resolve_save_directory(str(directory))
    path = snapshot_path(directory, label, fmt=fmt)
    ext = path.suffix.lower()
    params: list[int] = []
    if ext in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    elif ext == ".png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
    ok = cv2.imwrite(str(path), frame, params)
    if not ok:
        raise CaptureError(f"Failed to write snapshot: {path}")
    return path


def write_clip(
    frames: list[np.ndarray],
    directory: Path,
    label: str,
    *,
    fps: float = 20.0,
) -> Path:
    if not frames:
        raise CaptureError("No frames to write")
    usable = [f for f in frames if f is not None and getattr(f, "size", 0) > 0]
    if not usable:
        raise CaptureError("No frames to write")
    directory = resolve_save_directory(str(directory))
    path = clip_path(directory, label)
    height, width = usable[0].shape[:2]
    # Normalize size — history may occasionally differ after rotate/reconnect.
    sized: list[np.ndarray] = []
    for frame in usable:
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        sized.append(frame)
    fps = max(5.0, min(60.0, float(fps)))
    writers = (
        ("mp4v", path),
        ("avc1", path),
        ("XVID", path.with_suffix(".avi")),
    )
    last_error = "no video backend"
    for fourcc_name, out_path in writers:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            writer.release()
            last_error = f"could not open writer ({fourcc_name})"
            continue
        for frame in sized:
            writer.write(frame)
        writer.release()
        if out_path.is_file() and out_path.stat().st_size > 0:
            return out_path
        last_error = f"empty output ({fourcc_name})"
    raise CaptureError(f"Failed to write clip: {last_error}")


@dataclass
class LiveClipJob:
    """Collect live frames for a fixed duration, then write an mp4."""

    label: str
    directory: Path
    duration: float
    fps: float
    started: float
    frames: list[np.ndarray]
    path: Path | None = None
    error: str = ""
    finished: bool = False

    @classmethod
    def start(
        cls,
        *,
        label: str,
        directory: Path,
        duration: float,
        fps: float,
    ) -> LiveClipJob:
        return cls(
            label=label,
            directory=resolve_save_directory(str(directory)),
            duration=max(1.0, float(duration)),
            fps=max(5.0, float(fps)),
            started=time.monotonic(),
            frames=[],
        )

    @property
    def elapsed(self) -> float:
        return max(0.0, time.monotonic() - self.started)

    @property
    def remaining(self) -> float:
        return max(0.0, self.duration - self.elapsed)

    def feed(self, frame: np.ndarray | None) -> bool:
        """Append a frame. Returns True when the job has finished (success or fail)."""
        if self.finished:
            return True
        if frame is not None and getattr(frame, "size", 0) > 0:
            self.frames.append(frame.copy())
        if self.elapsed < self.duration:
            return False
        try:
            self.path = write_clip(
                self.frames,
                self.directory,
                self.label,
                fps=self.fps,
            )
        except CaptureError as exc:
            self.error = str(exc)
        self.finished = True
        self.frames.clear()
        return True
