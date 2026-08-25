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
CAPTURE_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
CAPTURE_VIDEO_EXTS = frozenset({".mp4", ".avi", ".mkv", ".mov", ".m4v"})


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


@dataclass(frozen=True)
class CaptureItem:
    path: Path
    kind: str  # image | video
    size: int
    mtime: float
    locked: bool = False

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def label(self) -> str:
        stamp = datetime.fromtimestamp(self.mtime).strftime("%Y-%m-%d %H:%M")
        tag = "IMG" if self.kind == "image" else "VID"
        lock = "LOCKED · " if self.locked else ""
        return f"{lock}[{tag}] {self.name}  ·  {_format_bytes(self.size)}  ·  {stamp}"


def _format_bytes(size: int) -> str:
    value = float(max(0, size))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{size} B"


def list_captures(directory: Path | str | None, *, limit: int = 200) -> list[CaptureItem]:
    """Newest-first snapshots and clips under the save directory."""
    root = resolve_save_directory(str(directory) if directory is not None else "")
    items: list[CaptureItem] = []
    try:
        entries = list(root.iterdir())
    except OSError:
        return []
    for path in entries:
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext in CAPTURE_IMAGE_EXTS:
            kind = "image"
        elif ext in CAPTURE_VIDEO_EXTS:
            kind = "video"
        else:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        locked = Path(str(path) + ".lock").is_file()
        items.append(
            CaptureItem(
                path=path,
                kind=kind,
                size=int(stat.st_size),
                mtime=float(stat.st_mtime),
                locked=locked,
            )
        )
    items.sort(key=lambda item: item.mtime, reverse=True)
    return items[: max(1, int(limit))]


def delete_capture(path: Path) -> None:
    target = Path(path)
    if not target.is_file():
        raise CaptureError(f"File not found: {target}")
    try:
        target.unlink()
    except OSError as exc:
        raise CaptureError(f"Could not delete {target.name}: {exc}") from exc
    marker = Path(str(target) + ".lock")
    if marker.is_file():
        try:
            marker.unlink()
        except OSError:
            pass


def delete_captures(paths: list[Path]) -> int:
    deleted = 0
    for path in paths:
        try:
            delete_capture(path)
            deleted += 1
        except CaptureError:
            continue
    return deleted


def load_capture_preview(path: Path, *, max_edge: int = 1280) -> np.ndarray | None:
    """Load an image or the first frame of a video for on-screen preview."""
    target = Path(path)
    if not target.is_file():
        return None
    ext = target.suffix.lower()
    frame: np.ndarray | None = None
    if ext in CAPTURE_IMAGE_EXTS:
        frame = cv2.imread(str(target), cv2.IMREAD_COLOR)
    elif ext in CAPTURE_VIDEO_EXTS:
        cap = cv2.VideoCapture(str(target))
        try:
            ok, grabbed = cap.read()
            if ok and grabbed is not None:
                frame = grabbed
        finally:
            cap.release()
    if frame is None or frame.size == 0:
        return None
    h, w = frame.shape[:2]
    edge = max(h, w)
    if edge > max_edge:
        scale = max_edge / edge
        frame = cv2.resize(
            frame,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return frame


def reveal_in_file_manager(path: Path) -> str:
    """Open the captures folder (or parent of a file) in the OS file manager."""
    import os
    import subprocess
    import sys

    target = Path(path)
    folder = target if target.is_dir() else target.parent
    folder.mkdir(parents=True, exist_ok=True)
    try:
        if os.name == "nt":
            if target.is_file():
                subprocess.Popen(["explorer", "/select,", str(target)])  # noqa: S603
            else:
                os.startfile(str(folder))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            if target.is_file():
                subprocess.Popen(["open", "-R", str(target)])  # noqa: S603
            else:
                subprocess.Popen(["open", str(folder)])  # noqa: S603
        else:
            subprocess.Popen(["xdg-open", str(folder)])  # noqa: S603
    except OSError as exc:
        raise CaptureError(f"Could not open folder: {exc}") from exc
    return str(folder)


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
