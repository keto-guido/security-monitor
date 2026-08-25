"""Tests for snapshot and clip export helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from security_monitor.buffer import FrameHistory
from security_monitor.capture import (
    CaptureError,
    LiveClipJob,
    resolve_save_directory,
    save_snapshot,
    write_clip,
)
from security_monitor.config import parse_config


def _frame(value: int, size: int = 48) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = (value, value // 2, 40)
    return img


def test_save_snapshot_writes_jpg(tmp_path: Path) -> None:
    path = save_snapshot(_frame(200), tmp_path, "Front Door", fmt="jpg")
    assert path.is_file()
    assert path.suffix == ".jpg"
    assert path.stat().st_size > 0


def test_write_clip_from_frames(tmp_path: Path) -> None:
    frames = [_frame(30 + i * 10) for i in range(12)]
    path = write_clip(frames, tmp_path, "Driveway", fps=10)
    assert path.is_file()
    assert path.suffix in {".mp4", ".avi"}
    assert path.stat().st_size > 0


def test_write_clip_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(CaptureError):
        write_clip([], tmp_path, "Empty")


def test_live_clip_job_finishes(tmp_path: Path) -> None:
    job = LiveClipJob.start(label="Cam", directory=tmp_path, duration=0.05, fps=20)
    for i in range(8):
        done = job.feed(_frame(50 + i))
        if done:
            break
    # Force completion if the loop was faster than duration.
    if not job.finished:
        job.started -= 1.0
        assert job.feed(_frame(99)) is True
    assert job.finished
    assert job.path is not None and job.path.is_file()


def test_history_export_and_config(tmp_path: Path) -> None:
    hist = FrameHistory()
    hist.configure(clip_seconds=3)
    import time

    t0 = time.monotonic()
    for i in range(10):
        hist.push(_frame(i * 12), when=t0 + i * 0.2)
    frames, fps = hist.export_frames(1.5)
    assert len(frames) >= 4
    assert fps > 0

    cfg = parse_config(
        {
            "display": {
                "clip_seconds": 20,
                "snapshot_format": "png",
                "save_directory": str(tmp_path / "out"),
            },
            "cameras": [{"name": "A", "url": "rtsp://192.168.1.10/live"}],
        }
    )
    assert cfg.display.clip_seconds == 20
    assert cfg.display.snapshot_format == "png"
    resolved = resolve_save_directory(cfg.display.save_directory)
    assert resolved == (tmp_path / "out").resolve()
    assert resolved.is_dir()
