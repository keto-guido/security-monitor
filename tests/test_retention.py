"""Tests for lockable capture/event retention."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from security_monitor.capture import save_snapshot
from security_monitor.events import PersonEventRecorder, list_person_events
from security_monitor.retention import (
    is_capture_locked,
    is_event_locked,
    purge_old_media,
    set_capture_locked,
    set_event_locked,
)


def _frame(value: int = 180) -> np.ndarray:
    img = np.zeros((48, 48, 3), dtype=np.uint8)
    img[:] = (value, 50, 90)
    return img


def test_capture_lock_sidecar(tmp_path: Path) -> None:
    path = save_snapshot(_frame(), tmp_path, "Gate", fmt="jpg")
    assert is_capture_locked(path) is False
    set_capture_locked(path, True)
    assert is_capture_locked(path) is True
    set_capture_locked(path, False)
    assert is_capture_locked(path) is False


def test_purge_skips_locked_and_removes_old(tmp_path: Path) -> None:
    old = save_snapshot(_frame(10), tmp_path, "Old", fmt="jpg")
    locked = save_snapshot(_frame(200), tmp_path, "Keep", fmt="jpg")
    set_capture_locked(locked, True)

    # Backdate the unlocked file.
    old_mtime = (datetime.now() - timedelta(days=30)).timestamp()
    import os

    os.utime(old, (old_mtime, old_mtime))

    result = purge_old_media(tmp_path, max_age_days=7, max_total_gb=0)
    assert result.deleted_files == 1
    assert not old.is_file()
    assert locked.is_file()
    assert is_capture_locked(locked) is True


def test_purge_respects_locked_person_event(tmp_path: Path) -> None:
    recorder = PersonEventRecorder.start(
        camera="Porch",
        save_directory=tmp_path,
        pre_frames=[_frame(i) for i in range(4)],
        snapshot_frame=_frame(220),
        fps=10,
        post_roll=0.05,
        max_seconds=5,
    )
    recorder.lost_at = recorder.started_mono - 1.0
    assert recorder.feed(_frame(90), person_present=False) is True
    events = list_person_events(tmp_path)
    assert len(events) == 1
    set_event_locked(events[0].path, True)
    assert is_event_locked(events[0].path) is True

    # Age the event via meta started_at.
    meta_path = events[0].path / "meta.json"
    meta = meta_path.read_text(encoding="utf-8")
    old = (datetime.now() - timedelta(days=40)).isoformat(timespec="seconds")
    meta_path.write_text(meta.replace(events[0].when.isoformat(timespec="seconds"), old), encoding="utf-8")

    result = purge_old_media(tmp_path, max_age_days=7, max_total_gb=0)
    assert result.deleted_events == 0
    assert events[0].path.is_dir()
