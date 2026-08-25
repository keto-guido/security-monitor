"""Tests for automatic person-detection event capture."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from security_monitor.events import (
    PersonEventRecorder,
    delete_person_event,
    list_person_events,
    stamp_datetime_overlay,
)
from security_monitor.config import parse_config, save_display_settings


def _frame(value: int, size: int = 64) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = (value, 40, 80)
    return img


def test_stamp_datetime_overlay_mutates_copy() -> None:
    frame = _frame(200)
    stamped = stamp_datetime_overlay(frame, camera="Porch")
    assert stamped.shape == frame.shape
    assert int(stamped.sum()) != int(frame.sum())


def test_person_event_recorder_writes_snapshot_and_clip(tmp_path: Path) -> None:
    pre = [_frame(20 + i) for i in range(6)]
    snap = _frame(220)
    recorder = PersonEventRecorder.start(
        camera="Front Door",
        save_directory=tmp_path,
        pre_frames=pre,
        snapshot_frame=snap,
        fps=10,
        post_roll=0.05,
        max_seconds=5,
        snapshot_format="jpg",
    )
    assert recorder.snapshot_path is not None and recorder.snapshot_path.is_file()
    # Still present briefly.
    assert recorder.feed(_frame(100), person_present=True) is False
    # Lost — arm post-roll, then finish after it elapses.
    assert recorder.feed(_frame(90), person_present=False) is False
    time.sleep(0.08)
    assert recorder.feed(_frame(85), person_present=False) is True
    assert recorder.finished
    assert recorder.error == ""
    assert recorder.clip_path is not None and recorder.clip_path.is_file()

    events = list_person_events(tmp_path)
    assert len(events) == 1
    assert events[0].camera == "Front Door"
    assert events[0].has_clip is True
    delete_person_event(events[0].path)
    assert list_person_events(tmp_path) == []


def test_auto_person_capture_config_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "display:\n  columns: 1\n  rows: 1\n"
        "cameras:\n  - name: A\n    url: rtsp://1.2.3.4/x\n",
        encoding="utf-8",
    )
    cfg = parse_config(
        {
            "display": {
                "auto_person_capture": True,
                "person_pre_roll_seconds": 3,
                "person_post_roll_seconds": 10,
            },
            "cameras": [{"name": "A", "url": "rtsp://1.2.3.4/x"}],
        },
        path=path,
    )
    assert cfg.display.auto_person_capture is True
    assert cfg.display.person_pre_roll_seconds == 3
    assert save_display_settings(cfg) == path
    text = path.read_text(encoding="utf-8")
    assert "auto_person_capture: true" in text
    assert "person_pre_roll_seconds: 3" in text
    assert "person_post_roll_seconds: 10" in text
