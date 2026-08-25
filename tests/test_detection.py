"""Tests for people / new-object detection helpers."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from security_monitor.config import parse_config, save_display_settings
from security_monitor.detection import Box, NewObjectTracker, draw_boxes


def test_box_iou_and_clip() -> None:
    a = Box(0, 0, 50, 50, "person")
    b = Box(25, 25, 75, 75, "person")
    assert a.iou(b) == pytest.approx(0.142857, rel=1e-3)
    clipped = Box(-10, -5, 1000, 1000, "x").clip(100, 80)
    assert clipped.x1 == 0 and clipped.y1 == 0
    assert clipped.x2 == 99 and clipped.y2 == 79


def test_new_object_appears_against_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    tracker = NewObjectTracker()
    baseline = np.zeros((240, 320, 3), dtype=np.uint8)
    baseline[:] = (40, 40, 40)
    tracker.set_baseline("Porch", baseline)

    # Empty scene — nothing new.
    assert tracker.detect("Porch", baseline.copy(), confirm_seconds=0.01) == []

    # Place a bright "package".
    frame = baseline.copy()
    frame[80:140, 100:180] = (200, 200, 200)
    # First sighting is unconfirmed.
    assert tracker.detect("Porch", frame, confirm_seconds=0.2) == []
    time.sleep(0.25)
    boxes = tracker.detect("Porch", frame, confirm_seconds=0.2)
    assert len(boxes) == 1
    assert boxes[0].label == "new object"

    # Remove package — box clears after grace.
    time.sleep(1.1)
    assert tracker.detect("Porch", baseline.copy(), confirm_seconds=0.2) == []


def test_draw_boxes_mutates_frame() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    before = int(frame.sum())
    draw_boxes(frame, [Box(10, 10, 60, 60, "person", 0.9)])
    assert int(frame.sum()) > before


def test_detection_config_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "display:\n  columns: 1\n  rows: 1\n"
        "cameras:\n  - name: Gate\n    url: rtsp://1.2.3.4/x\n",
        encoding="utf-8",
    )
    cfg = parse_config(
        {
            "display": {"people_detection": True, "object_detection": True},
            "cameras": [
                {
                    "name": "Gate",
                    "url": "rtsp://1.2.3.4/x",
                    "detect_people": True,
                    "detect_objects": True,
                }
            ],
        },
        path=path,
    )
    assert cfg.display.people_detection is True
    assert cfg.cameras[0].detect_objects is True
    assert save_display_settings(cfg) == path
    text = path.read_text(encoding="utf-8")
    assert "people_detection: true" in text
    assert "detect_people: true" in text
