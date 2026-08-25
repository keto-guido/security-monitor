"""Tests for people / new-object detection helpers."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from security_monitor.config import parse_config, save_display_settings
from security_monitor.detection import Box, DetectionEngine, NewObjectTracker, draw_boxes


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
    # Disable adapt during this test so the seeded baseline stays fixed.
    tracker = NewObjectTracker(adapt_tau_seconds=1e9, persist_every_seconds=1e9)
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

    # Remove package — box clears after grace (~1.25s).
    time.sleep(1.35)
    assert tracker.detect("Porch", baseline.copy(), confirm_seconds=0.2) == []


def test_baseline_adapts_around_frozen_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    tracker = NewObjectTracker(
        adapt_tau_seconds=0.05,
        absorb_non_reinforced_seconds=1e9,
        persist_every_seconds=1e9,
    )
    baseline = np.zeros((240, 320, 3), dtype=np.uint8)
    baseline[:] = (40, 40, 40)
    tracker.set_baseline("Yard", baseline)

    package = baseline.copy()
    package[80:140, 100:180] = (220, 220, 220)
    assert tracker.detect("Yard", package, confirm_seconds=0.05) == []
    time.sleep(0.08)
    assert len(tracker.detect("Yard", package, confirm_seconds=0.05)) == 1

    # Surroundings shift (season / leaves) while the package stays put.
    seasonal = package.copy()
    seasonal[:, :] = (90, 110, 70)
    seasonal[80:140, 100:180] = (220, 220, 220)
    for _ in range(40):
        boxes = tracker.detect(
            "Yard", seasonal, confirm_seconds=0.05, adapt_tau_seconds=0.05
        )
        assert len(boxes) == 1
        time.sleep(0.02)

    # Same seasonal scene without the package should eventually clear.
    clear = seasonal.copy()
    clear[80:140, 100:180] = (90, 110, 70)
    time.sleep(1.35)
    # Allow a bit more adapt so the old package hole fills from neighbors over time
    # once the track expires — run a few empty frames.
    for _ in range(20):
        tracker.detect("Yard", clear, confirm_seconds=0.05, adapt_tau_seconds=0.05)
        time.sleep(0.02)
    assert tracker.detect("Yard", clear, confirm_seconds=0.05) == []


def test_gradual_scene_drift_does_not_false_alarm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    tracker = NewObjectTracker(
        adapt_tau_seconds=0.05,
        persist_every_seconds=1e9,
    )
    baseline = np.zeros((240, 320, 3), dtype=np.uint8)
    baseline[:] = (50, 50, 50)
    tracker.set_baseline("Drive", baseline)

    frame = baseline.copy()
    for step in range(25):
        # Mild global drift — should be absorbed, not flagged as a package.
        value = min(50 + step * 4, 140)
        frame[:, :] = (value, value, value)
        boxes = tracker.detect(
            "Drive", frame, confirm_seconds=0.05, adapt_tau_seconds=0.05
        )
        time.sleep(0.02)
    assert boxes == []


def test_draw_boxes_mutates_frame() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    before = int(frame.sum())
    draw_boxes(frame, [Box(10, 10, 60, 60, "person", 0.9)])
    assert int(frame.sum()) > before


def test_detection_engine_people_backend() -> None:
    eng = DetectionEngine()
    backend = eng.ensure_ready()
    assert backend in {"yolov8n", "mobilenet-ssd", "hog", "unavailable"}


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
