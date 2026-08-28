"""Capture-thread helpers must not raise when OpenCV misbehaves."""

from __future__ import annotations

import time

import numpy as np

from security_monitor.config import CameraConfig, DisplayConfig
from security_monitor.stream import DemoWorker, safe_cap_read


class _BoomCap:
    def read(self):
        raise RuntimeError("Unknown C++ exception from FFmpeg")


class _BadCap:
    def read(self):
        return False, None


class _GoodCap:
    def read(self):
        return True, np.zeros((8, 8, 3), dtype=np.uint8)


def test_safe_cap_read_swallows_opencv_exceptions() -> None:
    ok, frame = safe_cap_read(_BoomCap())
    assert ok is False
    assert frame is None


def test_safe_cap_read_handles_failed_read() -> None:
    ok, frame = safe_cap_read(_BadCap())
    assert ok is False
    assert frame is None


def test_safe_cap_read_passes_through_frames() -> None:
    ok, frame = safe_cap_read(_GoodCap())
    assert ok is True
    assert frame is not None
    assert frame.shape == (8, 8, 3)


def test_demo_worker_pause_freezes_last_frame() -> None:
    cam = CameraConfig(name="Demo", url="demo://1")
    display = DisplayConfig(cell_width=80, cell_height=45, fps=30)
    worker = DemoWorker(cam, display, 0)
    worker.start()
    try:
        frame = None
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            snap = worker.snapshot()
            if snap.frame is not None:
                frame = snap.frame.copy()
                break
            time.sleep(0.02)
        assert frame is not None
        worker.set_paused(True)
        time.sleep(0.15)
        paused = worker.snapshot().frame
        assert paused is not None
        assert np.array_equal(frame, paused)
        worker.set_paused(False)
        moved = False
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            later = worker.snapshot().frame
            if later is not None and not np.array_equal(frame, later):
                moved = True
                break
            time.sleep(0.02)
        assert moved
    finally:
        worker.stop()
        worker.join(timeout=2.0)

