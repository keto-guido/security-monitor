"""Capture-thread helpers must not raise when OpenCV misbehaves."""

from __future__ import annotations

import time

import numpy as np

from security_monitor.config import CameraConfig, DisplayConfig
from security_monitor.stream import DemoWorker, Snapshot, safe_cap_read


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



def _demo_worker() -> DemoWorker:
    display = DisplayConfig(cell_width=64, cell_height=48, fps=30)
    return DemoWorker(CameraConfig(name="demo", url="demo://x"), display, 0)


def test_snapshot_frame_key_distinguishes_pictures() -> None:
    assert Snapshot(frame=None, status="live", seq=4).frame_key == (4, 0.0)
    assert Snapshot(frame=None, status="live", seq=4) .frame_key != (
        Snapshot(frame=None, status="live", seq=5).frame_key
    )


def test_snapshot_fps_is_the_source_rate_not_the_displayed_rate() -> None:
    # Documented contract: renderers must not present this as "displayed fps".
    snap = Snapshot(frame=None, status="live", fps=25.0, seq=1)
    assert snap.fps == 25.0
    assert snap.frame_key == (1, 0.0)


def test_worker_sequence_advances_with_each_frame() -> None:
    worker = _demo_worker()
    worker.start()
    try:
        time.sleep(0.3)
        first = worker.snapshot()
        time.sleep(0.2)
        second = worker.snapshot()
    finally:
        worker.stop()
    assert second.seq > first.seq
    assert second.frame_key != first.frame_key


def test_snapshot_copy_false_avoids_a_full_frame_memcpy() -> None:
    worker = _demo_worker()
    worker.start()
    try:
        time.sleep(0.3)
        shared = worker.snapshot(copy=False)
        owned = worker.snapshot()
    finally:
        worker.stop()
    assert shared.frame is not None
    assert owned.frame is not None
    # copy=True must hand back an array the caller may safely draw on.
    assert owned.frame is not shared.frame


def test_snapshot_copy_false_returns_the_workers_own_array() -> None:
    worker = _demo_worker()
    worker.start()
    try:
        time.sleep(0.3)
        a = worker.snapshot(copy=False)
        b = worker.snapshot(copy=False)
    finally:
        worker.stop()
    if a.seq == b.seq:
        assert a.frame is b.frame
