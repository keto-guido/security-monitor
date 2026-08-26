"""Capture-thread helpers must not raise when OpenCV misbehaves."""

from __future__ import annotations

import numpy as np

from security_monitor.stream import safe_cap_read


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
