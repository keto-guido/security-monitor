"""Tests for smooth / rewind frame history."""

from __future__ import annotations

import time

import numpy as np
import pytest

from security_monitor.buffer import (
    REWIND_BUFFER_CHOICES,
    SMOOTH_BUFFER_CHOICES,
    FrameHistory,
    next_choice,
)
from security_monitor.config import parse_config, save_display_settings


def _frame(value: int) -> np.ndarray:
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    img[:] = (value, value, value)
    return img


def test_next_choice_cycles() -> None:
    assert next_choice(SMOOTH_BUFFER_CHOICES, 1.0, 1) == 1.5
    assert next_choice(SMOOTH_BUFFER_CHOICES, 3.0, 1) == 0.5
    assert next_choice(REWIND_BUFFER_CHOICES, 30, -1) == 15


def test_smooth_view_returns_delayed_frame() -> None:
    hist = FrameHistory()
    hist.configure(smooth_enabled=True, smooth_seconds=0.2, rewind_enabled=False)
    t0 = time.monotonic()
    hist.push(_frame(10), when=t0)
    hist.push(_frame(80), when=t0 + 0.1)
    hist.push(_frame(200), when=t0 + 0.25)
    view = hist.view()
    assert view.frame is not None
    # Target is newest - 0.2s ≈ t0+0.05 → first/second sample, not the live 200.
    assert int(view.frame[0, 0, 0]) < 150
    assert view.behind >= 0.1


def test_rewind_offset_scrub() -> None:
    hist = FrameHistory()
    hist.configure(smooth_enabled=False, rewind_enabled=True, rewind_seconds=5)
    t0 = time.monotonic()
    for i in range(6):
        hist.push(_frame(i * 10), when=t0 + i * 0.5)
    assert hist.nudge_rewind(1.0) == pytest.approx(1.0)
    view = hist.view()
    assert view.rewinding
    assert view.behind >= 0.9
    hist.go_live()
    assert hist.view().rewinding is False


def test_inactive_history_does_not_store() -> None:
    hist = FrameHistory()
    hist.push(_frame(1))
    assert hist.span_seconds() == 0.0
    assert hist.view(latest=_frame(9)).frame is not None
    assert int(hist.view(latest=_frame(9)).frame[0, 0, 0]) == 9


def test_config_parses_buffer_flags() -> None:
    cfg = parse_config(
        {
            "display": {
                "smooth_buffer": True,
                "smooth_buffer_seconds": 2,
                "rewind_buffer": True,
                "rewind_buffer_seconds": 45,
            },
            "cameras": [{"name": "A", "url": "rtsp://192.168.1.10/live"}],
        }
    )
    assert cfg.display.smooth_buffer is True
    assert cfg.display.smooth_buffer_seconds == 2.0
    assert cfg.display.rewind_buffer is True
    assert cfg.display.rewind_buffer_seconds == 45.0


def test_save_display_settings_roundtrip(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "display:\n  columns: 2\ncameras:\n  - name: A\n    url: rtsp://1.2.3.4/x\n",
        encoding="utf-8",
    )
    cfg = parse_config(
        {
            "display": {"smooth_buffer": True, "smooth_buffer_seconds": 1.5},
            "cameras": [{"name": "A", "url": "rtsp://1.2.3.4/x"}],
        },
        path=path,
    )
    cfg.display.rewind_buffer = True
    cfg.display.rewind_buffer_seconds = 60
    assert save_display_settings(cfg) == path
    text = path.read_text(encoding="utf-8")
    assert "smooth_buffer: true" in text
    assert "rewind_buffer: true" in text
    assert "rewind_buffer_seconds: 60" in text
