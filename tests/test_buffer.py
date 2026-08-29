"""Tests for smooth / rewind frame history."""

from __future__ import annotations

import time

import numpy as np
import pytest

from security_monitor import buffer as buffer_module
from security_monitor.buffer import (
    HISTORY_PROFILES,
    REWIND_BUFFER_CHOICES,
    SMOOTH_BUFFER_CHOICES,
    FrameHistory,
    next_choice,
    resolve_history_mode,
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


def test_history_always_retains_clip_window() -> None:
    hist = FrameHistory()
    hist.configure(clip_seconds=2)
    t0 = time.monotonic()
    for i in range(5):
        hist.push(_frame(i * 20), when=t0 + i * 0.4)
    frames, fps = hist.export_frames(2)
    assert len(frames) >= 3
    assert fps > 0


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


def _fill(history: FrameHistory, *, fps: float, seconds: float, start: float = 100.0) -> None:
    """Feed the history as a camera thread would: one push per decoded frame."""
    step = 1.0 / fps
    now = start
    for i in range(int(fps * seconds)):
        history.push(_frame(i % 200), when=now)
        now += step


def test_push_honours_the_sample_rate_cap() -> None:
    """The cap has to save CPU, not just memory.

    It used to re-encode every frame anyway to "refresh the tip", so a 25 fps
    camera paid 25 JPEG encodes a second to keep a 12-sample-per-second ring.
    """
    history = FrameHistory()
    history.configure(history_mode="full", clip_seconds=15.0)
    calls: list[int] = []
    original = buffer_module._encode

    def counting_encode(frame, **kwargs):
        calls.append(1)
        return original(frame, **kwargs)

    buffer_module._encode = counting_encode
    try:
        _fill(history, fps=25.0, seconds=2.0)
    finally:
        buffer_module._encode = original
    # 50 frames arrived; the store is capped at 12/s, so it must encode far
    # fewer than one per frame — previously it encoded all 50.
    cap = HISTORY_PROFILES["full"].max_fps
    assert len(calls) <= cap * 2 + 1, len(calls)
    assert len(calls) < 25, len(calls)


def test_smooth_playback_still_gets_every_frame() -> None:
    """Smooth buffering reads the store, so it must not be rate-capped."""
    history = FrameHistory()
    history.configure(smooth_enabled=True, smooth_seconds=1.0, history_mode="lite")
    calls: list[int] = []
    original = buffer_module._encode

    def counting_encode(frame, **kwargs):
        calls.append(1)
        return original(frame, **kwargs)

    buffer_module._encode = counting_encode
    try:
        _fill(history, fps=25.0, seconds=1.0)
    finally:
        buffer_module._encode = original
    assert len(calls) == 25


def test_history_off_stores_nothing_when_nobody_reads_it() -> None:
    history = FrameHistory()
    history.configure(history_mode="off", clip_seconds=15.0)
    assert history.active is False
    _fill(history, fps=25.0, seconds=1.0)
    assert history.span_seconds() == 0.0


def test_history_off_still_serves_smooth_and_rewind() -> None:
    history = FrameHistory()
    history.configure(history_mode="off", rewind_enabled=True, rewind_seconds=10.0)
    assert history.active is True
    _fill(history, fps=25.0, seconds=2.0)
    assert history.span_seconds() > 0.5


def test_live_view_bypasses_the_store_while_rewind_sits_at_live() -> None:
    """Rewind being armed must not cap live playback at the store's rate."""
    history = FrameHistory()
    history.configure(rewind_enabled=True, rewind_seconds=10.0, history_mode="lite")
    _fill(history, fps=25.0, seconds=2.0)
    latest = _frame(123)
    view = history.view(latest=latest)
    assert view.frame is latest  # no JPEG round trip, no staleness
    assert view.stamp == 0.0
    assert view.rewinding is False


def test_scrubbed_view_comes_from_the_store() -> None:
    history = FrameHistory()
    history.configure(rewind_enabled=True, rewind_seconds=10.0, history_mode="full")
    _fill(history, fps=25.0, seconds=3.0)
    history.nudge_rewind(1.0)
    view = history.view(latest=_frame(123))
    assert view.rewinding is True
    assert view.stamp > 0.0
    assert view.behind == pytest.approx(1.0, abs=0.3)


def test_view_decode_is_memoized_for_repeat_calls() -> None:
    """The mosaic asks several times per composed frame — decode once."""
    history = FrameHistory()
    history.configure(smooth_enabled=True, smooth_seconds=0.5, history_mode="full")
    _fill(history, fps=25.0, seconds=2.0)
    first = history.view()
    second = history.view()
    assert first.frame is second.frame


def test_lite_profile_is_cheaper_than_full() -> None:
    lite, full = HISTORY_PROFILES["lite"], HISTORY_PROFILES["full"]
    assert lite.max_edge < full.max_edge
    assert lite.jpeg_quality <= full.jpeg_quality
    assert lite.max_fps < full.max_fps
    assert lite.retain is True  # clips and pre-roll still work


def test_resolve_history_mode_passes_explicit_choices_through() -> None:
    assert resolve_history_mode("full") == "full"
    assert resolve_history_mode("off") == "off"
    assert resolve_history_mode("auto") in {"full", "lite"}
    assert resolve_history_mode("nonsense") in {"full", "lite"}
