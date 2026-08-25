"""Rolling frame history for smooth playback delay and quick rewind."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

# Menu presets (seconds).
SMOOTH_BUFFER_CHOICES = (0.5, 1.0, 1.5, 2.0, 3.0)
REWIND_BUFFER_CHOICES = (10, 15, 30, 60, 120)

# Keep rewind memory bounded: JPEG + max long-edge pixels + sample rate.
_REWIND_MAX_EDGE = 960
_REWIND_JPEG_QUALITY = 70
_REWIND_MAX_FPS = 12.0


@dataclass(frozen=True)
class HistoryView:
    """A frame sampled from history plus how far behind live it is."""

    frame: np.ndarray | None
    behind: float = 0.0  # seconds behind the newest capture
    buffered: float = 0.0  # seconds of history currently held
    rewinding: bool = False


def next_choice(choices: tuple[float, ...] | tuple[int, ...], current: float, step: int) -> float:
    """Cycle through preset lengths. step=+1 forward, -1 backward."""
    if not choices:
        return current
    values = [float(v) for v in choices]
    # Snap to nearest preset, then move.
    nearest = min(range(len(values)), key=lambda i: abs(values[i] - float(current)))
    index = (nearest + int(step)) % len(values)
    return values[index]


def _encode(frame: np.ndarray) -> tuple[int, int, bytes]:
    h, w = frame.shape[:2]
    edge = max(h, w)
    if edge > _REWIND_MAX_EDGE:
        scale = _REWIND_MAX_EDGE / edge
        frame = cv2.resize(
            frame,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        h, w = frame.shape[:2]
    ok, buf = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), _REWIND_JPEG_QUALITY],
    )
    if not ok:
        # Extremely unlikely; store a tiny placeholder.
        blank = np.zeros((max(1, h), max(1, w), 3), dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", blank)
    return h, w, buf.tobytes()


def _decode(payload: bytes) -> np.ndarray | None:
    arr = np.frombuffer(payload, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


class FrameHistory:
    """
    Thread-safe ring of recent frames.

    - smooth_buffer: display the frame from ``smooth_seconds`` ago (absorbs jitter)
    - rewind_buffer: keep up to ``rewind_seconds`` for scrubbing with an offset
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: deque[tuple[float, bytes]] = deque()
        self.smooth_enabled = False
        self.smooth_seconds = 1.0
        self.rewind_enabled = False
        self.rewind_seconds = 30.0
        self.rewind_offset = 0.0  # 0 = live (or smooth-delay point)
        self._last_store_t = 0.0

    @property
    def retention_seconds(self) -> float:
        hold = 0.0
        if self.smooth_enabled:
            hold = max(hold, float(self.smooth_seconds) + 0.5)
        if self.rewind_enabled:
            hold = max(hold, float(self.rewind_seconds) + 0.5)
        return hold

    @property
    def active(self) -> bool:
        return self.smooth_enabled or self.rewind_enabled

    def configure(
        self,
        *,
        smooth_enabled: bool | None = None,
        smooth_seconds: float | None = None,
        rewind_enabled: bool | None = None,
        rewind_seconds: float | None = None,
    ) -> None:
        with self._lock:
            if smooth_enabled is not None:
                self.smooth_enabled = bool(smooth_enabled)
            if smooth_seconds is not None:
                self.smooth_seconds = max(0.1, float(smooth_seconds))
            if rewind_enabled is not None:
                self.rewind_enabled = bool(rewind_enabled)
            if rewind_seconds is not None:
                self.rewind_seconds = max(1.0, float(rewind_seconds))
            if not self.rewind_enabled:
                self.rewind_offset = 0.0
            else:
                self.rewind_offset = min(self.rewind_offset, self.rewind_seconds)
            self._trim_locked(time.monotonic())

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self.rewind_offset = 0.0
            self._last_store_t = 0.0

    def push(self, frame: np.ndarray, when: float | None = None) -> None:
        if frame is None or not self.active:
            return
        now = time.monotonic() if when is None else float(when)
        with self._lock:
            # When only rewind is on, sample at a capped rate to save memory.
            # Smooth playback needs denser samples, so skip the cap then.
            min_gap = 0.0
            if self.rewind_enabled and not self.smooth_enabled:
                min_gap = 1.0 / _REWIND_MAX_FPS
            if min_gap > 0 and self._items and (now - self._last_store_t) < min_gap:
                # Refresh the tip so live viewers still see a current end frame.
                _h, _w, payload = _encode(frame)
                self._items[-1] = (now, payload)
                self._trim_locked(now)
                return
            _h, _w, payload = _encode(frame)
            self._items.append((now, payload))
            self._last_store_t = now
            self._trim_locked(now)

    def nudge_rewind(self, delta_seconds: float) -> float:
        """Move the rewind playhead. Positive delta goes further into the past."""
        with self._lock:
            if not self.rewind_enabled:
                self.rewind_offset = 0.0
                return 0.0
            available = 0.0
            if self._items:
                available = max(0.0, self._items[-1][0] - self._items[0][0])
            limit = min(float(self.rewind_seconds), available)
            self.rewind_offset = min(max(0.0, self.rewind_offset + float(delta_seconds)), limit)
            return self.rewind_offset

    def go_live(self) -> None:
        with self._lock:
            self.rewind_offset = 0.0

    def view(self, latest: np.ndarray | None = None) -> HistoryView:
        """
        Pick the frame that should be shown right now.

        ``latest`` is used when history is inactive or empty so callers can fall
        back to the live capture without an extra copy when buffering is off.
        """
        with self._lock:
            if not self.active or not self._items:
                return HistoryView(frame=None if latest is None else latest, behind=0.0, buffered=0.0)

            newest_t = self._items[-1][0]
            oldest_t = self._items[0][0]
            buffered = max(0.0, newest_t - oldest_t)
            delay = float(self.smooth_seconds) if self.smooth_enabled else 0.0
            target = newest_t - delay - float(self.rewind_offset)
            if target < oldest_t:
                target = oldest_t
            # Nearest sample at or before target (prefer not to show the future).
            chosen = self._items[0]
            for item in self._items:
                if item[0] <= target:
                    chosen = item
                else:
                    break
            frame = _decode(chosen[1])
            behind = max(0.0, newest_t - chosen[0])
            return HistoryView(
                frame=frame if frame is not None else latest,
                behind=behind,
                buffered=buffered,
                rewinding=self.rewind_offset > 0.05,
            )

    def span_seconds(self) -> float:
        with self._lock:
            if len(self._items) < 2:
                return 0.0
            return max(0.0, self._items[-1][0] - self._items[0][0])

    def _trim_locked(self, now: float) -> None:
        keep = self.retention_seconds
        if keep <= 0:
            self._items.clear()
            return
        cutoff = now - keep
        while self._items and self._items[0][0] < cutoff:
            self._items.popleft()
