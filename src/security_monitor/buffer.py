"""Rolling frame history for smooth playback delay and quick rewind."""

from __future__ import annotations

import os
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

HISTORY_MODE_CHOICES = ("auto", "full", "lite", "off")

# Machines at or below this core count default to the "lite" store under "auto".
_AUTO_LITE_MAX_CORES = 4


@dataclass(frozen=True)
class HistoryProfile:
    """How much CPU the rolling clip buffer is allowed to spend per frame.

    Every stored sample costs a downscale plus a JPEG encode on the capture
    thread, for every camera, whether or not anything ever reads it back. On a
    weak CPU that alone can eat a core, so the store is tunable independently
    of the playback features that read from it.
    """

    name: str
    max_edge: int = _REWIND_MAX_EDGE
    jpeg_quality: int = _REWIND_JPEG_QUALITY
    max_fps: float = _REWIND_MAX_FPS
    retain: bool = True


HISTORY_PROFILES = {
    # Today's behaviour: near-source stills, 12 samples/second.
    "full": HistoryProfile("full", 960, 70, 12.0, True),
    # Half the pixels, cheaper JPEG, half the sample rate — clips and person
    # pre-roll still work, they are just softer.
    "lite": HistoryProfile("lite", 640, 55, 6.0, True),
    # Nothing is stored unless smooth/rewind needs it. "Save clip" then records
    # forward from the keypress instead of including the preceding seconds.
    "off": HistoryProfile("off", 640, 55, 6.0, False),
}


def resolve_history_mode(mode: str) -> str:
    """Map ``auto`` to the concrete store profile for this machine."""
    key = (mode or "auto").strip().lower()
    if key in HISTORY_PROFILES:
        return key
    return "lite" if (os.cpu_count() or 1) <= _AUTO_LITE_MAX_CORES else "full"


def history_profile(mode: str) -> HistoryProfile:
    return HISTORY_PROFILES[resolve_history_mode(mode)]


def history_mode_label(mode: str) -> str:
    key = (mode or "auto").strip().lower()
    if key == "full":
        return "Full — best clip quality, highest CPU"
    if key == "lite":
        return "Lite — softer clips, much lower CPU"
    if key == "off":
        return "Off — no pre-roll (clips record forward only)"
    return f"Auto ({resolve_history_mode(key)} on this machine)"


@dataclass(frozen=True)
class HistoryView:
    """A frame sampled from history plus how far behind live it is."""

    frame: np.ndarray | None
    behind: float = 0.0  # seconds behind the newest capture
    buffered: float = 0.0  # seconds of history currently held
    rewinding: bool = False
    # Identity of the sample shown: the history timestamp, or 0.0 when this is
    # the live frame. Renderers use it to tell "same picture as last time".
    stamp: float = 0.0


def next_choice(choices: tuple[float, ...] | tuple[int, ...], current: float, step: int) -> float:
    """Cycle through preset lengths. step=+1 forward, -1 backward."""
    if not choices:
        return current
    values = [float(v) for v in choices]
    # Snap to nearest preset, then move.
    nearest = min(range(len(values)), key=lambda i: abs(values[i] - float(current)))
    index = (nearest + int(step)) % len(values)
    return values[index]


def _encode(
    frame: np.ndarray,
    *,
    max_edge: int = _REWIND_MAX_EDGE,
    quality: int = _REWIND_JPEG_QUALITY,
) -> tuple[int, int, bytes]:
    h, w = frame.shape[:2]
    edge = max(h, w)
    if edge > max_edge:
        scale = max_edge / edge
        frame = cv2.resize(
            frame,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        h, w = frame.shape[:2]
    ok, buf = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
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
        self.clip_seconds = 15.0  # retain this much for Save clip when storing
        self.rewind_offset = 0.0  # 0 = live (or smooth-delay point)
        self.profile = HISTORY_PROFILES["full"]
        self._last_store_t = 0.0
        # Memoized decode of the most recently viewed sample: the mosaic asks
        # several times per composed frame (tile, detection, clip feed).
        self._decoded_at = 0.0
        self._decoded: np.ndarray | None = None

    @property
    def retention_seconds(self) -> float:
        # Keep enough history to export a clip, plus optional smooth/rewind.
        hold = 0.0
        if self.profile.retain:
            hold = max(1.0, float(self.clip_seconds) + 0.5)
        if self.smooth_enabled:
            hold = max(hold, float(self.smooth_seconds) + 0.5)
        if self.rewind_enabled:
            hold = max(hold, float(self.rewind_seconds) + 0.5)
        return hold

    @property
    def active(self) -> bool:
        """Whether anything actually reads the store — if not, do not fill it."""
        return bool(self.profile.retain or self.smooth_enabled or self.rewind_enabled)

    @property
    def store_max_fps(self) -> float:
        """Sample cap for the store. Smooth playback reads it, so it is uncapped."""
        if self.smooth_enabled:
            return 0.0
        return float(self.profile.max_fps)

    def _store_params(self) -> tuple[int, int]:
        """Downscale/quality for stored samples.

        Smooth playback puts these frames on screen, so it always gets the
        full-detail store no matter how cheap the profile is.
        """
        if self.smooth_enabled:
            full = HISTORY_PROFILES["full"]
            return full.max_edge, full.jpeg_quality
        return self.profile.max_edge, self.profile.jpeg_quality

    def configure(
        self,
        *,
        smooth_enabled: bool | None = None,
        smooth_seconds: float | None = None,
        rewind_enabled: bool | None = None,
        rewind_seconds: float | None = None,
        clip_seconds: float | None = None,
        history_mode: str | None = None,
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
            if clip_seconds is not None:
                self.clip_seconds = max(1.0, float(clip_seconds))
            if history_mode is not None:
                self.profile = history_profile(history_mode)
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
            self._decoded_at = 0.0
            self._decoded = None

    def push(self, frame: np.ndarray, when: float | None = None) -> None:
        """
        Store a sample if — and only if — something will read it back.

        This runs on the capture thread for every decoded frame of every
        camera, and a downscale + JPEG encode is not cheap. Two guards keep it
        from burning a core on nobody's behalf:

        * nothing reads the store (``active`` is False) → store nothing;
        * the sample-rate cap has not elapsed → return without encoding.

        The second case used to re-encode anyway to "refresh the tip", which
        made the cap free up memory but no CPU. Live viewers never read the tip
        (``view`` hands back the capture frame), so the work was pure waste.
        """
        if frame is None:
            return
        if not self.active:
            return
        now = time.monotonic() if when is None else float(when)
        with self._lock:
            max_fps = self.store_max_fps
            min_gap = (1.0 / max_fps) if max_fps > 0 else 0.0
            if min_gap > 0 and self._items and (now - self._last_store_t) < min_gap:
                self._trim_locked(now)
                return
            max_edge, quality = self._store_params()
            _h, _w, payload = _encode(frame, max_edge=max_edge, quality=quality)
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

    def _needs_sample(self) -> bool:
        """True only when the picture on screen must come out of the store.

        Rewind being *enabled* is not enough — while the playhead sits at live,
        the newest stored sample is a rate-capped, downscaled JPEG of a frame
        the caller already holds at full rate. Serving that would cap live
        playback at the store's sample rate and pay a decode per tile per
        frame, for a strictly worse picture.
        """
        return bool(self.smooth_enabled or self.rewind_offset > 0.05)

    def _decode_cached(self, stamp: float, payload: bytes) -> np.ndarray | None:
        """Decode a sample, reusing the last one when it is the same sample."""
        if self._decoded is not None and self._decoded_at == stamp:
            return self._decoded
        frame = _decode(payload)
        self._decoded_at = stamp
        self._decoded = frame
        return frame

    def view(self, latest: np.ndarray | None = None) -> HistoryView:
        """
        Pick the frame that should be shown right now.

        ``latest`` is used whenever the live capture is the right picture, so
        the common case costs no JPEG decode and no copy.
        """
        with self._lock:
            buffered = (
                max(0.0, self._items[-1][0] - self._items[0][0])
                if len(self._items) >= 2
                else 0.0
            )
            if not self._needs_sample() or not self._items:
                return HistoryView(
                    frame=None if latest is None else latest,
                    behind=0.0,
                    buffered=buffered,
                    stamp=0.0,
                )

            newest_t = self._items[-1][0]
            oldest_t = self._items[0][0]
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
            frame = self._decode_cached(chosen[0], chosen[1])
            behind = max(0.0, newest_t - chosen[0])
            return HistoryView(
                frame=frame if frame is not None else latest,
                behind=behind,
                buffered=buffered,
                rewinding=self.rewind_offset > 0.05,
                stamp=chosen[0] if frame is not None else 0.0,
            )

    def export_frames(self, seconds: float) -> tuple[list[np.ndarray], float]:
        """
        Decode frames covering the last ``seconds`` of history.

        Returns (frames, estimated_fps). Frames are oldest→newest.
        """
        seconds = max(0.5, float(seconds))
        with self._lock:
            if not self._items:
                return [], 0.0
            newest_t = self._items[-1][0]
            cutoff = newest_t - seconds
            selected = [item for item in self._items if item[0] >= cutoff]
            if not selected:
                selected = [self._items[-1]]
            payloads = list(selected)
        frames: list[np.ndarray] = []
        times: list[float] = []
        for when, payload in payloads:
            frame = _decode(payload)
            if frame is None:
                continue
            frames.append(frame)
            times.append(when)
        fps = 0.0
        if len(times) >= 2:
            span = times[-1] - times[0]
            if span > 0:
                fps = (len(times) - 1) / span
        return frames, fps

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
