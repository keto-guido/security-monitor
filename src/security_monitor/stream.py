"""Background capture threads for live cameras and synthetic demo feeds."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from security_monitor.config import CameraConfig, DisplayConfig

Status = str  # live | reconnecting | disconnected | error | demo
_OPEN_LOCK = threading.Lock()


@dataclass
class Snapshot:
    frame: np.ndarray | None
    status: Status
    fps: float = 0.0
    detail: str = ""


class FrameSource(Protocol):
    name: str

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def snapshot(self) -> Snapshot: ...
    def reconnect(self) -> None: ...


def apply_ffmpeg_defaults(transport: str) -> None:
    """Low-latency FFmpeg options used by OpenCV's RTSP/RTP backend."""
    options = [
        "fflags;nobuffer",
        "flags;low_delay",
        "max_delay;500000",
    ]
    if transport == "tcp":
        options.insert(0, "rtsp_transport;tcp")
    elif transport == "udp":
        options.insert(0, "rtsp_transport;udp")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(options)


class CameraWorker(threading.Thread):
    """Read the newest frame from an RTSP/RTP/file/webcam source."""

    def __init__(self, camera: CameraConfig, display: DisplayConfig) -> None:
        super().__init__(name=f"cam-{camera.name}", daemon=True)
        self.camera = camera
        self.display = display
        self.name = camera.name  # type: ignore[assignment]
        self._stop = threading.Event()
        self._kick = threading.Event()
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._status: Status = "disconnected"
        self._detail = "starting"
        self._fps = 0.0
        self._cap: cv2.VideoCapture | None = None

    def start(self) -> None:  # noqa: A003 - Thread.start
        if not self.is_alive():
            super().start()

    def stop(self) -> None:
        self._stop.set()
        self._kick.set()
        self._wake.set()
        self._release()

    def reconnect(self) -> None:
        self._kick.set()
        self._wake.set()
        self._release()

    def snapshot(self) -> Snapshot:
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
            return Snapshot(frame=frame, status=self._status, fps=self._fps, detail=self._detail)

    def run(self) -> None:
        while not self._stop.is_set():
            self._kick.clear()
            self._set_status("reconnecting", "opening stream")
            cap = self._open()
            if cap is None or not cap.isOpened():
                self._set_status("disconnected", "connect failed")
                self._wait_retry()
                continue
            self._cap = cap
            self._set_status("live", "connected")
            last_ok = time.monotonic()
            stamps: list[float] = []
            while not self._stop.is_set() and not self._kick.is_set():
                ok, frame = cap.read()
                now = time.monotonic()
                if not ok or frame is None:
                    if now - last_ok > max(1.0, self.display.read_timeout_ms / 1000):
                        self._set_status("error", "stream stalled")
                        break
                    continue
                last_ok = now
                stamps.append(now)
                stamps = [t for t in stamps if now - t <= 2.0]
                fps = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) >= 2 else 0.0
                with self._lock:
                    self._frame = frame
                    self._fps = fps
                    self._status = "live"
                    self._detail = ""
            immediate = self._kick.is_set()
            self._release()
            if self._stop.is_set():
                break
            self._set_status("reconnecting", "retrying")
            if not immediate:
                self._wait_retry()

    def _open(self) -> cv2.VideoCapture | None:
        source = self.camera.capture_source()
        transport = self.camera.transport or self.display.default_transport
        try:
            with _OPEN_LOCK:
                apply_ffmpeg_defaults(transport)
                cap = _create_capture(source)
        except cv2.error:
            return None
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            return None
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        _try_set(cap, "CAP_PROP_OPEN_TIMEOUT_MSEC", self.display.open_timeout_ms)
        _try_set(cap, "CAP_PROP_READ_TIMEOUT_MSEC", self.display.read_timeout_ms)
        return cap

    def _release(self) -> None:
        cap = self._cap
        self._cap = None
        if cap is not None:
            try:
                cap.release()
            except cv2.error:
                pass

    def _wait_retry(self) -> None:
        self._wake.clear()
        if self._stop.is_set() or self._kick.is_set():
            return
        self._wake.wait(self.display.reconnect_seconds)

    def _set_status(self, status: Status, detail: str) -> None:
        with self._lock:
            self._status = status
            self._detail = detail


class DemoWorker(threading.Thread):
    """Synthetic moving feed so the mosaic can be tested without cameras."""

    PALETTE = (
        (46, 92, 184),
        (42, 140, 86),
        (38, 96, 168),
        (120, 72, 48),
        (96, 64, 140),
        (40, 140, 140),
    )

    def __init__(self, camera: CameraConfig, display: DisplayConfig, index: int) -> None:
        super().__init__(name=f"demo-{camera.name}", daemon=True)
        self.camera = camera
        self.display = display
        self.index = index
        self.name = camera.name  # type: ignore[assignment]
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._fps = float(display.fps)

    def start(self) -> None:  # noqa: A003
        if not self.is_alive():
            super().start()

    def stop(self) -> None:
        self._stop.set()

    def reconnect(self) -> None:
        return

    def snapshot(self) -> Snapshot:
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
        return Snapshot(frame=frame, status="demo", fps=self._fps, detail="synthetic")

    def run(self) -> None:
        width, height = self.display.cell_width, self.display.cell_height
        color = self.PALETTE[self.index % len(self.PALETTE)]
        interval = 1.0 / max(1, self.display.fps)
        started = time.monotonic()
        while not self._stop.is_set():
            t = time.monotonic() - started
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:] = (18, 18, 22)
            band_y = int((np.sin(t + self.index) * 0.5 + 0.5) * (height - 40))
            frame[band_y : band_y + 24, :] = color
            cv2.circle(
                frame,
                (int((t * 80 + self.index * 40) % width), height // 2),
                28,
                color,
                -1,
            )
            label = time.strftime("%H:%M:%S")
            cv2.putText(
                frame,
                f"{self.camera.name}  {label}",
                (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (230, 230, 230),
                2,
                cv2.LINE_AA,
            )
            with self._lock:
                self._frame = frame
            self._stop.wait(interval)


def build_sources(cameras: list[CameraConfig], display: DisplayConfig) -> list[FrameSource]:
    sources: list[FrameSource] = []
    for index, camera in enumerate(cameras):
        url = camera.url or ""
        if url.startswith("demo://"):
            sources.append(DemoWorker(camera, display, index))
        else:
            sources.append(CameraWorker(camera, display))
    return sources


def _create_capture(source: str | int) -> cv2.VideoCapture:
    if isinstance(source, int):
        backend = getattr(cv2, "CAP_DSHOW", None) if os.name == "nt" else None
        if backend is not None:
            return cv2.VideoCapture(source, backend)
        return cv2.VideoCapture(source)
    return cv2.VideoCapture(source, cv2.CAP_FFMPEG)


def _try_set(cap: cv2.VideoCapture, attr: str, value: int) -> None:
    prop = getattr(cv2, attr, None)
    if prop is None:
        return
    try:
        cap.set(prop, value)
    except cv2.error:
        return
