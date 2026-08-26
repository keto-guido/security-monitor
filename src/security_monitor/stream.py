"""Background capture threads for live cameras and synthetic demo feeds."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from security_monitor.buffer import FrameHistory, HistoryView
from security_monitor.config import CameraConfig, DisplayConfig
from security_monitor.decode import apply_ffmpeg_capture_options

Status = str  # live | reconnecting | disconnected | error | demo
_OPEN_LOCK = threading.Lock()


@dataclass
class Snapshot:
    frame: np.ndarray | None
    status: Status
    fps: float = 0.0
    detail: str = ""
    behind: float = 0.0
    buffered: float = 0.0
    rewinding: bool = False
    decode: str = ""  # cpu | auto/vaapi | gpu/cuda | cpu (fallback) | ...


class FrameSource(Protocol):
    name: str
    history: FrameHistory

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def snapshot(self) -> Snapshot: ...
    def reconnect(self) -> None: ...
    def apply_buffer_settings(self, display: DisplayConfig) -> None: ...


def apply_ffmpeg_defaults(
    transport: str,
    *,
    low_latency: bool = True,
    decode_mode: str = "auto",
    hwaccel: str = "auto",
    hwaccel_device: str = "",
    force_cpu: bool = False,
) -> str:
    """FFmpeg options used by OpenCV's RTSP/RTP backend. Returns decode label."""
    return apply_ffmpeg_capture_options(
        transport,
        low_latency=low_latency,
        decode_mode=decode_mode,
        hwaccel=hwaccel,
        hwaccel_device=hwaccel_device,
        force_cpu=force_cpu,
    )


class CameraWorker(threading.Thread):
    """Read frames from an RTSP/RTP/file/webcam source into a rolling history."""

    def __init__(self, camera: CameraConfig, display: DisplayConfig) -> None:
        super().__init__(name=f"cam-{camera.name}", daemon=True)
        self.camera = camera
        self.display = display
        self.name = camera.name  # type: ignore[assignment]
        self.history = FrameHistory()
        self._stop = threading.Event()
        self._kick = threading.Event()
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._status: Status = "disconnected"
        self._detail = "starting"
        self._fps = 0.0
        self._cap: cv2.VideoCapture | None = None
        self._decode = "cpu"
        self.apply_buffer_settings(display)

    def apply_buffer_settings(self, display: DisplayConfig) -> None:
        self.display = display
        self.history.configure(
            smooth_enabled=display.smooth_buffer,
            smooth_seconds=display.smooth_buffer_seconds,
            rewind_enabled=display.rewind_buffer,
            rewind_seconds=display.rewind_buffer_seconds,
            clip_seconds=display.clip_seconds,
        )

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
        self.history.clear()

    def snapshot(self) -> Snapshot:
        with self._lock:
            latest = None if self._frame is None else self._frame
            status = self._status
            fps = self._fps
            detail = self._detail
            decode = self._decode
        view = self._history_view(latest)
        return Snapshot(
            frame=view.frame.copy() if view.frame is not None else None,
            status=status,
            fps=fps,
            detail=detail,
            behind=view.behind,
            buffered=view.buffered,
            rewinding=view.rewinding,
            decode=decode,
        )

    def _history_view(self, latest: np.ndarray | None) -> HistoryView:
        return self.history.view(latest=latest)

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                self._run_session()
            except Exception as exc:  # noqa: BLE001 — OpenCV often raises raw C++ errors
                print(f"{self.name}: capture error ({exc})")
                try:
                    self._release()
                except Exception:  # noqa: BLE001
                    pass
                if self._stop.is_set():
                    break
                self._set_status("error", "capture crashed")
                self._wait_retry()

    def _run_session(self) -> None:
        self._kick.clear()
        self._set_status("reconnecting", "opening stream")
        cap = self._open()
        if cap is None or not _cap_opened(cap):
            self._set_status("disconnected", "connect failed")
            self._wait_retry()
            return
        self._cap = cap
        self._set_status("live", "connected")
        last_ok = time.monotonic()
        stamps: list[float] = []
        while not self._stop.is_set() and not self._kick.is_set():
            ok, frame = safe_cap_read(cap)
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
            frame = rotate_frame(frame, self.camera.rotate)
            self.history.push(frame, when=now)
            with self._lock:
                self._frame = frame
                if self._fps > 0 and fps > 0:
                    self._fps = self._fps * 0.85 + fps * 0.15
                else:
                    self._fps = fps
                self._status = "live"
                self._detail = ""
        immediate = self._kick.is_set()
        self._release()
        if self._stop.is_set():
            return
        self._set_status("reconnecting", "retrying")
        if not immediate:
            self._wait_retry()

    def _open(self) -> cv2.VideoCapture | None:
        source = self.camera.capture_source()
        # Local webcams use OS backends — treat as CPU device capture.
        if isinstance(source, int):
            try:
                cap = _create_capture(source)
            except Exception:  # noqa: BLE001
                return None
            if cap is None or not _cap_opened(cap):
                if cap is not None:
                    _safe_release(cap)
                return None
            with self._lock:
                self._decode = "cpu/device"
            return self._finish_open(cap)

        transport = self.camera.transport or self.display.default_transport
        low_latency = not self.display.smooth_buffer
        decode_mode = getattr(self.display, "decode_mode", "auto")
        hwaccel = getattr(self.display, "hwaccel", "auto")
        hwaccel_device = getattr(self.display, "hwaccel_device", "") or ""

        # Try preferred decode first; on failure fall back to CPU once.
        attempts: list[bool] = [False]
        if (decode_mode or "auto").lower() != "cpu" and (hwaccel or "auto").lower() != "none":
            attempts = [False, True]  # GPU/auto request, then force CPU

        last_label = "cpu"
        for force_cpu in attempts:
            try:
                with _OPEN_LOCK:
                    last_label = apply_ffmpeg_defaults(
                        transport,
                        low_latency=low_latency,
                        decode_mode=decode_mode,
                        hwaccel=hwaccel,
                        hwaccel_device=hwaccel_device,
                        force_cpu=force_cpu,
                    )
                    cap = _create_capture(source)
            except Exception:  # noqa: BLE001
                cap = None
            if cap is not None and _cap_opened(cap):
                # Confirm we can read at least one frame before accepting GPU path.
                if not force_cpu and not self._smoke_read(cap):
                    _safe_release(cap)
                    continue
                label = last_label
                if force_cpu and len(attempts) > 1:
                    label = "cpu (fallback)"
                with self._lock:
                    self._decode = label
                return self._finish_open(cap)
            if cap is not None:
                _safe_release(cap)
        with self._lock:
            self._decode = last_label
        return None

    def _smoke_read(self, cap: cv2.VideoCapture) -> bool:
        """Return True if the capture can produce a frame (GPU path sanity check)."""
        ok, frame = safe_cap_read(cap)
        return bool(ok and frame is not None and getattr(frame, "size", 0) > 0)

    def _finish_open(self, cap: cv2.VideoCapture) -> cv2.VideoCapture:
        buf_size = 4 if self.display.smooth_buffer else 1
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, buf_size)
        except Exception:  # noqa: BLE001
            pass
        _try_set(cap, "CAP_PROP_OPEN_TIMEOUT_MSEC", self.display.open_timeout_ms)
        _try_set(cap, "CAP_PROP_READ_TIMEOUT_MSEC", self.display.read_timeout_ms)
        # Best-effort OpenCV HW acceleration property when present.
        self._try_set_hw_acceleration(cap)
        return cap

    def _try_set_hw_acceleration(self, cap: cv2.VideoCapture) -> None:
        mode = (getattr(self.display, "decode_mode", "auto") or "auto").lower()
        if mode == "cpu":
            return
        prop = getattr(cv2, "CAP_PROP_HW_ACCELERATION", None)
        any_accel = getattr(cv2, "VIDEO_ACCELERATION_ANY", None)
        if prop is None or any_accel is None:
            return
        try:
            cap.set(prop, any_accel)
        except Exception:  # noqa: BLE001
            return

    def _release(self) -> None:
        cap = self._cap
        self._cap = None
        if cap is not None:
            _safe_release(cap)

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
        self.history = FrameHistory()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._fps = float(display.fps)
        self.apply_buffer_settings(display)

    def apply_buffer_settings(self, display: DisplayConfig) -> None:
        self.display = display
        self._fps = float(display.fps)
        self.history.configure(
            smooth_enabled=display.smooth_buffer,
            smooth_seconds=display.smooth_buffer_seconds,
            rewind_enabled=display.rewind_buffer,
            rewind_seconds=display.rewind_buffer_seconds,
            clip_seconds=display.clip_seconds,
        )

    def start(self) -> None:  # noqa: A003
        if not self.is_alive():
            super().start()

    def stop(self) -> None:
        self._stop.set()

    def reconnect(self) -> None:
        self.history.clear()
        return

    def snapshot(self) -> Snapshot:
        with self._lock:
            latest = None if self._frame is None else self._frame
        view = self.history.view(latest=latest)
        return Snapshot(
            frame=view.frame.copy() if view.frame is not None else None,
            status="demo",
            fps=self._fps,
            detail="synthetic",
            behind=view.behind,
            buffered=view.buffered,
            rewinding=view.rewinding,
            decode="cpu/demo",
        )

    def run(self) -> None:
        width, height = self.display.cell_width, self.display.cell_height
        color = self.PALETTE[self.index % len(self.PALETTE)]
        interval = 1.0 / max(1, self.display.fps)
        started = time.monotonic()
        while not self._stop.is_set():
            try:
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
                frame = rotate_frame(frame, self.camera.rotate)
                now = time.monotonic()
                self.history.push(frame, when=now)
                with self._lock:
                    self._frame = frame
            except Exception as exc:  # noqa: BLE001
                print(f"{self.name}: demo error ({exc})")
            self._stop.wait(interval)


def rotate_frame(frame: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate a BGR frame clockwise by 0/90/180/270 degrees."""
    try:
        degrees = int(degrees) % 360
    except (TypeError, ValueError):
        return frame
    if degrees == 0:
        return frame
    flag = {
        90: getattr(cv2, "ROTATE_90_CLOCKWISE", None),
        180: getattr(cv2, "ROTATE_180", None),
        270: getattr(cv2, "ROTATE_90_COUNTERCLOCKWISE", None),
    }.get(degrees)
    if flag is None:
        return frame
    try:
        return cv2.rotate(frame, flag)
    except Exception:  # noqa: BLE001
        return frame


def build_sources(cameras: list[CameraConfig], display: DisplayConfig) -> list[FrameSource]:
    sources: list[FrameSource] = []
    for index, camera in enumerate(cameras):
        url = camera.url or ""
        if url.startswith("demo://"):
            sources.append(DemoWorker(camera, display, index))
        else:
            sources.append(CameraWorker(camera, display))
    return sources


def safe_cap_read(cap: object) -> tuple[bool, np.ndarray | None]:
    """Read one frame; never raise — OpenCV C++ errors must not kill the thread."""
    try:
        ok, frame = cap.read()  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        return False, None
    if not ok or frame is None:
        return False, None
    return True, frame


def _cap_opened(cap: cv2.VideoCapture | None) -> bool:
    if cap is None:
        return False
    try:
        return bool(cap.isOpened())
    except Exception:  # noqa: BLE001
        return False


def _safe_release(cap: cv2.VideoCapture) -> None:
    try:
        cap.release()
    except Exception:  # noqa: BLE001
        return


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
    except Exception:  # noqa: BLE001
        return
