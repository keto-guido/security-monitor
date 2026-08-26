"""GPU / CPU video decode preferences for OpenCV FFmpeg capture."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from functools import lru_cache

VALID_DECODE_MODES = ("auto", "cpu", "gpu")
VALID_HWACCELS = ("auto", "none", "cuda", "qsv", "vaapi", "d3d11va", "videotoolbox")

# Prefer these when mode=gpu/auto and hwaccel=auto.
_LINUX_PREF = ("vaapi", "cuda", "qsv")
_WINDOWS_PREF = ("d3d11va", "qsv", "cuda")
_MAC_PREF = ("videotoolbox",)


def platform_hwaccels() -> tuple[str, ...]:
    """Backends that are even theoretically valid on this OS."""
    system = platform.system().lower()
    if system == "windows":
        return _WINDOWS_PREF
    if system == "darwin":
        return _MAC_PREF
    return _LINUX_PREF


def hwaccel_menu_choices() -> tuple[str, ...]:
    """Decode backends the UI should offer — probed + this OS, never a foreign GPU API."""
    found = set(probe_hwaccels())
    extras = [name for name in platform_hwaccels() if name in found]
    return ("auto", "none", *extras)


def sanitize_hwaccel(hwaccel: str) -> str:
    """
    Map a saved/requested backend to something this machine can try.

    Foreign APIs (D3D on Linux, VAAPI on Windows) and backends the probe did
    not see become ``auto`` so OpenCV/FFmpeg is not asked to open them.
    """
    choice = (hwaccel or "auto").strip().lower()
    if choice in {"none", "off", "cpu"}:
        return "none"
    if choice in {"", "auto"}:
        return "auto"
    if choice not in platform_hwaccels():
        return "auto"
    available = set(probe_hwaccels())
    if available and choice not in available:
        return "auto"
    if not available:
        return "auto"
    return choice


def decode_mode_label(mode: str, hwaccel: str) -> str:
    mode = (mode or "auto").lower()
    hwaccel = (hwaccel or "auto").lower()
    if mode == "cpu" or hwaccel == "none":
        return "CPU"
    if mode == "gpu":
        if hwaccel in {"auto", "none"}:
            return "GPU (auto backend)"
        return f"GPU ({hwaccel})"
    # auto
    if hwaccel not in {"auto", "none"}:
        return f"Auto → try {hwaccel}"
    return "Auto (prefer GPU, fall back CPU)"


def next_decode_mode(current: str, step: int = 1) -> str:
    values = list(VALID_DECODE_MODES)
    try:
        index = values.index((current or "auto").lower())
    except ValueError:
        index = 0
    return values[(index + int(step)) % len(values)]


def next_hwaccel(
    current: str,
    step: int = 1,
    *,
    choices: tuple[str, ...] | None = None,
) -> str:
    values = list(choices if choices is not None else hwaccel_menu_choices())
    if not values:
        values = ["auto", "none"]
    key = (current or "auto").lower()
    if key not in values:
        index = 0
        return values[(index + int(step)) % len(values)] if step else values[0]
    index = values.index(key)
    return values[(index + int(step)) % len(values)]


@lru_cache(maxsize=1)
def probe_hwaccels() -> tuple[str, ...]:
    """
    Best-effort list of FFmpeg hwaccels available on this machine.

    Uses ``ffmpeg -hwaccels`` when present, plus OpenCV build info hints.
    Stock ``opencv-python`` wheels often still decode on CPU even when a
    name appears here — treat results as "may be requestable".
    """
    found: set[str] = set()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        try:
            proc = subprocess.run(
                [ffmpeg, "-hide_banner", "-hwaccels"],
                check=False,
                capture_output=True,
                text=True,
                timeout=4,
            )
            text = (proc.stdout or "") + "\n" + (proc.stderr or "")
            for line in text.splitlines():
                name = line.strip().lower()
                if name in VALID_HWACCELS or name == "videotoolbox":
                    found.add(name)
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        import cv2

        info = cv2.getBuildInformation()
        lower = info.lower()
        if "cuda" in lower and "yes" in lower:
            found.add("cuda")
        if "vaapi" in lower:
            found.add("vaapi")
        if "qsv" in lower or "mfx" in lower:
            found.add("qsv")
        if "d3d11" in lower or "dxva" in lower:
            found.add("d3d11va")
    except Exception:  # noqa: BLE001
        pass
    # Only advertise backends this OS can actually request. OpenCV build
    # strings often mention D3D/VideoToolbox on every platform.
    allowed = set(platform_hwaccels())
    ordered = [
        name
        for name in ("cuda", "qsv", "vaapi", "d3d11va", "videotoolbox")
        if name in found and name in allowed
    ]
    return tuple(ordered)


@lru_cache(maxsize=1)
def opencv_decode_summary() -> str:
    """One-line OpenCV/FFmpeg capability summary for menus and ``check``."""
    try:
        import cv2
    except ImportError:
        return "OpenCV not installed"
    ver = getattr(cv2, "__version__", "?")
    info = ""
    try:
        info = cv2.getBuildInformation()
    except Exception:  # noqa: BLE001
        return f"OpenCV {ver}"
    ffmpeg = "yes" if re.search(r"FFMPEG:\s+YES", info, re.I) else "no"
    cuda = "yes" if re.search(r"CUDA:\s+YES", info, re.I) else "no"
    accels = ", ".join(probe_hwaccels()) or "none detected"
    return f"OpenCV {ver}  FFmpeg={ffmpeg}  CUDA={cuda}  hwaccels=[{accels}]"


def preferred_hwaccel(hwaccel: str = "auto") -> str | None:
    """Resolve which hwaccel name to request, or None for software decode."""
    choice = sanitize_hwaccel(hwaccel)
    if choice in {"none", "off", "cpu"}:
        return None
    available = set(probe_hwaccels())
    if choice != "auto":
        if available and choice not in available:
            return None
        if choice not in platform_hwaccels():
            return None
        return choice
    for name in platform_hwaccels():
        if name in available:
            return name
    return None


def resolve_decode_request(decode_mode: str, hwaccel: str) -> tuple[str | None, str]:
    """
    Return ``(hwaccel_or_none, human_label)`` for the next open attempt.

    ``None`` means software/CPU decode options only. Never force a backend
    this OS / probe does not support — that path crashes OpenCV on Linux.
    """
    mode = (decode_mode or "auto").lower()
    hw = sanitize_hwaccel(hwaccel)
    if mode == "cpu" or hw in {"none", "off", "cpu"}:
        return None, "cpu"
    accel = preferred_hwaccel(hw)
    if accel is None:
        return None, "cpu"
    if mode == "gpu":
        return accel, f"gpu/{accel}"
    return accel, f"auto/{accel}"


def ffmpeg_option_pairs(
    transport: str,
    *,
    low_latency: bool = True,
    decode_mode: str = "auto",
    hwaccel: str = "auto",
    hwaccel_device: str = "",
    force_cpu: bool = False,
) -> list[tuple[str, str]]:
    """Build OpenCV ``OPENCV_FFMPEG_CAPTURE_OPTIONS`` key/value pairs."""
    pairs: list[tuple[str, str]] = []
    if transport == "tcp":
        pairs.append(("rtsp_transport", "tcp"))
    elif transport == "udp":
        pairs.append(("rtsp_transport", "udp"))
    if low_latency:
        pairs.extend(
            [
                ("fflags", "nobuffer"),
                ("flags", "low_delay"),
                ("max_delay", "500000"),
            ]
        )
    else:
        pairs.extend(
            [
                ("fflags", "+genpts"),
                ("max_delay", "2000000"),
            ]
        )
    if not force_cpu:
        accel, _label = resolve_decode_request(decode_mode, hwaccel)
        if accel:
            pairs.append(("hwaccel", accel))
            device = (hwaccel_device or "").strip()
            if device:
                pairs.append(("hwaccel_device", device))
            elif accel == "vaapi" and sys.platform.startswith("linux"):
                # Common default render node; ignored if missing.
                if os.path.exists("/dev/dri/renderD128"):
                    pairs.append(("hwaccel_device", "/dev/dri/renderD128"))
    return pairs


def format_ffmpeg_capture_options(pairs: list[tuple[str, str]]) -> str:
    return "|".join(f"{key};{value}" for key, value in pairs)


def apply_ffmpeg_capture_options(
    transport: str,
    *,
    low_latency: bool = True,
    decode_mode: str = "auto",
    hwaccel: str = "auto",
    hwaccel_device: str = "",
    force_cpu: bool = False,
) -> str:
    """Set ``OPENCV_FFMPEG_CAPTURE_OPTIONS`` and return the decode label used."""
    pairs = ffmpeg_option_pairs(
        transport,
        low_latency=low_latency,
        decode_mode=decode_mode,
        hwaccel=hwaccel,
        hwaccel_device=hwaccel_device,
        force_cpu=force_cpu,
    )
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = format_ffmpeg_capture_options(pairs)
    if force_cpu:
        return "cpu"
    _accel, label = resolve_decode_request(decode_mode, hwaccel)
    return label if not force_cpu else "cpu"
