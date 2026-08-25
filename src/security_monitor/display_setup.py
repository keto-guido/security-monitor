"""Linux display helpers: screen size for layout, optional xrandr rotation."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass


VALID_ROTATIONS = ("none", "normal", "left", "right", "inverted")

_XRANDR_ROTATE = {
    "none": None,
    "normal": "normal",
    "left": "left",
    "right": "right",
    "inverted": "inverted",
}


class DisplaySetupError(RuntimeError):
    """Raised when the active output cannot be queried or rotated."""


@dataclass(frozen=True)
class OutputInfo:
    name: str
    connected: bool
    primary: bool
    width: int | None = None
    height: int | None = None
    rotation: str = "normal"  # normal | left | right | inverted
    preferred: str | None = None  # e.g. 1920x1080


# Rotation sits before the axes parenthetical in real xrandr output.
_OUTPUT_RE = re.compile(
    r"^(?P<name>\S+)\s+(?P<state>connected|disconnected)"
    r"(?:\s+(?P<primary>primary))?"
    r"(?:\s+(?P<geom>(?P<w>\d+)x(?P<h>\d+)\+(?P<x>\d+)\+(?P<y>\d+)))?"
    r"(?:\s+(?P<rotate>normal|left|right|inverted))?"
    r"(?:\s+\([^)]*\))?"
)


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def normalize_rotation(value: str) -> str:
    key = value.strip().lower()
    if key not in VALID_ROTATIONS:
        raise DisplaySetupError(
            f"Invalid rotation {value!r}; use one of {', '.join(VALID_ROTATIONS)}"
        )
    return key


def xrandr_available() -> bool:
    return is_linux() and shutil.which("xrandr") is not None


def _run_xrandr(args: list[str]) -> str:
    if not xrandr_available():
        raise DisplaySetupError("xrandr is not available (Linux X11/XWayland only)")
    try:
        completed = subprocess.run(
            ["xrandr", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DisplaySetupError(f"xrandr failed: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise DisplaySetupError(detail or f"xrandr exited {completed.returncode}")
    return completed.stdout


def list_outputs(query_text: str | None = None) -> list[OutputInfo]:
    """Parse `xrandr` query output into connected/disconnected outputs."""
    text = query_text if query_text is not None else _run_xrandr([])
    outputs: list[OutputInfo] = []
    current: OutputInfo | None = None
    for line in text.splitlines():
        match = _OUTPUT_RE.match(line)
        if match:
            if current is not None:
                outputs.append(current)
            width = int(match.group("w")) if match.group("w") else None
            height = int(match.group("h")) if match.group("h") else None
            current = OutputInfo(
                name=match.group("name"),
                connected=match.group("state") == "connected",
                primary=bool(match.group("primary")),
                width=width,
                height=height,
                rotation=(match.group("rotate") or "normal"),
            )
            continue
        if current is None or not current.connected:
            continue
        mode = re.match(r"^\s+(\d+x\d+)\s+", line)
        if mode and ("*" in line or "+" in line) and current.preferred is None:
            current = OutputInfo(
                name=current.name,
                connected=current.connected,
                primary=current.primary,
                width=current.width,
                height=current.height,
                rotation=current.rotation,
                preferred=mode.group(1),
            )
    if current is not None:
        outputs.append(current)
    return outputs


def pick_output(outputs: list[OutputInfo], name: str | None = None) -> OutputInfo:
    connected = [out for out in outputs if out.connected]
    if not connected:
        raise DisplaySetupError("No connected display outputs found")
    if name and name.strip().lower() not in {"", "auto"}:
        wanted = name.strip()
        for out in connected:
            if out.name == wanted:
                return out
        available = ", ".join(out.name for out in connected)
        raise DisplaySetupError(f"Output {wanted!r} not found. Connected: {available}")
    for out in connected:
        if out.primary:
            return out
    return connected[0]


def screen_size(output: str | None = None) -> tuple[int, int] | None:
    """Return the active output's pixel size, or None if unknown."""
    if not xrandr_available():
        return None
    try:
        target = pick_output(list_outputs(), output)
    except DisplaySetupError:
        return None
    if target.width and target.height and target.width >= 320 and target.height >= 180:
        return int(target.width), int(target.height)
    if target.preferred and "x" in target.preferred:
        try:
            w_s, h_s = target.preferred.lower().split("x", 1)
            width, height = int(w_s), int(h_s)
        except ValueError:
            return None
        if width >= 320 and height >= 180:
            return width, height
    return None


def sanitize_window_size(
    reported: tuple[int, int] | None,
    *,
    fullscreen: bool,
    screen: tuple[int, int] | None,
    fallback: tuple[int, int],
) -> tuple[int, int]:
    """
    Pick a trustworthy paint size for the mosaic.

    OpenCV's Qt HighGUI backend (especially fullscreen) often returns unstable or
    wrong getWindowImageRect() values. Composing at those sizes then letting the
    toolkit stretch the image is what makes the mosaic look squished.
    """
    fb_w, fb_h = fallback
    if fb_w < 320 or fb_h < 180:
        fb_w, fb_h = 1280, 720

    if fullscreen and screen is not None:
        sw, sh = screen
        if sw >= 320 and sh >= 180:
            # Fullscreen: prefer the real output mode over HighGUI's rect.
            if reported is None:
                return sw, sh
            ww, wh = reported
            if ww < 320 or wh < 180:
                return sw, sh
            aspect = ww / wh
            if aspect < 0.85 or aspect > 2.6:
                return sw, sh
            # Allow small insets (panels) but not a totally different mode.
            if abs(ww - sw) > max(32, sw * 0.04) or abs(wh - sh) > max(64, sh * 0.08):
                return sw, sh
            return int(ww), int(wh)

    if reported is None:
        return int(fb_w), int(fb_h)
    ww, wh = reported
    if ww < 320 or wh < 180:
        return int(fb_w), int(fb_h)
    aspect = ww / wh
    if aspect < 0.85 or aspect > 2.6:
        return int(fb_w), int(fb_h)
    return int(ww), int(wh)


def apply_screen_rotate(
    rotation: str,
    *,
    output: str | None = None,
    dry_run: bool = False,
) -> str:
    """Rotate the active monitor with xrandr. Returns a short summary."""
    key = normalize_rotation(rotation)
    xrandr_value = _XRANDR_ROTATE[key]
    if xrandr_value is None:
        return "screen_rotate is none — left display unchanged"

    outputs = list_outputs()
    target = pick_output(outputs, output)
    if target.rotation == xrandr_value and not dry_run:
        return f"{target.name}: already {xrandr_value}"

    args = ["--output", target.name, "--rotate", xrandr_value]
    summary = f"xrandr --output {target.name} --rotate {xrandr_value}"
    if dry_run:
        return f"dry-run: {summary}"
    _run_xrandr(args)
    return f"Applied {summary} (was {target.rotation})"


def describe_outputs() -> str:
    """Multi-line status for `security-monitor display`."""
    if not is_linux():
        return "Display helpers are only available on Linux."
    if not xrandr_available():
        return (
            "xrandr not found. Install x11-xserver-utils (Debian/Ubuntu) "
            "or use Settings → Displays."
        )
    try:
        outputs = list_outputs()
    except DisplaySetupError as exc:
        return f"Could not query displays: {exc}"

    if not outputs:
        return "No outputs reported by xrandr."

    lines = ["Connected displays:"]
    for out in outputs:
        if not out.connected:
            continue
        geom = (
            f"{out.width}x{out.height}"
            if out.width and out.height
            else (out.preferred or "unknown size")
        )
        flags = []
        if out.primary:
            flags.append("primary")
        flags.append(f"rotate={out.rotation}")
        if out.preferred:
            flags.append(f"preferred={out.preferred}")
        lines.append(f"  {out.name}: {geom}  ({', '.join(flags)})")

    disconnected = [out.name for out in outputs if not out.connected]
    if disconnected:
        lines.append("Disconnected: " + ", ".join(disconnected))
    size = screen_size()
    if size:
        lines.append(f"Active screen size used for fullscreen layout: {size[0]}x{size[1]}")
    lines.append(
        "If the mosaic looks squished, this build ignores bad OpenCV window "
        "rects and paints at the screen size above."
    )
    return "\n".join(lines)


def maybe_apply_config_rotation(rotation: str, output: str | None = None) -> str | None:
    """
    Apply config screen_rotate on Linux. Returns a log line, or None if skipped.
    Never raises — startup should still open the mosaic if xrandr fails.
    """
    key = normalize_rotation(rotation) if rotation else "none"
    if key == "none":
        return None
    if not is_linux():
        return f"Ignoring screen_rotate={key!r} (not Linux)"
    try:
        return apply_screen_rotate(key, output=output)
    except DisplaySetupError as exc:
        return f"screen_rotate failed: {exc}"
