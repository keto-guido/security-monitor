"""Crash-safe startup and low-power FPS policy for weaker machines."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

VALID_POWER_MODES = ("auto", "on", "off")
POWER_MODE_CHOICES = VALID_POWER_MODES


def state_dir() -> Path:
    """Per-user state (crash marker). Mirrors config.yaml's XDG / APPDATA root."""
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "security-monitor"
        return Path.home() / "security-monitor"
    xdg = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    return Path(xdg) / "security-monitor"


def crash_marker_path() -> Path:
    return state_dir() / "unclean-exit"


def crash_marker_present() -> bool:
    return crash_marker_path().is_file()


def write_crash_marker() -> Path:
    path = crash_marker_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{time.time():.3f}\n", encoding="utf-8")
    return path


def clear_crash_marker() -> None:
    path = crash_marker_path()
    try:
        path.unlink()
    except FileNotFoundError:
        return


def should_start_safe_mode(
    *,
    force: bool = False,
    ignore: bool = False,
    on_crash: bool = True,
) -> bool:
    """Decide whether this process should start with extras disabled."""
    if ignore:
        return False
    if force:
        return True
    return bool(on_crash and crash_marker_present())


class CrashGuard:
    """
    Write a crash marker at start; clear it only after a clean exit.

    If the process is killed or raises, the next launch can enter safe mode.
    """

    def __init__(self) -> None:
        self._armed = True

    def __enter__(self) -> CrashGuard:
        write_crash_marker()
        return self

    def disarm(self) -> None:
        self._armed = False

    def __exit__(self, exc_type, exc, _tb) -> bool:
        intentional = exc_type is None or exc_type is KeyboardInterrupt
        if (not self._armed) or intentional:
            clear_crash_marker()
        return False


@dataclass
class PowerPolicy:
    mode: str = "auto"  # auto | on | off
    fps_threshold: float = 12.0
    enter_seconds: float = 2.0
    exit_seconds: float = 5.0

    @property
    def recover_fps(self) -> float:
        return float(self.fps_threshold) + 3.0


@dataclass
class PowerTracker:
    """Hysteresis around UI FPS so low-power does not flicker on and off."""

    policy: PowerPolicy = field(default_factory=PowerPolicy)
    low_power: bool = False
    _below_since: float | None = None
    _above_since: float | None = None

    def reset(self) -> None:
        self._below_since = None
        self._above_since = None

    def set_mode(self, mode: str) -> bool:
        mode = str(mode).strip().lower()
        if mode not in VALID_POWER_MODES:
            mode = "auto"
        changed_mode = self.policy.mode != mode
        self.policy.mode = mode
        if mode == "on":
            changed = not self.low_power or changed_mode
            self.low_power = True
            self.reset()
            return changed
        if mode == "off":
            changed = self.low_power or changed_mode
            self.low_power = False
            self.reset()
            return changed
        if changed_mode:
            self.reset()
        return changed_mode

    def update(self, ui_fps: float, now: float) -> bool:
        """
        Update ``low_power`` from a UI FPS sample.

        Returns True when the low-power flag flipped.
        """
        if self.policy.mode == "on":
            if not self.low_power:
                self.low_power = True
                return True
            return False
        if self.policy.mode == "off":
            if self.low_power:
                self.low_power = False
                return True
            return False
        if ui_fps <= 0.5:
            return False
        threshold = float(self.policy.fps_threshold)
        if not self.low_power:
            if ui_fps < threshold:
                if self._below_since is None:
                    self._below_since = now
                elif now - self._below_since >= self.policy.enter_seconds:
                    self.low_power = True
                    self.reset()
                    return True
            else:
                self._below_since = None
            return False
        if ui_fps >= self.policy.recover_fps:
            if self._above_since is None:
                self._above_since = now
            elif now - self._above_since >= self.policy.exit_seconds:
                self.low_power = False
                self.reset()
                return True
        else:
            self._above_since = None
        return False


@dataclass
class RateMeter:
    """
    Sliding-window "events per second" estimate.

    Used for the rate that actually reaches the screen, so it has to fall to
    zero when nothing is being shown — a meter anchored to the last two events
    would happily keep reporting 25 fps through a stall.
    """

    window: float = 2.0
    _stamps: list[float] = field(default_factory=list)
    _started: float | None = None

    def reset(self) -> None:
        self._stamps.clear()
        self._started = None

    def mark(self, now: float) -> None:
        if self._started is None:
            self._started = now
        self._stamps.append(now)
        self._trim(now)

    def _trim(self, now: float) -> None:
        cutoff = now - self.window
        stamps = self._stamps
        while stamps and stamps[0] < cutoff:
            stamps.pop(0)

    def rate(self, now: float) -> float:
        """Events per second over the trailing window, anchored at ``now``."""
        if self._started is None:
            return 0.0
        self._trim(now)
        span = min(self.window, now - self._started)
        if span < 0.25 or not self._stamps:
            return 0.0
        return len(self._stamps) / span


@dataclass
class FramePacer:
    """
    Deadline-based pacing for the render loop.

    Sleeping a fixed ``1/fps`` *after* composing makes the real period
    ``compose + 1/fps``, so a 25 fps target with a 40 ms compose actually runs
    at 12.5 fps — and wobbles with every change in compose cost, which is what
    a viewer reads as jerky. Pacing to a moving deadline spends the slack that
    is left instead, and stays on the same phase frame to frame.
    """

    target_fps: float = 25.0
    # Frames of lateness tolerated before giving up on catching up. Without
    # this, a long stall (menu, window resize, reconnect storm) leaves a
    # backlog of missed deadlines that the loop sprints through at zero wait.
    max_catch_up: float = 2.0
    _deadline: float | None = None

    @property
    def period(self) -> float:
        return 1.0 / max(1.0, float(self.target_fps))

    def set_target_fps(self, fps: float) -> None:
        fps = max(1.0, float(fps))
        if fps != self.target_fps:
            self.target_fps = fps
            self._deadline = None

    def reset(self) -> None:
        self._deadline = None

    def wait_seconds(self, now: float) -> float:
        """Time to idle before the next frame is due. Advances the deadline."""
        period = self.period
        if self._deadline is None:
            self._deadline = now + period
            return period
        remaining = self._deadline - now
        if remaining < -period * self.max_catch_up:
            self._deadline = now + period
            return 0.0
        self._deadline += period
        return max(0.0, remaining)

    def wait_ms(self, now: float, *, minimum: int = 1) -> int:
        """``wait_seconds`` as whole milliseconds, never 0.

        OpenCV reads ``waitKey(0)`` as "block until a key is pressed", so the
        floor matters: returning 0 here would freeze the video.
        """
        return max(int(minimum), int(round(self.wait_seconds(now) * 1000.0)))


def power_mode_label(mode: str, *, active: bool = False, threshold: float = 12.0) -> str:
    key = str(mode).strip().lower()
    if key == "on":
        return "On — pause extras (video + HUD)"
    if key == "off":
        return "Off — detection and extras run"
    if active:
        return f"Auto — ON now (<{threshold:g} fps)"
    return f"Auto (if UI FPS < {threshold:g})"
