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


def power_mode_label(mode: str, *, active: bool = False, threshold: float = 12.0) -> str:
    key = str(mode).strip().lower()
    if key == "on":
        return "On (video + HUD only)"
    if key == "off":
        return "Off"
    if active:
        return f"Auto — ON now (<{threshold:g} fps)"
    return f"Auto (if UI FPS < {threshold:g})"
