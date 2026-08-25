"""Cross-platform alert beeps for encroachment alarms (no extra deps)."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import threading
import wave
from pathlib import Path


def _write_beep_wav(path: Path, *, frequency: float = 880.0, duration: float = 0.22) -> None:
    import math
    import struct

    rate = 22050
    n = int(rate * duration)
    amplitude = 12000
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        frames = bytearray()
        for i in range(n):
            # Soft attack/decay so it is less harsh on speakers.
            env = 1.0
            if i < rate * 0.02:
                env = i / (rate * 0.02)
            elif i > n - rate * 0.04:
                env = max(0.0, (n - i) / (rate * 0.04))
            sample = int(amplitude * env * math.sin(2 * math.pi * frequency * (i / rate)))
            frames += struct.pack("<h", sample)
        handle.writeframes(frames)


def _play_wav_file(path: Path) -> bool:
    if sys.platform.startswith("win"):
        try:
            import winsound

            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            return True
        except Exception:  # noqa: BLE001
            pass
    players: list[list[str]] = []
    if sys.platform == "darwin":
        players.append(["afplay", str(path)])
    for binary in ("paplay", "aplay", "ffplay"):
        if shutil.which(binary):
            if binary == "ffplay":
                players.append(
                    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]
                )
            elif binary == "aplay":
                players.append(["aplay", "-q", str(path)])
            else:
                players.append([binary, str(path)])
    for cmd in players:
        try:
            subprocess.Popen(  # noqa: S603
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except OSError:
            continue
    return False


def play_alert_beep(*, double: bool = True) -> None:
    """
    Fire a short alert tone in a background thread.

    Best-effort: WAV via OS player, Windows winsound, else terminal bell.
    """

    def _run() -> None:
        try:
            with tempfile.TemporaryDirectory(prefix="sm-alarm-") as tmp:
                path = Path(tmp) / "beep.wav"
                _write_beep_wav(path, frequency=920.0, duration=0.18)
                played = _play_wav_file(path)
                if double and played:
                    import time

                    time.sleep(0.12)
                    _write_beep_wav(path, frequency=1175.0, duration=0.2)
                    _play_wav_file(path)
                if not played:
                    # Last resort — may be ignored by some terminals.
                    sys.stdout.write("\a")
                    sys.stdout.flush()
        except Exception:  # noqa: BLE001
            try:
                sys.stdout.write("\a")
                sys.stdout.flush()
            except Exception:  # noqa: BLE001
                return

    threading.Thread(target=_run, name="sm-alarm", daemon=True).start()
