"""Tests for GPU/CPU decode preferences."""

from __future__ import annotations

import os

import pytest

from security_monitor.config import ConfigError, parse_config, save_display_settings
from security_monitor.decode import (
    apply_ffmpeg_capture_options,
    decode_mode_label,
    ffmpeg_option_pairs,
    format_ffmpeg_capture_options,
    next_decode_mode,
    next_hwaccel,
    resolve_decode_request,
)


def test_next_decode_mode_cycles() -> None:
    assert next_decode_mode("auto", 1) == "cpu"
    assert next_decode_mode("cpu", 1) == "gpu"
    assert next_decode_mode("gpu", 1) == "auto"
    assert next_decode_mode("auto", -1) == "gpu"


def test_next_hwaccel_cycles() -> None:
    assert next_hwaccel("auto", 1) == "none"
    assert next_hwaccel("none", -1) == "auto"
    assert next_hwaccel("vaapi", 1) == "d3d11va"


def test_decode_mode_label() -> None:
    assert decode_mode_label("cpu", "auto") == "CPU"
    assert decode_mode_label("gpu", "none") == "CPU"
    assert "vaapi" in decode_mode_label("gpu", "vaapi")
    assert "prefer GPU" in decode_mode_label("auto", "auto")


def test_resolve_cpu_and_none() -> None:
    assert resolve_decode_request("cpu", "cuda") == (None, "cpu")
    assert resolve_decode_request("gpu", "none") == (None, "cpu")
    assert resolve_decode_request("auto", "none") == (None, "cpu")


def test_ffmpeg_option_pairs_cpu_omits_hwaccel() -> None:
    pairs = ffmpeg_option_pairs(
        "tcp",
        decode_mode="gpu",
        hwaccel="cuda",
        force_cpu=True,
    )
    keys = [k for k, _ in pairs]
    assert "rtsp_transport" in keys
    assert "hwaccel" not in keys


def test_ffmpeg_option_pairs_gpu_requests_hwaccel() -> None:
    pairs = ffmpeg_option_pairs(
        "tcp",
        decode_mode="gpu",
        hwaccel="vaapi",
        hwaccel_device="/dev/dri/renderD128",
    )
    assert ("hwaccel", "vaapi") in pairs
    assert ("hwaccel_device", "/dev/dri/renderD128") in pairs
    text = format_ffmpeg_capture_options(pairs)
    assert "hwaccel;vaapi" in text
    assert "rtsp_transport;tcp" in text


def test_apply_ffmpeg_capture_options_sets_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", raising=False)
    label = apply_ffmpeg_capture_options(
        "tcp",
        decode_mode="cpu",
        hwaccel="auto",
    )
    assert label == "cpu"
    env = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")
    assert "hwaccel" not in env
    assert "rtsp_transport;tcp" in env


def test_config_parses_decode_settings() -> None:
    cfg = parse_config(
        {
            "display": {
                "decode_mode": "gpu",
                "hwaccel": "qsv",
                "hwaccel_device": "/dev/dri/renderD129",
            },
            "cameras": [{"name": "A", "url": "rtsp://192.168.1.10/live"}],
        }
    )
    assert cfg.display.decode_mode == "gpu"
    assert cfg.display.hwaccel == "qsv"
    assert cfg.display.hwaccel_device == "/dev/dri/renderD129"


def test_config_rejects_bad_decode_mode() -> None:
    with pytest.raises(ConfigError, match="decode_mode"):
        parse_config(
            {
                "display": {"decode_mode": "vulkan"},
                "cameras": [{"name": "A", "url": "rtsp://192.168.1.10/live"}],
            }
        )


def test_save_display_settings_persists_decode(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "display:\n  columns: 2\ncameras:\n  - name: A\n    url: rtsp://1.2.3.4/x\n",
        encoding="utf-8",
    )
    cfg = parse_config(
        {
            "display": {"decode_mode": "auto", "hwaccel": "auto"},
            "cameras": [{"name": "A", "url": "rtsp://1.2.3.4/x"}],
        },
        path=path,
    )
    cfg.display.decode_mode = "cpu"
    cfg.display.hwaccel = "none"
    cfg.display.hwaccel_device = "/dev/dri/renderD128"
    assert save_display_settings(cfg) == path
    text = path.read_text(encoding="utf-8")
    assert "decode_mode: cpu" in text
    assert "hwaccel: none" in text
    assert "hwaccel_device: /dev/dri/renderD128" in text
    reloaded = parse_config(
        __import__("yaml").safe_load(text),
        path=path,
    )
    assert reloaded.display.decode_mode == "cpu"
    assert reloaded.display.hwaccel == "none"
