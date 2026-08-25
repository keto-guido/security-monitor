from __future__ import annotations

import os

from security_monitor.config import CameraConfig, parse_config
from security_monitor.reboot import camera_host, ping_command, reboot_targets


def test_ping_command_is_platform_specific() -> None:
    cmd = ping_command("192.168.1.10")
    assert cmd[-1] == "192.168.1.10"
    if os.name == "nt":
        assert cmd[:3] == ["ping", "-n", "1"]
    else:
        assert cmd[:3] == ["ping", "-c", "1"]


def test_reboot_targets_from_config() -> None:
    cfg = parse_config(
        {
            "cameras": [
                {
                    "name": "Porch",
                    "url": "rtsp://192.168.1.10:554/live",
                    "type": "ubiquiti",
                    "username": "cam",
                    "password": "secret",
                },
                {
                    "name": "Yard",
                    "url": "rtsp://192.168.1.11/s0",
                    "type": "reolink",
                    "username": "cam",
                    "password": "secret",
                    "http_port": 80,
                },
                {"name": "Webcam", "device": 0},
                {
                    "name": "Skipped",
                    "url": "rtsp://192.168.1.12/live",
                    "type": "ubiquiti",
                    "username": "cam",
                    "password": "secret",
                    "reboot": False,
                },
                {
                    "name": "NoType",
                    "url": "rtsp://192.168.1.13/live",
                    "username": "cam",
                    "password": "secret",
                },
            ]
        }
    )
    devices = reboot_targets(cfg.cameras)
    assert [d.name for d in devices] == ["Porch", "Yard"]
    assert devices[0].kind == "ubiquiti"
    assert devices[0].host == "192.168.1.10"
    assert devices[1].kind == "reolink"


def test_camera_host_ignores_webcams() -> None:
    assert camera_host(CameraConfig(name="Cam", url="rtsp://10.0.0.5:554/x")) == "10.0.0.5"
    assert camera_host(CameraConfig(name="Web", device=0)) is None
