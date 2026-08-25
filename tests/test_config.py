from __future__ import annotations

from pathlib import Path

import pytest

from security_monitor.config import (
    ConfigError,
    demo_config,
    example_config_text,
    inject_credentials,
    load_config,
    parse_config,
    redact_url,
)


def test_default_grid_is_four_tiles() -> None:
    cfg = parse_config(
        {
            "cameras": [
                {"name": "A", "url": "rtsp://192.168.1.10/live"},
                {"name": "B", "url": "rtsp://192.168.1.11/live"},
                {"name": "C", "url": "rtsp://192.168.1.12/live"},
                {"name": "D", "url": "rtsp://192.168.1.13/live"},
            ]
        }
    )
    assert cfg.display.columns == 2
    assert cfg.display.rows == 2
    assert cfg.display.tile_count == 4
    assert len(cfg.visible_cameras()) == 4


def test_extra_cameras_are_truncated(capsys: pytest.CaptureFixture[str]) -> None:
    cfg = parse_config(
        {
            "display": {"columns": 1, "rows": 1},
            "cameras": [
                {"name": "A", "url": "rtsp://192.168.1.10/live"},
                {"name": "B", "url": "rtsp://192.168.1.11/live"},
            ],
        }
    )
    assert [cam.name for cam in cfg.visible_cameras()] == ["A"]
    assert "exceed" in capsys.readouterr().err


def test_disabled_cameras_skipped() -> None:
    cfg = parse_config(
        {
            "cameras": [
                {"name": "A", "url": "rtsp://192.168.1.10/live", "enabled": False},
                {"name": "B", "url": "rtsp://192.168.1.11/live"},
            ]
        }
    )
    assert [cam.name for cam in cfg.visible_cameras()] == ["B"]


def test_duplicate_names_rejected() -> None:
    with pytest.raises(ConfigError, match="Duplicate"):
        parse_config(
            {
                "cameras": [
                    {"name": "Porch", "url": "rtsp://192.168.1.10/live"},
                    {"name": "porch", "url": "rtsp://192.168.1.11/live"},
                ]
            }
        )


def test_credentials_injected_and_redacted() -> None:
    url = inject_credentials("rtsp://192.168.1.10:554/stream", "user", "p@ss:word")
    assert "p@ss" not in redact_url(url)
    assert "***:***@" in redact_url(url)
    assert "user" in url
    assert "p%40ss%3Aword" in url


def test_existing_userinfo_not_overwritten() -> None:
    original = "rtsp://old:pw@192.168.1.10/live"
    assert inject_credentials(original, "new", "secret") == original


def test_webcam_and_file_sources() -> None:
    cfg = parse_config(
        {
            "cameras": [
                {"name": "Webcam", "device": 0},
                {"name": "Clip", "url": r"C:\clips\cam.mp4"},
            ]
        }
    )
    assert cfg.cameras[0].capture_source() == 0
    assert cfg.cameras[0].redacted_source() == "device:0"
    assert cfg.cameras[1].url.endswith("cam.mp4")


def test_unsupported_scheme() -> None:
    with pytest.raises(ConfigError, match="scheme"):
        parse_config({"cameras": [{"name": "X", "url": "ftp://example/x"}]})


def test_missing_camera_source() -> None:
    with pytest.raises(ConfigError, match="url or device"):
        parse_config({"cameras": [{"name": "X"}]})


def test_demo_config_matches_grid() -> None:
    cfg = demo_config(3, 2)
    assert cfg.display.tile_count == 6
    assert len(cfg.cameras) == 6
    assert all(cam.url and cam.url.startswith("demo://") for cam in cfg.cameras)


def test_load_config_from_disk(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "display:\n  columns: 1\n  rows: 2\ncameras:\n  - name: Gate\n    url: rtp://192.168.1.20:5004\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.cameras[0].url and cfg.cameras[0].url.startswith("rtp://")
    assert cfg.display.canvas_size == (640, 720)


def test_example_config_is_packaged() -> None:
    text = example_config_text()
    assert "display:" in text
    assert "cameras:" in text
