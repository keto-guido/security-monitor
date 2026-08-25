# Security Monitor

Command-line viewer for multiple IP cameras. It reads camera URLs and layout from a YAML config file, then shows a live mosaic window on Windows and Linux.

Default layout is **2×2 (4 cameras)**. Streams can be `rtsp://` or `rtp://` (HTTP MJPEG, local files, and webcams work too).

## Install

Python 3.10+ is required. From the project directory:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux
source .venv/bin/activate

pip install -e .
```

That installs the `security-monitor` command.

## Quick start

1. Create a local config (it is gitignored so credentials stay off git):

```bash
security-monitor init
```

2. Edit `config.yaml` with your camera URLs, usernames, and the grid size you want.

3. Confirm the file parses:

```bash
security-monitor check
```

4. Open the mosaic:

```bash
security-monitor
```

No cameras yet? Try synthetic panes:

```bash
security-monitor --demo
```

## Config

`config.yaml` has two sections: `display` and `cameras`.

```yaml
display:
  columns: 2
  rows: 2
  cell_width: 640
  cell_height: 360
  scale_mode: fit          # fit | fill | stretch
  window_title: Security Monitor
  fullscreen: false
  show_labels: true
  show_fps: true
  fps: 25
  reconnect_seconds: 5
  default_transport: tcp   # tcp | udp | auto
  open_timeout_ms: 8000
  read_timeout_ms: 5000

cameras:
  - name: Front Door
    url: rtsp://192.168.1.10:554/Streaming/Channels/101
    username: camera
    password: secret
    type: ubiquiti          # ubiquiti | reolink | amcrest | dahua
    enabled: true
    transport: tcp
```

Config search order:

1. `--config path/to/config.yaml`
2. `SECURITY_MONITOR_CONFIG`
3. `./config.yaml`
4. `%APPDATA%\security-monitor\config.yaml` on Windows, or `~/.config/security-monitor/config.yaml` on Linux

Only the first `columns * rows` **enabled** cameras are shown. Raise the grid to display more.

### Camera URLs

Put credentials in `username` / `password` instead of the URL. Typical manufacturer paths:

| Brand / type | Example path |
| --- | --- |
| Hikvision | `rtsp://HOST:554/Streaming/Channels/101` |
| Dahua | `rtsp://HOST:554/cam/realmonitor?channel=1&subtype=0` |
| Generic RTSP | `rtsp://HOST:554/stream1` |
| RTP | `rtp://HOST:5004` |
| Webcam | `device: 0` (no `url`) |
| File | `url: C:/clips/cam.mp4` |

`tcp` transport is the better default through routers. Use `udp` only if you need lower latency and the path is clean.

## Controls

| Key | Action |
| --- | --- |
| `Esc` | Back to the grid; on the grid, open the options menu |
| `q` | Quit (or Exit from the options menu) |
| `f` | Toggle fullscreen |
| `1`–`9` | Focus that camera |
| `g` / `0` | Back to the grid |
| Scroll wheel | Zoom in / out (toward the cursor) |
| `+` / `-` | Zoom in / out |
| Arrow keys | Pan / strafe when zoomed; move in the options menu |
| `Enter` | Choose the highlighted options-menu item |
| `Home` | Reset zoom |
| `r` | Reconnect every stream |
| `h` | Toggle on-screen help |
| Click a tile | Focus camera (click again for grid; click while zoomed resets zoom) |

The options menu also has **Reboot cameras**. That uses each camera's `type`, host, and credentials from `config.yaml` (Ubiquiti over SSH, Reolink/Amcrest/Dahua over HTTP). You will be asked to confirm. Progress shows in the window; when it finishes, streams reconnect.

## Commands

```bash
security-monitor                 # mosaic from config.yaml
security-monitor -c other.yaml
security-monitor --fullscreen
security-monitor --columns 3 --rows 2
security-monitor demo
security-monitor check
security-monitor reboot
security-monitor init
python -m security_monitor       # same as security-monitor
```

## How it works

Each camera is read on its own thread and only the latest frame is kept, so a slow stream does not stall the others. The UI thread composites a grid, letterboxes (or crops) each tile, and draws name / status / FPS. Dropped streams retry on `reconnect_seconds`.

OpenCV’s FFmpeg backend is used for RTSP/RTP. The GUI package must be `opencv-python`, not `opencv-python-headless`.

## Troubleshooting

- **NO SIGNAL / connect failed** — verify the URL in VLC first, then try `transport: tcp`.
- **Window never appears** — `pip uninstall opencv-python-headless` then `pip install opencv-python`.
- **High latency** — `tcp` is stable but buffered; `udp` is snappier. Cell size also drives decode cost.
- **Linux display** — needs an X11/Wayland session. SSH needs X forwarding or a desktop.
- **Passwords with `@` or `:`** — use the `username` and `password` fields so they are escaped.

## Development

```bash
pip install -e ".[dev]"
pytest
```
