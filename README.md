# Security Monitor

Desktop mosaic viewer for IP cameras on **Windows** and **Linux**. It reads camera URLs and layout from a YAML file, captures each stream on its own thread, and composites a live grid with zoom, rewind, snapshots, camera reboot, and optional people / left-behind-object detection.

**Version 0.2.0** · Python 3.10+ · OpenCV GUI (not headless)

Default layout is **2×2 (4 cameras)**. Streams can be `rtsp://` or `rtp://` (HTTP MJPEG, local files, and webcams work too).

## Features

- Live mosaic from RTSP / RTP / HTTP / files / webcams, one capture thread per camera
- Focus a tile, scroll-wheel zoom toward the cursor, pan when zoomed
- Smooth playback buffer and rolling rewind (`,` / `.` / `l`)
- Snapshots (`s`) and short clips (`c`), with an on-screen Capture menu
- Opt-in **YOLOv8n** people boxes and lighting-normalized “new object” (package) detection
- Reboot Ubiquiti (SSH) or Reolink / Amcrest / Dahua (HTTP) cameras from the menu or CLI
- Ubuntu login autostart (XDG `.desktop` or systemd user unit) and Windows Startup folder
- Linux fullscreen uses the real screen size from `xrandr` so OpenCV Qt does not squash the grid

## Install

Python 3.10+ and a desktop session (Windows, or Linux X11/Wayland) are required.

One-line install (recommended):

```bash
python -m pip install -U "git+https://github.com/keto-guido/security-monitor.git"
```

That installs the `security-monitor` command and Python dependencies (`opencv-python`, Ultralytics, etc.).

### Ubuntu system packages

On Ubuntu Desktop, install these once before the pip line (venv optional if you use `--user`):

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv \
  fonts-dejavu-core libgl1 libglib2.0-0
```

### Editable / from a clone

```bash
git clone https://github.com/keto-guido/security-monitor.git
cd security-monitor
python -m pip install -e .
```

## Quick start

1. Create a local config (it is gitignored so credentials stay off git):

```bash
security-monitor init
```

On Ubuntu, prefer the per-user path so login autostart finds it regardless of working directory:

```bash
security-monitor init --user
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

## Run on Ubuntu startup

After the app works from a terminal on Ubuntu Desktop:

```bash
# Optional: keep config in ~/.config/security-monitor/config.yaml
security-monitor init --user
# edit that file, then:

security-monitor check
security-monitor autostart install --fullscreen
```

That writes `~/.config/autostart/security-monitor.desktop` so the mosaic launches after you log into the desktop. A short delay (default 15s) waits for the network and session to settle. Useful flags:

```bash
security-monitor autostart install --fullscreen --delay 20
security-monitor autostart install --config /path/to/config.yaml --fullscreen
security-monitor autostart status
security-monitor autostart uninstall
```

Linux notes:

- Default method is an XDG `.desktop` autostart entry (best for GNOME / Ubuntu Desktop).
- Optional: `security-monitor autostart install --method systemd` installs a systemd user unit under `graphical-session.target`.
- OpenCV windows prefer X11 (`QT_QPA_PLATFORM=xcb`) on Wayland so the GUI opens reliably.
- Autostart embeds an absolute path to your config when one is found, so it does not depend on the project folder being the current directory at login.

On Windows, the same commands install a Startup-folder `.bat` instead.

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
  # Linux fullscreen uses the real screen size (fixes OpenCV Qt squish)
  # screen_rotate: none
  # screen_output: auto
  smooth_buffer: false           # delay playback for smoother video
  smooth_buffer_seconds: 1.0
  rewind_buffer: false           # rolling history for quick rewind
  rewind_buffer_seconds: 30
  # save_directory: ~/security-monitor/captures
  snapshot_format: jpg           # jpg | png
  clip_seconds: 15               # default Save clip length
  people_detection: false        # master switch (also enable per camera)
  object_detection: false        # packages / left-behind items vs baseline

cameras:
  - name: Front Door
    url: rtsp://192.168.1.10:554/Streaming/Channels/101
    username: camera
    password: secret
    type: ubiquiti          # ubiquiti | reolink | amcrest | dahua
    enabled: true
    transport: tcp
    # rotate: 180           # 0 | 90 | 180 | 270 if the camera hangs upside-down
    # detect_people: false  # opt this camera into people boxes
    # detect_objects: false # opt this camera into "new object" boxes
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
| Arrow keys | Pan / strafe when zoomed; move in the options menu; scrub rewind when rewind buffer is on |
| `,` / `.` | Rewind back / toward live (when rewind buffer is on) |
| `l` | Jump to live |
| `s` | Save a snapshot (focused camera, or full mosaic) |
| `c` | Save a clip of `clip_seconds` (from recent history, or record live) |
| `Enter` | Choose the highlighted options-menu item |
| `Home` | Reset zoom |
| `r` | Reconnect every stream |
| `h` | Toggle on-screen help |
| Click a tile | Focus camera (click again for grid; click again while zoomed resets zoom) |

The options menu also has **Capture** (snapshot, clip length/format, save folder), **Detection** (people / new-object masters, per-camera includes, set empty-area baseline), **Video settings** (smooth buffer + rewind), and **Reboot cameras**. Settings are saved back into `config.yaml` when changed. Files go to `~/security-monitor/captures` by default (override with `display.save_directory`). Baselines are stored under `~/.config/security-monitor/baselines/` (or `%APPDATA%\security-monitor\baselines` on Windows). Reboot uses each camera's `type`, host, and credentials from `config.yaml` (Ubiquiti over SSH, Reolink/Amcrest/Dahua over HTTP). You will be asked to confirm. Progress shows in the window; when it finishes, streams reconnect.

### Detection notes

**Stack:** Ultralytics **YOLOv8n** (COCO) is the primary people detector — real-time neural detection. If YOLO can’t load, the app falls back to MobileNet-SSD, then OpenCV HOG. New packages use a **lighting-normalized baseline diff** (CLAHE + adaptive threshold + persistence). YOLO “thing” classes (backpack, suitcase, handbag, …) that overlap a change blob confirm arrivals faster. People are masked out of the change map so walkers don’t look like packages.

**Seasonal drift:** You seed an empty-area baseline once. After that the baseline **slowly adapts** (hours-scale EMA) to leaves, snow texture, and lawn growth, and is saved back to disk periodically. Pixels under a confirmed package (and under people) are **frozen**, so a box left for days keeps its `new object` flag instead of being absorbed. A whole-scene shift with no packages (e.g. overnight snow) speeds up adaptation after it settles. Manual **Set empty-area baseline** still resets everything when you want a clean slate.

- **People** and **new object** detection are off until you opt in globally **and** per camera (`Esc` → Detection).
- **Set empty-area baseline** while the porch/yard is clear (no packages). New persistent blobs vs that baseline get a `new object` box until they disappear.
- First run downloads `yolov8n.pt` via Ultralytics (cached). OpenCV fallback models (if needed) go under `~/.cache/security-monitor/models/`.

## Commands

```bash
security-monitor                 # mosaic from config.yaml
security-monitor -c other.yaml
security-monitor --fullscreen
security-monitor --delay 15      # wait before opening (login / network)
security-monitor --columns 3 --rows 2
security-monitor demo
security-monitor check
security-monitor reboot
security-monitor init
security-monitor init --user     # ~/.config/security-monitor/config.yaml
security-monitor display         # show active screen size used for layout
security-monitor display --rotate left   # optional xrandr orientation tweak
security-monitor autostart install --fullscreen
security-monitor autostart status
security-monitor autostart uninstall
python -m security_monitor       # same as security-monitor
```

## How it works

```
config.yaml ──► CLI ──► MosaicApp (UI thread)
                          │
                          ├─ CameraWorker threads ──► OpenCV/FFmpeg RTSP
                          │         └─ FrameHistory (smooth delay / rewind JPEG ring)
                          ├─ DetectionEngine (YOLOv8n → MobileNet-SSD → HOG)
                          │         └─ NewObjectTracker (CLAHE baseline + freeze mask)
                          └─ Capture / Reboot jobs
```

Each camera is read on its own thread and only the latest frame is kept (plus optional history), so a slow stream does not stall the others. The UI thread composites a grid, letterboxes (or crops) each tile, and draws name / status / FPS. Dropped streams retry on `reconnect_seconds`.

OpenCV’s FFmpeg backend is used for RTSP/RTP. The GUI package must be `opencv-python`, not `opencv-python-headless`.

### Layout

| Path | Role |
| --- | --- |
| `src/security_monitor/cli.py` | argparse entry point (`run`, `demo`, `check`, `init`, `reboot`, `autostart`, `display`) |
| `src/security_monitor/config.py` | YAML load / validate / persist menu settings |
| `src/security_monitor/stream.py` | Per-camera capture threads and FFmpeg options |
| `src/security_monitor/buffer.py` | Smooth-delay and rewind ring (JPEG-compressed) |
| `src/security_monitor/mosaic.py` | OpenCV window, HUD, menus, zoom, input |
| `src/security_monitor/detection.py` | People + left-behind object pipeline |
| `src/security_monitor/capture.py` | Snapshots and MP4 clips |
| `src/security_monitor/reboot.py` | SSH / HTTP camera reboot with ping wait |
| `src/security_monitor/autostart.py` | XDG / systemd / Windows Startup installers |
| `src/security_monitor/display_setup.py` | Linux `xrandr` size and rotation |
| `src/security_monitor/overlay.py` | TrueType HUD drawing via Pillow |

## Troubleshooting

- **NO SIGNAL / connect failed** — verify the URL in VLC first, then try `transport: tcp`.
- **Window never appears** — `pip uninstall opencv-python-headless` then `pip install opencv-python`.
- **High latency** — `tcp` is stable but buffered; `udp` is snappier. Cell size also drives decode cost.
- **Linux display** — needs an X11/Wayland session. SSH needs X forwarding or a desktop.
- **Squished / stretched mosaic on Linux** — OpenCV’s Qt window backend often reports the wrong window size (especially fullscreen). This build ignores bad rects and paints at the real screen size from `xrandr`. Update to the latest package, then:
  ```bash
  security-monitor display          # confirm detected screen size
  security-monitor --fullscreen
  ```
  Also disable fractional scaling for the session, or log into “Ubuntu on Xorg”, if the whole desktop (not just this app) looks distorted.
- **Camera feed upside-down** — set `rotate: 180` (or `90` / `270`) on that camera in `config.yaml`.
- **Autostart did nothing** — confirm you log into a desktop user session (not only SSH). Run `security-monitor autostart status` and check `~/.config/autostart/security-monitor.desktop`. Increase `--delay` if cameras are offline at boot.
- **Wayland blank window** — the app sets `QT_QPA_PLATFORM=xcb`; also try logging into an “Ubuntu on Xorg” session.
- **Passwords with `@` or `:`** — use the `username` and `password` fields so they are escaped.
- **Mosaic hitches when detection is on** — inference currently runs on the UI thread (with a ~0.3s cache). See [Suggested upgrades](#suggested-upgrades).

## Development

```bash
pip install -e ".[dev]"
pytest
```

Tests cover config parsing, CLI `init`/`check`, buffers, capture paths, detection helpers, autostart file generation, reboot targeting, and mosaic geometry. They do not open a live GUI or pull RTSP.

## Suggested upgrades

These come from a pass over the current tree (`mosaic.py` ~1.5k lines, detection on the UI thread, plaintext credentials, no CI). Highest payoff first.

### 1. Keep the mosaic smooth while detecting

`DetectionEngine.process()` is called from `_render_cell` on the UI thread. YOLO is cached (~3 fps) and frames are downscaled to 960px, but a slow inference still stalls `imshow`. Move people/object inference onto a worker per camera (or a shared pool), draw the last boxes on the UI thread, and add a config knob for interval / model size (`yolov8n` vs `s`).

### 2. Make heavy ML optional

`ultralytics` is a hard install dependency even when detection stays off. Split extras, for example:

```bash
pip install security-monitor            # viewer only
pip install "security-monitor[detect]"  # YOLO + torch
```

Keep the MobileNet-SSD / HOG fallbacks for machines that should not pull PyTorch.

### 3. In-app camera management

A Cameras menu (grid size, show/hide, reorder, add RTSP/webcam, focus cycling with `n`/`p`) is already prototyped on branch `cursor/buffer-rewind-menu-66f4`. Finishing and merging that removes most hand-edits of `config.yaml` after first install.

### 4. Alerts, not just boxes

Detection currently draws overlays. Next step for a monitor kiosk:

- Snapshot + optional clip when a person or new object is confirmed
- Webhook / MQTT / ntfy / email
- Offline-camera alerts after N reconnect failures
- A small on-disk event log (timestamp, camera, kind, image path)

### 5. Secrets and config hygiene

Passwords live in plaintext YAML. Prefer OS keyring, `password_env: CAMERA_FRONT_PASSWORD`, or a `0600` secrets file. When the menu saves settings, `yaml.safe_dump` rewrites the file and **drops comments** — switch persist to [ruamel.yaml](https://yaml.readthedocs.io/) round-trip so the example comments survive.

### 6. Split the UI module and add CI

`mosaic.py` owns windowing, menus, capture HUD, reboot overlay, and input. Split menu / input / layout so each stays testable. Add GitHub Actions (`pytest` on Ubuntu + Windows), Ruff, and a single version source (`importlib.metadata.version("security-monitor")` instead of a second `__version__`).

### 7. Optional remote view

The app is a local OpenCV window. A thin FastAPI (or similar) MJPEG/WebRTC page would let a phone check the mosaic without a desktop session, while the kiosk display stays fullscreen.

### 8. Decode and detect on the GPU when present

FFmpeg/OpenCV software-decode of several 1080p RTSP streams plus YOLO will tax a small NUC. Optional NVIDIA NVDEC / VAAPI decode and ONNX Runtime / OpenVINO / TensorRT for the detector would make 6–9 cameras realistic on modest hardware.

### Later / nice to have

- Event-triggered or continuous recording (lightweight NVR), with disk caps
- macOS windowing pass (fonts and autostart are Windows/Linux today)
- Docker / kiosk image (X11 + unprivileged user) for a dedicated monitor PC
- Per-camera PTZ keys when the vendor API exists
- Audio from cameras, or a chime on detection
- Watchdog that hard-restarts a hung FFmpeg capture instead of only reconnecting
- Drop leftover `test.txt` from the repo root (install cheat-sheet, not product)

## License

MIT. See [LICENSE](LICENSE).
