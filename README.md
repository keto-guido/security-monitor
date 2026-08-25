# Security Monitor

Desktop mosaic viewer for IP cameras on **Windows** and **Linux**. It reads camera URLs and layout from a YAML file, captures each stream on its own thread, and composites a live grid with zoom, rewind, capture, detection, Home Assistant, and weather HUD.

**Version 0.2.0** · Python 3.10+ · OpenCV GUI (not headless)

Default layout is **2×2 (4 cameras)**. Streams can be `rtsp://` or `rtp://` (HTTP MJPEG, local files, and webcams work too).

## Built features

These are in `main` (merged via PRs #1–#4). Nothing below is “coming later.”

- Live mosaic from RTSP / RTP / HTTP / files / webcams; one capture thread per camera; auto-reconnect
- Focus a tile (`1`–`9`, click), `n` / `p` next/prev, scroll-wheel zoom toward the cursor, pan when zoomed
- **Cameras menu** — grid layout 1×1–4×4, show/hide, reorder, add RTSP/webcam/demo, remove, auto cycle-focus
- Smooth playback buffer and rolling rewind (`,` / `.` / `l`)
- Snapshots (`s`) and clips (`c`); **Saved captures** browser with preview, lock, delete, folder reveal
- Auto **person-event** snapshot + pre/during/post clip; lockable retention (age + max GB)
- Opt-in **YOLOv8n** people boxes and lighting-normalized “new object” (package) detection
- **Encroachment** — tripwire and polygon ROIs, alarm banner/sound, optional autofocus
- **Weather HUD** — Open-Meteo widget, placeable on the mosaic, per-line toggles, opacity
- **Home Assistant** — local door/sensor links (popup, HUD, highlight, autofocus, sound) and a right-side lights panel (`]`)
- GPU/CPU **decode** controls (`auto` / `cpu` / `gpu`, VAAPI/CUDA/QSV/D3D11/VideoToolbox) plus decode status
- Reboot Ubiquiti (SSH) or Reolink / Amcrest / Dahua (HTTP) from the menu or CLI
- Ubuntu login autostart (XDG `.desktop` or systemd user unit) and Windows Startup folder
- Linux fullscreen uses the real screen size from `xrandr` so OpenCV Qt does not squash the grid

## Install

Python 3.10+ and a desktop session (Windows, or Linux X11/Wayland) are required.

### Ubuntu / Debian (one line, install or update)

Ubuntu 23.04+ and Debian 12 block `pip install` into the system Python (`externally-managed-environment`). Install into a **venv** instead — same command both times:

```bash
sudo apt install -y python3-venv fonts-dejavu-core libgl1 libglib2.0-0 && python3 -m venv ~/.venvs/security-monitor && ~/.venvs/security-monitor/bin/pip install -U "git+https://github.com/keto-guido/security-monitor.git" && mkdir -p ~/.local/bin && ln -sfn ~/.venvs/security-monitor/bin/security-monitor ~/.local/bin/security-monitor
```

Then (new terminal if `~/.local/bin` is not on PATH yet):

```bash
security-monitor
```

Rerun that one line whenever you want the latest from GitHub `main`.

### Windows (one line, install or update)

```bash
python -m pip install -U "git+https://github.com/keto-guido/security-monitor.git"
```

### Editable / from a clone

```bash
git clone https://github.com/keto-guido/security-monitor.git
cd security-monitor
python3 -m venv .venv && source .venv/bin/activate   # Windows: py -m venv .venv && .venv\Scripts\activate
pip install -e ".[dev]"
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
  encroachment_detection: false  # tripwire / polygon ROI alerts (also enable per camera)
  encroachment_autofocus: false  # focus that camera while someone is in a zone
  encroachment_alarm: true       # pulsing banner + strong tile highlight
  encroachment_alarm_sound: true # beep on entry (and periodically while active)
  cycle_focus: false             # auto-advance focused camera
  cycle_focus_seconds: 10

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
    # detect_encroachment: false
    # encroach_zones:
    #   - name: Porch
    #     kind: polygon
    #     points: [[0.05, 0.55], [0.95, 0.55], [0.95, 0.98], [0.05, 0.98]]
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
| `n` / `p` | Next / previous camera (focus) |
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
| Click a tile | Focus camera (click again for grid; click while zoomed resets zoom) |

The options menu also has **Cameras**, **Capture** / **Saved captures**, **Detection**, **Weather HUD**, **Home Assistant**, **Video settings**, and **Reboot cameras**. Changes are saved back into `config.yaml`. Files go to `~/security-monitor/captures` by default (override with `display.save_directory`). Baselines are stored under `~/.config/security-monitor/baselines/` (or `%APPDATA%\security-monitor\baselines` on Windows). Reboot uses each camera's `type`, host, and credentials from `config.yaml` (Ubiquiti over SSH, Reolink/Amcrest/Dahua over HTTP). You will be asked to confirm. Progress shows in the window; when it finishes, streams reconnect.

### Weather HUD

`Esc` → **Weather HUD…**:

- Toggle the on-mosaic weather widget (Open-Meteo; no API key)
- **Placement** — bottom left/right, between columns, between rows, or custom
- **Place widget on layout…** — still frame of the mosaic; drag the widget (←→↑↓ nudge, Enter save, Esc cancel)
- Fine X/Y and width/height for precise positioning
- Toggle each line independently: temperature, conditions, storm warnings, lightning tracker
- **Opacity** — blend the weather panel (← →); lower values let the camera show through
- **Overlay cameras** — paint on top of full tiles instead of shrinking them around the widget
- °F / °C units; optional `weather_latitude` / `weather_longitude` in config (blank = IP auto-locate)

By default, camera tiles **shrink around** the widget so they never draw under it. When placed between two feeds, both cameras lose a shared strip. Turn on **Overlay cameras** (and lower **Opacity**) if you prefer a translucent HUD on top of the feed instead. Camera name/status strip opacity is under **Video settings** → **HUD opacity**.

### Home Assistant (door sensors)

`Esc` → **Home Assistant…** (local LAN only):

1. Set HA base URL + a [long-lived access token](https://www.home-assistant.io/docs/authentication/)
2. **Browse HA entities…** — pick a domain, then an entity from the live list
3. Toggle which **state(s)** should alert (e.g. `on` / `open` / `unlocked`)
4. Pick a **camera**, or **No camera** for HUD/popup-only alerts
5. Choose notification types per link:
   - **Popup toast** — temporary on-screen notice (not tied to a camera)
   - **Persistent HUD strip** — stays while the sensor is active
   - **Highlight / Autofocus** — requires a linked camera
   - **Alarm sound**
6. **Save link**

**Light panel buttons…** — pin `light.*` / `switch.*` entities. Press **`]`** (or click the right-edge **HA** tab) to slide out the panel and tap buttons to toggle lights. Esc closes the panel.

Example:

```yaml
display:
  ha_enabled: true
  ha_url: http://homeassistant.local:8123
  ha_token: YOUR_LONG_LIVED_ACCESS_TOKEN
  ha_popup_seconds: 5
  ha_panel_enabled: true
  ha_doors:
    - entity_id: binary_sensor.front_door
      label: Front door
      camera: Front Door          # omit for popup/HUD only
      open_states: [on]
      notify_popup: true
      notify_hud: true
      notify_highlight: true
      notify_autofocus: true
      notify_sound: true
  ha_lights:
    - entity_id: light.kitchen
      label: Kitchen
```

### Video decode (GPU / CPU)

`Esc` → **Video settings**:

- **Decode** — `auto` (prefer GPU, fall back to CPU), `cpu`, or `gpu`
- **HW backend** — `auto`, `none`, or a specific FFmpeg accel (`cuda`, `qsv`, `vaapi`, `d3d11va`, `videotoolbox`)
- **HUD opacity** — transparency of the camera name/status strip on each tile
- **Decode status…** — OpenCV/FFmpeg capability summary plus the path each camera actually opened with (e.g. `auto/vaapi`, `cpu (fallback)`)

Changing decode mode or backend reconnects streams. Optional `display.hwaccel_device` (e.g. `/dev/dri/renderD128`) is for VAAPI. Stock `opencv-python` wheels often still decode on the CPU even when hardware acceleration is requested — use `security-monitor check` to see what this build reports.

### Person events (auto capture)

`Esc` → **Detection** → **Auto person capture** (also **Person events…** on the root menu):

- On rising edge (person appears), saves a **snapshot** with person boxes + date/time overlay
- Records a clip with configurable **pre-roll** (from the rolling buffer), the time the person is present, and **post-roll** after they leave
- Requires people detection master + per-camera include (`Cameras included…`)
- Browse events as stamped snapshots; Enter opens the recording playback (Esc stops)
- **Lock (keep forever)** so auto-erase / max-storage cleanup never removes that event

Files land under `~/security-monitor/captures/events/<timestamp>_<camera>/` (`snapshot.jpg` + `clip.mp4`).

### Saved captures

`Esc` → **Saved captures…** (or **Capture** → **Browse saved captures…**):

- List snapshots and clips newest-first
- Open a file to preview (images + first frame of videos)
- ← → browse neighboring files
- **Lock (keep forever)** / **Unlock** — locked files are never auto-erased
- **Show in folder** opens the OS file manager
- **Delete** / **Delete all** with confirmation (delete-all skips locked files)

**Storage cleanup** (`Esc` → **Capture**):

- **Auto-erase after** — Off / 1d … 90d (unlocked only)
- **Max storage** — Off / 1–100 GB; when over the cap, oldest unlocked items are removed first
- **Erase old unlocked now…** — run a sweep immediately
- Defaults: 14 days and 20 GB; person events and manual captures share the same rules

### Cameras menu

`Esc` → **Cameras…** updates `config.yaml` without hand-editing:

- **Layout** — cycle grid size (1×1 … 4×4); empty slots show placeholders
- **Cycle focus** — Off / 5s / 10s / 30s / 60s auto-advance (`n` / `p` also step manually)
- **Arrange tiles** — ← → moves a camera earlier/later in the mosaic order
- **Show / hide** — toggle `enabled` (hidden cameras stay in the file)
- **Add camera** — RTSP/URL (on-screen text entry), webcam device 0/1, or a demo tile
- **Remove camera** — delete from the running mosaic and from `config.yaml`

### Encroachment (tripwire + polygon ROIs)

`Esc` → **Detection**:

- **Encroachment** — master switch (also turns on people detection)
- **Autofocus on encroach** — while a person is in any zone, focus that camera; return to the grid when they leave
- **On-screen alarm** — pulsing red mosaic banner + stronger tile border while active
- **Alarm sound** — double beep on entry, then a reminder beep every few seconds while someone remains in a zone
- **Cameras included…** — per-camera **encroach** opt-in
- **Add tripwire preset** / **Tripwire zone side** / **Draw tripwire** — directed half-plane zones
- **Add polygon preset** / **Draw polygon ROI** — click corners, **Enter** to finish (≥3 points); Esc cancels
- **Clear all zones** — remove every ROI on the focused camera

A camera can have **multiple zones** (mix of lines and polygons). Any person whose feet land inside a zone triggers highlight + alarm. Zones are stored under `encroach_zones` in `config.yaml` (legacy `encroach_line` still works as a fallback).

### Detection notes

**Stack:** Ultralytics **YOLOv8n** (COCO) is the primary people detector — real-time neural detection. If YOLO can’t load, the app falls back to MobileNet-SSD, then OpenCV HOG. New packages use a **lighting-normalized baseline diff** (CLAHE + adaptive threshold + persistence). YOLO “thing” classes (backpack, suitcase, handbag, …) that overlap a change blob confirm arrivals faster. People are masked out of the change map so walkers don’t look like packages.

**Seasonal drift:** You seed an empty-area baseline once. After that the baseline **slowly adapts** (hours-scale EMA) to leaves, snow texture, and lawn growth, and is saved back to disk periodically. Pixels under a confirmed package (and under people) are **frozen**, so a box left for days keeps its `new object` flag instead of being absorbed. A whole-scene shift with no packages (e.g. overnight snow) speeds up adaptation after it settles. Manual **Set empty-area baseline** still resets everything when you want a clean slate.

- **People**, **new object**, and **encroachment** detection are off until you opt in globally **and** per camera (`Esc` → Detection).
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

Each camera is read on its own thread and only the latest frame is kept (plus optional history), so a slow stream does not stall the others. The UI thread composites a grid, letterboxes (or crops) each tile, and draws name / status / FPS. Dropped streams retry on `reconnect_seconds`.

OpenCV’s FFmpeg backend is used for RTSP/RTP. The GUI package must be `opencv-python`, not `opencv-python-headless`. Decode mode (`display.decode_mode` / Video settings) sets `OPENCV_FFMPEG_CAPTURE_OPTIONS` (`hwaccel=…`) when opening RTSP streams and falls back to software decode if the GPU path fails a smoke read.

| Path | Role |
| --- | --- |
| `cli.py` | argparse entry (`run`, `demo`, `check`, `init`, `reboot`, `autostart`, `display`) |
| `config.py` | YAML load / validate / persist menu settings |
| `stream.py` | Per-camera capture threads |
| `buffer.py` | Smooth-delay and rewind ring |
| `mosaic.py` | OpenCV window, HUD, menus, zoom, input |
| `detection.py` | People + left-behind object pipeline |
| `encroachment.py` | Tripwire / polygon ROI alarms |
| `events.py` | Auto person-event snapshot + clip |
| `capture.py` / `retention.py` | Manual snapshots/clips and auto-erase |
| `weather.py` | Open-Meteo HUD |
| `home_assistant.py` | Local HA sensors and light panel |
| `decode.py` | GPU/CPU FFmpeg hwaccel |
| `reboot.py` / `autostart.py` / `display_setup.py` | Camera reboot, login start, Linux screen size |

## Troubleshooting

- **NO SIGNAL / connect failed** — verify the URL in VLC first, then try `transport: tcp`.
- **`externally-managed-environment`** — do not use `python3 -m pip install` on Ubuntu/Debian. Use the venv one-liner in [Install](#ubuntu--debian-one-line-install-or-update).
- **Window never appears** — `pip uninstall opencv-python-headless` then `pip install opencv-python` (inside the venv, not system pip).
- **High latency** — `tcp` is stable but buffered; `udp` is snappier. Cell size also drives decode cost. Try `decode_mode: cpu` if a bad GPU path stalls opens.
- **Want GPU decode** — set `decode_mode: gpu` (or `auto`) and pick a backend under Video settings. Confirm with **Decode status…** or `security-monitor check`. Pip wheels frequently lack working CUDA/VAAPI decode; a custom OpenCV/FFmpeg build may be required.
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

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Remaining upgrades

These are **not** in the tree yet. Cameras menu, weather, HA, encroachment, GPU decode, person events, and retention **are already merged**.

- Move YOLO off the UI thread so inference cannot hitch the mosaic; add interval / model-size knobs
- Optional extra: `pip install "security-monitor[detect]"` so a viewer-only install does not pull PyTorch
- Secrets via OS keyring or `password_env` instead of plaintext YAML; persist with ruamel.yaml so comments survive menu saves
- GitHub Actions (`pytest` on Ubuntu + Windows), Ruff, and a single version source
- Optional remote MJPEG/WebRTC page for phone checks
- Split `mosaic.py` (it still owns windowing, every menu, and input)
- macOS autostart; Docker/kiosk image; vendor PTZ keys

## License

MIT. See [LICENSE](LICENSE).
