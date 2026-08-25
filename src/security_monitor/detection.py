"""People and new-object detection (Windows + Linux).

People: Ultralytics YOLOv8n (modern real-time detector). Falls back to
OpenCV MobileNet-SSD, then HOG, if YOLO cannot load.

New objects: lighting-normalized baseline change detection against an
empty-area baseline, with multi-frame persistence tracking. The baseline
slowly adapts (EMA) to seasonal / lighting drift, but pixels under
confirmed packages (and people) are frozen so a left-behind box stays
flagged for days. Walking people are masked out. Optional YOLO "thing"
hits that overlap a change blob reinforce confirmation (packages are
often not a COCO class, so baseline change remains the source of truth).
"""

from __future__ import annotations

import math
import os
import re
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# COCO / YOLOv8 class id for person.
_YOLO_PERSON = 0
# COCO classes that often appear as left-behind porch/yard items.
_YOLO_THING_CLASSES = frozenset(
    {
        24,  # backpack
        26,  # handbag
        28,  # suitcase
        39,  # bottle
        41,  # cup
        56,  # chair
        63,  # laptop
        64,  # mouse
        65,  # remote
        66,  # keyboard
        67,  # cell phone
        73,  # book
        74,  # clock
        75,  # vase
        77,  # teddy bear
    }
)

# Legacy OpenCV DNN fallback (VOC MobileNet-SSD).
_PERSON_CLASS_ID = 15
_MOBILENET_PROTOTXT = "MobileNetSSD_deploy.prototxt"
_MOBILENET_MODEL = "MobileNetSSD_deploy.caffemodel"
_MOBILENET_PROTOTXT_URL = (
    "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/"
    "MobileNetSSD_deploy.prototxt"
)
_MOBILENET_MODEL_URL = (
    "https://github.com/chuanqi305/MobileNet-SSD/raw/master/"
    "MobileNetSSD_deploy.caffemodel"
)

_YOLO_WEIGHTS = "yolov8n.pt"

# Adaptive baseline: ~6h time-constant for leaves / gradual season drift.
_DEFAULT_ADAPT_TAU_S = 6 * 3600
# Faster absorb when the whole scene shifts (snow dump) with no packages.
_ENV_ADAPT_TAU_S = 20 * 60
_ENV_CHANGE_RATIO = 0.35
_ENV_SETTLE_S = 15 * 60
# Non-YOLO "new object" blobs (leaf piles) may merge into baseline after this.
_ABSORB_NON_REINFORCED_S = 48 * 3600
_PERSIST_EVERY_S = 15 * 60
_TRACK_GRACE_S = 1.25
_HEAL_SECONDS = 45.0
_HEAL_ADAPT_TAU_S = 8.0


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    conf: float = 1.0

    def clip(self, width: int, height: int) -> Box:
        return Box(
            x1=max(0, min(self.x1, width - 1)),
            y1=max(0, min(self.y1, height - 1)),
            x2=max(0, min(self.x2, width - 1)),
            y2=max(0, min(self.y2, height - 1)),
            label=self.label,
            conf=self.conf,
        )

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def iou(self, other: Box) -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass
class _TrackedObject:
    box: Box
    first_seen: float
    last_seen: float
    confirmed: bool = False
    reinforced: bool = False  # overlapped a YOLO "thing" at least once
    first_confirmed: float | None = None


@dataclass
class _HealRegion:
    """After a package leaves, heal the frozen baseline hole quickly."""

    box: Box
    until: float


def models_dir() -> Path:
    path = Path.home() / ".cache" / "security-monitor" / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def baselines_dir() -> Path:
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            path = Path(appdata) / "security-monitor" / "baselines"
        else:
            path = Path.home() / "security-monitor" / "baselines"
    else:
        path = Path.home() / ".config" / "security-monitor" / "baselines"
    path.mkdir(parents=True, exist_ok=True)
    return path


def baseline_path(camera_name: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", camera_name.strip()).strip("-._") or "camera"
    return baselines_dir() / f"{slug}.jpg"


def _download(url: str, dest: Path, timeout: float = 120.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as resp, tmp.open("wb") as out:
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dest)


def _lighting_normalized_gray(frame: np.ndarray) -> np.ndarray:
    """Grayscale that is more stable under global lighting / shadow shifts."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    luminance = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(luminance)


class PersonDetector:
    """
    People detector cascade:

    1. YOLOv8n via ultralytics (preferred)
    2. OpenCV MobileNet-SSD
    3. OpenCV HOG
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._yolo = None
        self._net = None
        self._hog = None
        self._backend = "none"
        self._init_error = ""

    @property
    def backend(self) -> str:
        self._ensure_backend()
        return self._backend

    @property
    def ready(self) -> bool:
        return self.backend in {"yolov8n", "mobilenet-ssd", "hog"}

    def _ensure_backend(self) -> None:
        with self._lock:
            if self._backend != "none":
                return
            # 1) YOLOv8n
            try:
                from ultralytics import YOLO

                weights = models_dir() / _YOLO_WEIGHTS
                # Ultralytics downloads weights on first use if path missing;
                # pin cache dir by chdir-less absolute path when present.
                model_path = str(weights) if weights.is_file() else _YOLO_WEIGHTS
                model = YOLO(model_path)
                # Warm-up + ensure weights land in our cache when downloaded.
                if not weights.is_file():
                    try:
                        # Copy/link default download into our cache if available.
                        default = Path.home() / ".cache" / "ultralytics" / _YOLO_WEIGHTS
                        # Ultralytics may store under cwd; probe common spots later.
                        _ = default
                    except OSError:
                        pass
                self._yolo = model
                self._backend = "yolov8n"
                return
            except Exception as exc:  # noqa: BLE001
                self._init_error = f"yolo={exc}"

            # 2) MobileNet-SSD
            try:
                proto = models_dir() / _MOBILENET_PROTOTXT
                caffe = models_dir() / _MOBILENET_MODEL
                if not proto.is_file():
                    _download(_MOBILENET_PROTOTXT_URL, proto)
                if not caffe.is_file():
                    _download(_MOBILENET_MODEL_URL, caffe)
                self._net = cv2.dnn.readNetFromCaffe(str(proto), str(caffe))
                self._backend = "mobilenet-ssd"
                return
            except Exception as exc:  # noqa: BLE001
                self._init_error = f"{self._init_error}; mobilenet={exc}"

            # 3) HOG
            try:
                hog = cv2.HOGDescriptor()
                hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                self._hog = hog
                self._backend = "hog"
            except Exception as exc:  # noqa: BLE001
                self._init_error = f"{self._init_error}; hog={exc}"
                self._backend = "unavailable"

    def detect(
        self,
        frame: np.ndarray,
        *,
        conf_thresh: float = 0.35,
        include_things: bool = False,
    ) -> tuple[list[Box], list[Box]]:
        """
        Returns (people_boxes, thing_boxes).

        ``thing_boxes`` are only populated when the YOLO backend is active and
        ``include_things`` is True (used to reinforce new-object blobs).
        """
        self._ensure_backend()
        if frame is None or frame.size == 0:
            return [], []
        h, w = frame.shape[:2]
        if self._backend == "yolov8n" and self._yolo is not None:
            return self._detect_yolo(frame, w, h, conf_thresh, include_things)
        if self._backend == "mobilenet-ssd" and self._net is not None:
            return self._detect_mobilenet(frame, w, h, conf_thresh), []
        if self._backend == "hog" and self._hog is not None:
            return self._detect_hog(frame, w, h), []
        return [], []

    def _detect_yolo(
        self,
        frame: np.ndarray,
        w: int,
        h: int,
        conf_thresh: float,
        include_things: bool,
    ) -> tuple[list[Box], list[Box]]:
        assert self._yolo is not None
        classes = [0]
        if include_things:
            classes = sorted({0, *_YOLO_THING_CLASSES})
        results = self._yolo.predict(
            source=frame,
            conf=conf_thresh,
            classes=classes,
            verbose=False,
            imgsz=640,
        )
        people: list[Box] = []
        things: list[Box] = []
        if not results:
            return people, things
        result = results[0]
        if result.boxes is None:
            return people, things
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
            box = Box(int(x1), int(y1), int(x2), int(y2), "person", float(conf)).clip(w, h)
            if int(cls_id) == _YOLO_PERSON:
                box = Box(box.x1, box.y1, box.x2, box.y2, "person", float(conf))
                if box.area > 400:
                    people.append(box)
            elif include_things and int(cls_id) in _YOLO_THING_CLASSES:
                things.append(
                    Box(box.x1, box.y1, box.x2, box.y2, "thing", float(conf)).clip(w, h)
                )
        return people, things

    def _detect_mobilenet(
        self, frame: np.ndarray, w: int, h: int, conf_thresh: float
    ) -> list[Box]:
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            0.007843,
            (300, 300),
            127.5,
        )
        assert self._net is not None
        self._net.setInput(blob)
        detections = self._net.forward()
        boxes: list[Box] = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            class_id = int(detections[0, 0, i, 1])
            if class_id != _PERSON_CLASS_ID or conf < conf_thresh:
                continue
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            box = Box(x1, y1, x2, y2, "person", conf).clip(w, h)
            if box.area > 400:
                boxes.append(box)
        return boxes

    def _detect_hog(self, frame: np.ndarray, w: int, h: int) -> list[Box]:
        scale = 1.0
        work = frame
        if max(h, w) > 640:
            scale = 640 / max(h, w)
            work = cv2.resize(frame, (int(w * scale), int(h * scale)))
        assert self._hog is not None
        rects, weights = self._hog.detectMultiScale(
            work,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
        boxes: list[Box] = []
        for (x, y, bw, bh), weight in zip(rects, weights):
            if float(weight) < 0.4:
                continue
            x1 = int(x / scale)
            y1 = int(y / scale)
            x2 = int((x + bw) / scale)
            y2 = int((y + bh) / scale)
            boxes.append(Box(x1, y1, x2, y2, "person", float(weight)).clip(w, h))
        return boxes


def _blends_with_surroundings(frame: np.ndarray, box: Box, *, pad: int = 18) -> bool:
    """True when the box interior looks like the ring around it (object gone)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return True
    inner = frame[y1:y2, x1:x2]
    ox1, oy1 = max(0, x1 - pad), max(0, y1 - pad)
    ox2, oy2 = min(w, x2 + pad), min(h, y2 + pad)
    ring = frame[oy1:oy2, ox1:ox2].astype(np.int16)
    iy1, ix1 = y1 - oy1, x1 - ox1
    iy2, ix2 = iy1 + (y2 - y1), ix1 + (x2 - x1)
    ring[iy1:iy2, ix1:ix2] = -1
    valid = ring.reshape(-1, 3)
    valid = valid[np.all(valid >= 0, axis=1)]
    if valid.size == 0:
        return True
    inner_mean = inner.reshape(-1, 3).mean(axis=0).astype(np.float64)
    ring_mean = valid.mean(axis=0).astype(np.float64)
    return float(np.linalg.norm(inner_mean - ring_mean)) < 22.0

class NewObjectTracker:
    """Find left-behind objects vs an adaptive empty-area baseline.

    The user seeds the baseline once (empty scene). After that the baseline
    slowly EMA-adapts to seasonal drift (leaves, snow texture, lawn growth).
    Confirmed package regions are frozen so a box left for days stays flagged.
    YOLO-reinforced packages never merge into the baseline; non-reinforced
    blobs (often leaf piles) may absorb after a long idle period.
    """

    def __init__(
        self,
        *,
        adapt_tau_seconds: float = _DEFAULT_ADAPT_TAU_S,
        absorb_non_reinforced_seconds: float = _ABSORB_NON_REINFORCED_S,
        persist_every_seconds: float = _PERSIST_EVERY_S,
    ) -> None:
        self._baselines: dict[str, np.ndarray] = {}
        self._baseline_f: dict[str, np.ndarray] = {}
        self._tracks: dict[str, list[_TrackedObject]] = {}
        self._heal: dict[str, list[_HealRegion]] = {}
        self._last_adapt_at: dict[str, float] = {}
        self._last_persist_at: dict[str, float] = {}
        self._env_change_since: dict[str, float | None] = {}
        self._lock = threading.Lock()
        self._adapt_tau = float(adapt_tau_seconds)
        self._absorb_after = float(absorb_non_reinforced_seconds)
        self._persist_every = float(persist_every_seconds)

    def load_baseline(self, camera_name: str, path: Path | None = None) -> bool:
        path = path or baseline_path(camera_name)
        if not path.is_file():
            return False
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            return False
        with self._lock:
            self._store_baseline_locked(camera_name, image, reset_tracks=True)
        return True

    def has_baseline(self, camera_name: str) -> bool:
        with self._lock:
            if camera_name in self._baselines:
                return True
        return baseline_path(camera_name).is_file()

    def set_baseline(self, camera_name: str, frame: np.ndarray) -> Path:
        if frame is None or frame.size == 0:
            raise ValueError("No frame available for baseline")
        path = baseline_path(camera_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(path), frame)
        if not ok:
            raise OSError(f"Could not write baseline: {path}")
        with self._lock:
            self._store_baseline_locked(camera_name, frame.copy(), reset_tracks=True)
            self._last_persist_at[camera_name] = time.monotonic()
        return path

    def _store_baseline_locked(
        self, camera_name: str, image: np.ndarray, *, reset_tracks: bool
    ) -> None:
        self._baselines[camera_name] = image
        self._baseline_f[camera_name] = image.astype(np.float32)
        self._last_adapt_at[camera_name] = time.monotonic()
        self._env_change_since[camera_name] = None
        if reset_tracks:
            self._tracks[camera_name] = []
            self._heal[camera_name] = []

    def detect(
        self,
        camera_name: str,
        frame: np.ndarray,
        *,
        person_boxes: list[Box] | None = None,
        thing_boxes: list[Box] | None = None,
        min_area_ratio: float = 0.0035,
        confirm_seconds: float = 2.0,
        adapt_tau_seconds: float | None = None,
    ) -> list[Box]:
        if frame is None or frame.size == 0:
            return []
        with self._lock:
            baseline = self._baselines.get(camera_name)
        if baseline is None:
            if not self.load_baseline(camera_name):
                return []
            with self._lock:
                baseline = self._baselines.get(camera_name)
        if baseline is None:
            return []

        h, w = frame.shape[:2]
        if baseline.shape[0] != h or baseline.shape[1] != w:
            baseline = cv2.resize(baseline, (w, h), interpolation=cv2.INTER_AREA)
            with self._lock:
                self._baselines[camera_name] = baseline
                self._baseline_f[camera_name] = baseline.astype(np.float32)

        gray = _lighting_normalized_gray(frame)
        base = _lighting_normalized_gray(baseline)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        base = cv2.GaussianBlur(base, (5, 5), 0)
        diff = cv2.absdiff(base, gray)

        # Adaptive threshold reduces false triggers from gradual lighting drift.
        thresh = max(18, int(np.median(diff) + 2.5 * (np.std(diff) + 1e-3)))
        thresh = min(thresh, 48)
        _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)

        change_ratio = float(np.count_nonzero(mask)) / float(max(1, mask.size))

        if person_boxes:
            for box in person_boxes:
                pad = 16
                x1, y1 = max(0, box.x1 - pad), max(0, box.y1 - pad)
                x2, y2 = min(w, box.x2 + pad), min(h, box.y2 + pad)
                mask[y1:y2, x1:x2] = 0

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = max(500, int(w * h * min_area_ratio))
        candidates: list[Box] = []
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            area = bw * bh
            if area < min_area or area > 0.4 * w * h:
                continue
            aspect = bw / max(bh, 1)
            if aspect > 6.0 or aspect < 0.12:
                continue
            candidates.append(Box(x, y, x + bw, y + bh, "new object", 1.0).clip(w, h))

        now = time.monotonic()
        with self._lock:
            heals = [
                h for h in self._heal.get(camera_name, []) if h.until > now
            ]
            self._heal[camera_name] = heals
            if heals:
                filtered: list[Box] = []
                for cand in candidates:
                    if any(cand.iou(h.box) >= 0.2 for h in heals):
                        continue
                    filtered.append(cand)
                candidates = filtered

            tracks = self._tracks.setdefault(camera_name, [])
            updated: list[_TrackedObject] = []
            for cand in candidates:
                matched = None
                best_iou = 0.25
                for track in tracks:
                    score = cand.iou(track.box)
                    if score > best_iou:
                        best_iou = score
                        matched = track
                reinforced = False
                if thing_boxes:
                    for thing in thing_boxes:
                        if cand.iou(thing) >= 0.15:
                            reinforced = True
                            break
                if matched is not None:
                    matched.box = cand
                    matched.last_seen = now
                    matched.reinforced = matched.reinforced or reinforced
                    needed = 1.0 if matched.reinforced else confirm_seconds
                    if not matched.confirmed and now - matched.first_seen >= needed:
                        matched.confirmed = True
                        matched.first_confirmed = now
                    # If a confirmed "object" now blends with the porch, it was
                    # removed — the leftover change is a baseline hole to heal.
                    if matched.confirmed and _blends_with_surroundings(frame, cand):
                        self._heal.setdefault(camera_name, []).append(
                            _HealRegion(box=cand, until=now + _HEAL_SECONDS)
                        )
                    else:
                        updated.append(matched)
                    tracks.remove(matched)
                else:
                    updated.append(
                        _TrackedObject(
                            box=cand,
                            first_seen=now,
                            last_seen=now,
                            confirmed=False,
                            reinforced=reinforced,
                        )
                    )
            for track in tracks:
                if track.confirmed and now - track.last_seen < _TRACK_GRACE_S:
                    updated.append(track)
                elif track.confirmed:
                    # Package left — heal the frozen hole so it does not
                    # immediately re-trigger as a new object.
                    self._heal.setdefault(camera_name, []).append(
                        _HealRegion(box=track.box, until=now + _HEAL_SECONDS)
                    )
            self._tracks[camera_name] = updated
            confirmed = [t.box for t in updated if t.confirmed]
            self._adapt_baseline_locked(
                camera_name,
                frame,
                updated,
                person_boxes or [],
                change_ratio=change_ratio,
                now=now,
                adapt_tau_seconds=adapt_tau_seconds,
            )
            return confirmed

    def _freeze_mask(
        self,
        h: int,
        w: int,
        tracks: list[_TrackedObject],
        person_boxes: list[Box],
        now: float,
    ) -> np.ndarray:
        """True where the baseline must not adapt (packages / people)."""
        freeze = np.zeros((h, w), dtype=bool)
        for box in person_boxes:
            pad = 20
            x1, y1 = max(0, box.x1 - pad), max(0, box.y1 - pad)
            x2, y2 = min(w, box.x2 + pad), min(h, box.y2 + pad)
            freeze[y1:y2, x1:x2] = True
        for track in tracks:
            # Keep unconfirmed candidates frozen during the confirm window.
            if not track.confirmed:
                box = track.box
                pad = 12
                x1, y1 = max(0, box.x1 - pad), max(0, box.y1 - pad)
                x2, y2 = min(w, box.x2 + pad), min(h, box.y2 + pad)
                freeze[y1:y2, x1:x2] = True
                continue
            # YOLO-reinforced packages stay frozen forever.
            # Non-reinforced blobs may eventually absorb (leaf piles, etc.).
            age = 0.0
            if track.first_confirmed is not None:
                age = now - track.first_confirmed
            if track.reinforced or age < self._absorb_after:
                box = track.box
                pad = 12
                x1, y1 = max(0, box.x1 - pad), max(0, box.y1 - pad)
                x2, y2 = min(w, box.x2 + pad), min(h, box.y2 + pad)
                freeze[y1:y2, x1:x2] = True
        return freeze

    def _adapt_baseline_locked(
        self,
        camera_name: str,
        frame: np.ndarray,
        tracks: list[_TrackedObject],
        person_boxes: list[Box],
        *,
        change_ratio: float,
        now: float,
        adapt_tau_seconds: float | None,
    ) -> None:
        h, w = frame.shape[:2]
        base_f = self._baseline_f.get(camera_name)
        if base_f is None:
            return
        if base_f.shape[0] != h or base_f.shape[1] != w:
            base_f = cv2.resize(base_f, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
            self._baseline_f[camera_name] = base_f

        confirmed_any = any(t.confirmed for t in tracks)
        people_present = bool(person_boxes)

        # Whole-scene environmental shift (snow dump): speed up adapt when
        # nothing is flagged as a package and nobody is walking through.
        env_since = self._env_change_since.get(camera_name)
        if (
            change_ratio >= _ENV_CHANGE_RATIO
            and not confirmed_any
            and not people_present
        ):
            if env_since is None:
                self._env_change_since[camera_name] = now
                env_since = now
        else:
            self._env_change_since[camera_name] = None
            env_since = None

        tau = float(adapt_tau_seconds) if adapt_tau_seconds is not None else self._adapt_tau
        if (
            env_since is not None
            and now - env_since >= _ENV_SETTLE_S
            and not confirmed_any
        ):
            tau = min(tau, float(_ENV_ADAPT_TAU_S))

        last = self._last_adapt_at.get(camera_name, now)
        dt = max(0.0, now - last)
        self._last_adapt_at[camera_name] = now
        if dt <= 0.0 or tau <= 0.0:
            return
        alpha = 1.0 - math.exp(-dt / tau)
        # Cap a single step so a hitch doesn't yank the baseline.
        alpha = min(alpha, 0.08)
        if alpha < 1e-6:
            return

        freeze = self._freeze_mask(h, w, tracks, person_boxes, now)
        frame_f = frame.astype(np.float32)
        if not np.any(freeze):
            base_f = (1.0 - alpha) * base_f + alpha * frame_f
        else:
            update3 = (~freeze)[:, :, None]
            mixed = (1.0 - alpha) * base_f + alpha * frame_f
            base_f = np.where(update3, mixed, base_f)

        # Fast-heal regions where a package recently left (frozen hole).
        heals = [h for h in self._heal.get(camera_name, []) if h.until > now]
        if heals:
            heal_alpha = 1.0 - math.exp(-dt / _HEAL_ADAPT_TAU_S)
            heal_alpha = min(heal_alpha, 0.25)
            if heal_alpha >= 1e-6:
                heal_mask = np.zeros((h, w), dtype=bool)
                for region in heals:
                    box = region.box
                    pad = 8
                    x1, y1 = max(0, box.x1 - pad), max(0, box.y1 - pad)
                    x2, y2 = min(w, box.x2 + pad), min(h, box.y2 + pad)
                    heal_mask[y1:y2, x1:x2] = True
                heal3 = heal_mask[:, :, None]
                healed = (1.0 - heal_alpha) * base_f + heal_alpha * frame_f
                base_f = np.where(heal3, healed, base_f)

        np.clip(base_f, 0, 255, out=base_f)
        self._baseline_f[camera_name] = base_f
        self._baselines[camera_name] = np.round(base_f).astype(np.uint8)

        last_persist = self._last_persist_at.get(camera_name, 0.0)
        if now - last_persist >= self._persist_every:
            self._persist_baseline_locked(camera_name)

    def _persist_baseline_locked(self, camera_name: str) -> None:
        image = self._baselines.get(camera_name)
        if image is None:
            return
        path = baseline_path(camera_name)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".part")
            if cv2.imwrite(str(tmp), image):
                tmp.replace(path)
                self._last_persist_at[camera_name] = time.monotonic()
        except OSError:
            pass


@dataclass
class DetectionState:
    boxes: list[Box] = field(default_factory=list)
    updated_at: float = 0.0


class DetectionEngine:
    """Runs people + new-object detection with light caching for the UI thread."""

    def __init__(self) -> None:
        self.people = PersonDetector()
        self.objects = NewObjectTracker()
        self._cache: dict[str, DetectionState] = {}
        self._interval = 0.30
        self._lock = threading.Lock()

    def ensure_ready(self) -> str:
        """Force backend init; return backend name or error."""
        return self.people.backend

    def process(
        self,
        camera_name: str,
        frame: np.ndarray,
        *,
        detect_people: bool,
        detect_objects: bool,
    ) -> list[Box]:
        if frame is None or frame.size == 0:
            return []
        if not detect_people and not detect_objects:
            return []
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(camera_name)
            if cached is not None and now - cached.updated_at < self._interval:
                return list(cached.boxes)

        h, w = frame.shape[:2]
        scale = 1.0
        work = frame
        if max(h, w) > 960:
            scale = 960 / max(h, w)
            work = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        def _unmap(b: Box) -> Box:
            return Box(
                int(b.x1 / scale),
                int(b.y1 / scale),
                int(b.x2 / scale),
                int(b.y2 / scale),
                b.label,
                b.conf,
            ).clip(w, h)

        # One detector pass: people for display/masking, optional "thing" cues
        # to reinforce package blobs. People are still masked out of the
        # change map when only object detection is enabled.
        people_boxes: list[Box] = []
        thing_boxes: list[Box] = []
        if detect_people or detect_objects:
            people_boxes, thing_boxes = self.people.detect(
                work,
                include_things=detect_objects,
            )
            if scale != 1.0:
                people_boxes = [_unmap(b) for b in people_boxes]
                thing_boxes = [_unmap(b) for b in thing_boxes]

        object_boxes: list[Box] = []
        if detect_objects:
            object_boxes = self.objects.detect(
                camera_name,
                frame,
                person_boxes=people_boxes,
                thing_boxes=thing_boxes,
            )

        boxes = [*(people_boxes if detect_people else []), *object_boxes]
        with self._lock:
            self._cache[camera_name] = DetectionState(boxes=boxes, updated_at=now)
        return boxes


def draw_boxes(frame: np.ndarray, boxes: list[Box]) -> np.ndarray:
    """Draw bounding boxes + labels onto a BGR frame (in place)."""
    if frame is None or not boxes:
        return frame
    colors = {
        "person": (70, 190, 90),
        "new object": (40, 180, 255),
    }
    for box in boxes:
        color = colors.get(box.label, (220, 220, 220))
        cv2.rectangle(frame, (box.x1, box.y1), (box.x2, box.y2), color, 2)
        label = box.label
        if box.label == "person" and box.conf < 0.99:
            label = f"person {box.conf:.2f}"
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(0, box.y1 - 4)
        cv2.rectangle(
            frame,
            (box.x1, max(0, ty - text_size[1] - 4)),
            (box.x1 + text_size[0] + 6, ty + 2),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (box.x1 + 3, ty - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return frame
