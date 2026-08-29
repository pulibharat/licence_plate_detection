import os
import threading
import time
from datetime import datetime, timedelta

# RTSP over UDP loses/reorders packets on plenty of real networks
# (seen firsthand against a real NVR: frequent "bad cseq" RTP errors
# that silently killed individual frame reads). Forcing TCP transport
# trades a little latency for packets that actually arrive in order.
# Must be set before any cv2.VideoCapture(...) call is made.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2
import torch
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from types import SimpleNamespace

import config
import dashboard
import db
from plate_ocr import create_reader, read_plate_text
from plate_tracker import PlateTracker, PlateUpdate
from preprocessing import pad_box

# Config for this camera's own standalone vehicle-level BYTETracker
# instance (see CameraWorker._track_vehicles) - same defaults as
# ultralytics' bundled bytetrack.yaml, used directly since this
# tracker is constructed in code, not loaded from that file.
_VEHICLE_TRACKER_ARGS = SimpleNamespace(
    track_high_thresh=0.25,
    track_low_thresh=0.1,
    new_track_thresh=0.25,
    track_buffer=30,
    match_thresh=0.8,
    fuse_score=True,
)

# Multiple camera threads append to the same log file; guard writes
# so lines from different cameras can't interleave mid-line.
_log_lock = threading.Lock()

# A live source dropping a frame is normal (network hiccup); a local
# file returning no frame means it legitimately ended. Only sources
# that look "live" get reconnect-and-retry instead of stopping.
# Backoff doubles each consecutive failed attempt rather than retrying
# on a fixed interval - a supervised feed that's mid-restart doesn't
# come back faster because we hammer it, and a tight fixed-interval
# retry loop is exactly what most live camera gateways rate-limit or
# flag as abuse.
RECONNECT_DELAY_SECONDS = 2
MAX_RECONNECT_DELAY_SECONDS = 30
MAX_RECONNECT_ATTEMPTS = 15

# A handful of consecutive failed reads is normal RTP noise and often
# clears on its own within a frame or two - only escalate to a full
# reconnect (which drops and reopens the capture) after this many in
# a row, rather than on the very first one.
QUICK_RETRY_ATTEMPTS = 8


def _is_live_source(source):
    if isinstance(source, int):
        return True
    if isinstance(source, str):
        return source.startswith(("rtsp://", "http://", "https://"))
    return False


def _coerce_source(source):
    """A camera's video_path may be a local file path, an rtsp:// /
    http(s):// stream URL (IP camera, VMS/NVR export, ONVIF-resolved
    stream), or a numeric device index (USB capture card, e.g. for an
    analog camera feeding through a DVR/encoder). cv2.VideoCapture
    handles all of these the same way once the index is a real int."""

    if isinstance(source, str) and source.isdigit():
        return int(source)
    return source


class _SerializedModel:
    """Wraps a model so many camera threads can share one loaded copy
    instead of each paying for its own. Only safe for models called
    without persisted state between calls (plain single-shot inference,
    no tracker) - the vehicle classifier and OCR reader both qualify,
    the plate tracker below does not (its persist=True tracker state
    is exactly what must stay per-camera). Calls are serialized rather
    than assumed thread-safe, since neither ultralytics' nor
    fast-plate-ocr's docs make concurrent-call guarantees for a single
    instance - one shared copy in memory is the actual win at scale
    (a GPU has to serialize the compute anyway); this just makes sure
    that's also correct, not merely cheaper."""

    def __init__(self, model):
        self._model = model
        self._lock = threading.Lock()

    def __call__(self, *args, **kwargs):
        with self._lock:
            return self._model(*args, **kwargs)

    def run(self, *args, **kwargs):
        with self._lock:
            return self._model.run(*args, **kwargs)


class CameraWorker:
    """Runs the full detect -> OCR -> track -> persist -> alert
    pipeline for one camera feed, in its own thread.

    The plate-tracking model, tracker state, and DB connection are
    always private to this camera - none of those are safe to share.
    vehicle_model/reader may be a shared instance passed in from the
    caller (see _SerializedModel above); if omitted, this camera loads
    its own, same as before. Cross-camera correlation happens
    afterwards, in the database.
    """

    def __init__(self, camera_cfg, device, vehicle_model=None, reader=None):
        self.camera_cfg = camera_cfg
        self.camera_id = camera_cfg["camera_id"]
        self.device = device
        self.detect_imgsz = camera_cfg.get("imgsz", config.DETECT_IMGSZ)
        self.detect_every_n = camera_cfg.get("detect_every_n", config.DETECT_EVERY_N)

        self.model = YOLO(config.MODEL_PATH)
        self.vehicle_model = vehicle_model if vehicle_model is not None else YOLO(config.VEHICLE_MODEL_PATH)
        self.reader = reader if reader is not None else create_reader()
        self.tracker = PlateTracker()
        self.db_conn = db.connect()

        self.lock = threading.Lock()
        self.latest_frame = None
        self.recent_detections = []
        self.active_alerts = []
        self.status = "starting"
        self.frame_count = 0

        self._stop = threading.Event()
        self._track_last_score = {}
        self._frame_shape = None

        # Dedup key -> {timestamp, score, unclear, detection_id,
        # snapshot} for the most recent thing logged under that key.
        # Catches the case ByteTrack's own per-track consensus can't:
        # the *same* vehicle resurfacing under a brand new track_id (a
        # looping demo clip, a vehicle that left and re-entered frame,
        # or a later/weaker read of a plate already confirmed) isn't a
        # new sighting worth a second row - see _log_update.
        self._last_logged = {}

        # Best (highest-OCR-confidence) frame seen so far per track,
        # saved as the evidence photo when that track finalizes.
        self._track_best_conf = {}
        self._track_best_frame = {}

        # Boxes drawn on the most recent frame that actually ran
        # detection - redrawn as-is on the frames in between (see
        # detect_every_n) so the overlay stays visually continuous
        # instead of flickering on/off every skipped frame.
        self._last_drawn_boxes = []

        # A second, independent tracking pipeline for *vehicles* (not
        # plates) - catches the case the plate model has a real, known
        # blind spot for: a vehicle (chiefly motorcycles) that the
        # vehicle classifier plainly sees, but the plate detector never
        # finds a plate for at all. Without this, such a vehicle gets
        # no track_id from the plate pipeline and is invisible to the
        # whole system - not even logged as unclear. This tracker is
        # its own private instance per camera (never shared, unlike
        # self.vehicle_model) precisely because persist=True-style
        # tracking state can't safely be shared across cameras - see
        # the shared vehicle_model/reader docstring above.
        self.vehicle_tracker = BYTETracker(_VEHICLE_TRACKER_ARGS)
        self._vehicle_active_ids = set()
        self._vehicle_claimed_ids = set()
        self._vehicle_type_by_id = {}
        self._vehicle_last_box = {}
        self._vehicle_best_conf = {}
        self._vehicle_best_frame = {}

        config.EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

        db.upsert_camera(
            self.db_conn,
            self.camera_id,
            camera_cfg["name"],
            camera_cfg["department"],
            camera_cfg["location"],
            camera_cfg["lat"],
            camera_cfg["lon"],
        )

    def stop(self):
        self._stop.set()

    def snapshot(self):
        """Thread-safe read of what this camera currently knows,
        for the combined dashboard to render."""

        with self.lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
            return frame, list(self.recent_detections), list(self.active_alerts), self.status

    def _save_evidence(self, frame, plate_text, timestamp, track_id):
        """Write the given frame to disk as this detection's evidence
        photo. Returns the filename, or None if there's no frame.
        Takes the frame directly rather than looking it up itself -
        the plate and vehicle-only tracking pipelines each keep their
        own best-frame cache, with track_id spaces that aren't
        comparable (two independent BYTETracker instances, each
        numbering from 1), so the caller resolves its own frame."""

        if frame is None:
            return None

        filename = (
            f"{self.camera_id}_{plate_text}_"
            f"{timestamp:%Y%m%d_%H%M%S}_{track_id}.jpg"
        )
        path = config.EVIDENCE_DIR / filename
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return filename

    @staticmethod
    def _is_better_reading(prior_unclear, prior_score, new_unclear, new_score):
        """Whether a newly-arrived reading of the same recent sighting
        should replace what's already stored. A confirmed reading
        always beats an unclear one, regardless of score (a low-weight
        confirmed consensus is still a trustworthy plate; a high-score
        unclear guess is still just a guess); otherwise higher score
        wins within the same tier."""

        if prior_unclear and not new_unclear:
            return True
        if not prior_unclear and new_unclear:
            return False
        return new_score > prior_score

    @staticmethod
    def _plausible_same_vehicle(box1, box2, frame_shape):
        """Rough spatial-continuity check: could box2 plausibly be the
        same physical vehicle as box1, just at a different moment (e.g.
        closer to the camera now)? Compares how far the box centers
        have moved relative to the frame size - tolerant enough to
        bridge a real, slow approach across many seconds, tight enough
        to reject a different vehicle elsewhere in the frame that just
        happens to share a type and arrive around the same time."""

        if box1 is None or box2 is None or frame_shape is None:
            return False
        x1a, y1a, x2a, y2a = box1
        x1b, y1b, x2b, y2b = box2
        cx_a, cy_a = (x1a + x2a) / 2, (y1a + y2a) / 2
        cx_b, cy_b = (x1b + x2b) / 2, (y1b + y2b) / 2
        h, w = frame_shape[0], frame_shape[1]
        diag = (w * w + h * h) ** 0.5
        dist = ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5
        return dist < diag * 0.4

    def _log_update(self, update, track_id, score, evidence_frame):
        timestamp = datetime.now()

        # A vehicle approaching the camera routinely gets tracked as
        # *several different track_ids* - ByteTrack losing and
        # re-acquiring it as its apparent size/sharpness changes from
        # "distant blur" to "close and legible" over what can be many
        # seconds on a wide junction. Two fragments of the very same
        # car almost never produce the same OCR guess, so matching on
        # exact text alone misses this - and matching on type+timing
        # alone isn't precise enough either: too short a window misses
        # a slow approach, too long a window merges two different real
        # vehicles of the same type that simply passed close together
        # (confirmed the hard way - see git history). So a third signal
        # is required alongside type+timing: is the new detection's
        # position a plausible continuation of the old one (roughly
        # where a vehicle moving toward/through frame would now be),
        # not just "some car, recently, somewhere in the picture".
        #
        # That type-based pool holds *only unclear* sightings, and a
        # confirmed reading is never at risk of being matched *against*
        # it - only of *upgrading* an unclear entry it finds there. A
        # shared type, timing, and rough position still isn't proof
        # two *confirmed* reads are the same car; it's only safe to
        # lean on for merging fragments that have nothing more precise
        # to go on in the first place.
        text_key = ("text", update.plate_text) if update.plate_text else None
        type_key = ("type", update.vehicle_type or "vehicle")

        text_prior = self._last_logged.get(text_key) if text_key else None
        text_recent = bool(
            text_prior and timestamp - text_prior["timestamp"] < timedelta(minutes=config.DUPLICATE_SUPPRESS_MINUTES)
        )
        type_prior = self._last_logged.get(type_key)
        type_recent = bool(
            type_prior and type_prior["unclear"]
            and timestamp - type_prior["timestamp"] < timedelta(seconds=config.UNCLEAR_SUPPRESS_SECONDS)
            and self._plausible_same_vehicle(type_prior.get("last_box"), update.last_box, self._frame_shape)
        )

        matched_via_type = False
        if text_recent:
            prior, is_recent = text_prior, True
        elif type_recent:
            prior, is_recent, matched_via_type = type_prior, True, True
        else:
            prior, is_recent = None, False

        if is_recent and not self._is_better_reading(prior["unclear"], prior["score"], update.unclear, score):
            print(f"[{self.camera_id}] Suppressed duplicate log for {update.plate_text or update.vehicle_type} "
                  f"(kept the reading from {(timestamp - prior['timestamp']).seconds}s ago)")
            return

        display_plate = update.plate_text if update.plate_text else "UNREADABLE"
        car_label = update.car_id if update.car_id is not None else "—"
        status_label = "UNCLEAR" if update.unclear else "confirmed"
        plate_status = "unclear" if update.unclear else "confirmed"

        with self.lock:
            self.recent_detections.append({
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "camera_id": self.camera_id,
                "car_id": update.car_id,
                "plate": display_plate,
                "weight": update.weight,
            })

        print(
            f"[{self.camera_id}] Car {car_label} Track {track_id} "
            f"Plate {display_plate} ({status_label}) Detection {score:.2f} "
            f"OCR weight {update.weight:.2f}"
        )

        with _log_lock:
            with open(config.LOG_FILE, "a", encoding="utf-8") as f:
                f.write(
                    f"{timestamp:%Y-%m-%d %H:%M:%S} | Camera: {self.camera_id} | "
                    f"Car ID: {car_label} | Track: {track_id} | "
                    f"Plate: {display_plate} ({status_label}) | Detection: {score:.2f} | "
                    f"OCR weight: {update.weight:.2f}\n"
                )

        snapshot_filename = self._save_evidence(evidence_frame, display_plate, timestamp, track_id)

        if is_recent:
            # A better reading of the same recent sighting - update
            # the existing row in place instead of adding a second one,
            # and drop the old evidence photo it's replacing.
            db.update_detection(
                self.db_conn, prior["detection_id"], display_plate, timestamp,
                float(score), float(update.weight),
                snapshot_path=snapshot_filename, vehicle_type=update.vehicle_type,
                plate_status=plate_status
            )
            detection_id = prior["detection_id"]
            old_snapshot = prior.get("snapshot")
            if old_snapshot and old_snapshot != snapshot_filename:
                (config.EVIDENCE_DIR / old_snapshot).unlink(missing_ok=True)
        else:
            detection_id = db.insert_detection(
                self.db_conn, display_plate, self.camera_id,
                timestamp, float(score), float(update.weight),
                snapshot_path=snapshot_filename, vehicle_type=update.vehicle_type,
                plate_status=plate_status
            )

        entry = {
            "timestamp": timestamp, "score": score, "unclear": update.unclear,
            "detection_id": detection_id, "snapshot": snapshot_filename,
            "last_box": update.last_box,
        }
        if text_key:
            self._last_logged[text_key] = entry

        if update.unclear:
            self._last_logged[type_key] = entry
        elif matched_via_type:
            # Graduated from unclear to confirmed through the type-key
            # match - this slot is resolved now, stop matching future
            # arrivals of this vehicle type against it.
            self._last_logged.pop(type_key, None)

        # An upgrade to an already-logged, already-alerted-on sighting
        # shouldn't re-fire the same alert a second time.
        if is_recent:
            return

        watch_hit = db.check_watchlist(self.db_conn, update.plate_text)
        if watch_hit:
            db.create_alert(
                self.db_conn, update.plate_text, watch_hit["category"],
                self.camera_id, detection_id, timestamp
            )
            print(
                f"\n*** HIGH PRIORITY ALERT [{self.camera_id}] ***\n"
                f"Plate: {update.plate_text}\n"
                f"Watchlist: {watch_hit['category']} ({watch_hit['priority']})\n"
            )
            with self.lock:
                self.active_alerts.append({
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "camera_id": self.camera_id,
                    "plate": update.plate_text,
                    "category": watch_hit["category"],
                    "priority": watch_hit["priority"],
                })

    def _detect_vehicle_boxes(self, frame):
        """Run the pretrained COCO detector once per frame. Returns
        ([(x1, y1, x2, y2, class_name), ...], raw_boxes) - the simple
        list for matching a found plate to its vehicle, and the raw
        ultralytics Boxes object (or None) for feeding this camera's
        own vehicle-level BYTETracker (see _track_vehicles). Kept
        separate from the plate model since it's off the shelf, not
        trained on our data."""

        results = self.vehicle_model(
            frame,
            conf=0.35,
            imgsz=config.VEHICLE_DETECT_IMGSZ,
            device=self.device,
            verbose=False,
            classes=list(config.VEHICLE_CLASS_NAMES.keys()),
        )

        # .cpu() matters here specifically: BYTETracker (fed this
        # object directly - see _track_vehicles) converts it to numpy
        # internally without moving it off the GPU first, and crashes
        # on a CUDA tensor if it's not already CPU-resident.
        raw_boxes = results[0].boxes.cpu() if results and results[0].boxes is not None else None

        boxes = []
        if raw_boxes is not None:
            xyxy = raw_boxes.xyxy.cpu().numpy()
            classes = raw_boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), cls_id in zip(xyxy, classes):
                name = config.VEHICLE_CLASS_NAMES.get(int(cls_id))
                if name:
                    boxes.append((x1, y1, x2, y2, name))
        return boxes, raw_boxes

    @staticmethod
    def _match_vehicle_type(vehicle_boxes, plate_box):
        """Which detected vehicle (if any) this plate belongs to, by
        checking whether the plate's center falls inside its box."""

        px1, py1, px2, py2 = plate_box
        cx, cy = (px1 + px2) / 2, (py1 + py2) / 2
        for vx1, vy1, vx2, vy2, name in vehicle_boxes:
            if vx1 <= cx <= vx2 and vy1 <= cy <= vy2:
                return name
        return "vehicle"

    def _track_vehicles(self, frame, plate_boxes, vehicle_raw):
        """Feeds this frame's vehicle detections into this camera's own
        standalone BYTETracker, and logs any vehicle whose track just
        ended without ever having a plate detected inside its box -
        the motorcycle-plate blind spot this pipeline exists to catch.
        A vehicle the plate model DID find something for (even an
        unclear reading) is left alone here - it's already handled by
        the normal plate pipeline; this only fills the gap where that
        pipeline produced nothing at all."""

        if vehicle_raw is None:
            return

        tracked = self.vehicle_tracker.update(vehicle_raw, img=frame)

        current_ids = set()
        for row in tracked:
            vx1, vy1, vx2, vy2, vid, vscore, vcls = row[:7]
            vid = int(vid)
            current_ids.add(vid)

            self._vehicle_type_by_id[vid] = config.VEHICLE_CLASS_NAMES.get(int(vcls), "vehicle")
            self._vehicle_last_box[vid] = (int(vx1), int(vy1), int(vx2), int(vy2))

            if vscore > self._vehicle_best_conf.get(vid, -1):
                self._vehicle_best_conf[vid] = vscore
                self._vehicle_best_frame[vid] = frame.copy()

            if vid not in self._vehicle_claimed_ids:
                for px1, py1, px2, py2 in plate_boxes:
                    cx, cy = (px1 + px2) / 2, (py1 + py2) / 2
                    if vx1 <= cx <= vx2 and vy1 <= cy <= vy2:
                        self._vehicle_claimed_ids.add(vid)
                        break

        finished_ids = self._vehicle_active_ids - current_ids
        self._vehicle_active_ids = current_ids
        for vid in finished_ids:
            self._finalize_vehicle(vid)

    def _finalize_vehicle(self, vid):
        """Logs one vehicle-only track if the plate pipeline never
        claimed it; always cleans up its per-track state either way."""

        claimed = vid in self._vehicle_claimed_ids
        vtype = self._vehicle_type_by_id.pop(vid, "vehicle")
        vbox = self._vehicle_last_box.pop(vid, None)
        evidence_frame = self._vehicle_best_frame.pop(vid, None)
        self._vehicle_best_conf.pop(vid, None)
        self._vehicle_claimed_ids.discard(vid)

        if claimed:
            return

        update = PlateUpdate(
            car_id=None, plate_text="", weight=0.0,
            vehicle_type=vtype, unclear=True, last_box=vbox
        )
        self._log_update(update, f"veh{vid}", 0.0, evidence_frame)

    def _finalize_all_vehicles(self):
        """Flushes every still-active vehicle-only track (call at
        video end, mirroring PlateTracker.finalize_all())."""

        for vid in list(self._vehicle_active_ids):
            self._finalize_vehicle(vid)
        self._vehicle_active_ids.clear()

    def _set_status(self, status):
        if status == self.status:
            return
        self.status = status
        db.update_camera_status(self.db_conn, self.camera_id, status)

    def run(self):
        source = _coerce_source(self.camera_cfg["video_path"])
        is_live = _is_live_source(source)

        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            self._set_status("offline")
            print(f"[{self.camera_id}] Could not open video source")
            return

        start_frame = self.camera_cfg.get("start_frame", 0)
        if start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # A live RTSP/HTTP source already arrives paced in real time by
        # the network - reading it as fast as possible is correct there.
        # A local recorded file has no such pacing: cv2.read() will
        # hand back frames as fast as the disk/decoder allows, which is
        # usually much faster (or, under load, slower) than the clip's
        # own timeline. Reference every basic OpenCV video-player
        # example for this: pace reads against the file's own fps so
        # playback matches real elapsed time instead of running in
        # fast-forward or slow motion.
        frame_interval = 0.0
        if not is_live:
            native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_interval = 1.0 / native_fps
        next_frame_due = time.monotonic()

        self._set_status("online")
        reconnect_attempts = 0
        quick_retries = 0
        frame_index = 0

        while not self._stop.is_set():
            ret, frame = cap.read()

            if not ret:
                if not is_live:
                    # A local file/demo clip stands in for a camera
                    # that's always "on" - loop it instead of letting
                    # the feed go dark and freeze at the last frame.
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    continue

                # A dropped RTP packet or two is normal network noise
                # and usually clears on the very next read - only pay
                # for a full reconnect once it's clearly not transient.
                quick_retries += 1
                if quick_retries <= QUICK_RETRY_ATTEMPTS:
                    continue

                quick_retries = 0
                reconnect_attempts += 1
                if reconnect_attempts > MAX_RECONNECT_ATTEMPTS:
                    print(f"[{self.camera_id}] giving up after "
                          f"{MAX_RECONNECT_ATTEMPTS} failed reconnects")
                    self._set_status("offline")
                    break

                delay = min(
                    RECONNECT_DELAY_SECONDS * (2 ** (reconnect_attempts - 1)),
                    MAX_RECONNECT_DELAY_SECONDS
                )
                print(f"[{self.camera_id}] stream read failed "
                      f"({reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS}), "
                      f"reconnecting in {delay}s...")
                self._set_status("reconnecting")

                cap.release()
                time.sleep(delay)
                cap = cv2.VideoCapture(source)
                continue

            reconnect_attempts = 0
            quick_retries = 0
            self._set_status("online")
            self._frame_shape = frame.shape

            frame_index += 1
            # Consecutive frames barely differ - running the full
            # detect+OCR+classify pipeline on every single one is
            # mostly wasted GPU time (same tradeoff production NVR
            # software like Frigate makes: detection runs at a fraction
            # of the actual display/record frame rate). Every frame
            # still gets shown at full rate below; only the expensive
            # inference is skipped on the frames in between.
            run_detection = (frame_index % self.detect_every_n == 0)

            if run_detection:
                results = self.model.track(
                    frame,
                    conf=config.CONFIDENCE,
                    imgsz=self.detect_imgsz,
                    device=self.device,
                    verbose=False,
                    persist=True,
                    tracker="bytetrack.yaml"
                )

                for result in results:
                    if result.boxes is None:
                        continue

                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    track_ids = (
                        result.boxes.id.int().cpu().numpy()
                        if result.boxes.id is not None else [None] * len(boxes)
                    )

                    frame_track_ids = []
                    drawn_boxes = []

                    # Always runs now, not just when a plate was found -
                    # the vehicle-only tracking pipeline below needs a
                    # vehicle reading on every detection frame regardless
                    # of whether the plate model found anything, to catch
                    # vehicles (chiefly motorcycles) it never does.
                    vehicle_boxes, vehicle_raw = self._detect_vehicle_boxes(frame)

                    for box, score, track_id in zip(boxes, scores, track_ids):
                        x1, y1, x2, y2 = map(int, box)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)

                        if x2 <= x1 or y2 <= y1:
                            continue

                        track_id = int(track_id) if track_id is not None else None
                        if track_id is not None:
                            frame_track_ids.append(track_id)
                            self._track_last_score[track_id] = score

                        px1, py1, px2, py2 = pad_box(x1, y1, x2, y2, frame.shape)
                        ocr_crop = frame[py1:py2, px1:px2]
                        if ocr_crop.size == 0:
                            continue

                        plate_text, ocr_confidence = read_plate_text(self.reader, ocr_crop)
                        vehicle_type = self._match_vehicle_type(vehicle_boxes, (x1, y1, x2, y2))
                        self.tracker.record(track_id, plate_text, ocr_confidence, vehicle_type, box=(x1, y1, x2, y2))

                        best = self.tracker.best_known(track_id)
                        label = f"ID:{best.car_id} {best.plate_text}" if best else None
                        dashboard.draw_detection_box(frame, x1, y1, x2, y2, label)
                        drawn_boxes.append((x1, y1, x2, y2, label))

                        if track_id is not None and ocr_confidence > self._track_best_conf.get(track_id, -1):
                            self._track_best_conf[track_id] = ocr_confidence
                            self._track_best_frame[track_id] = frame.copy()

                    self._last_drawn_boxes = drawn_boxes

                    for finished_track_id, update in self.tracker.sync(frame_track_ids):
                        evidence_frame = self._track_best_frame.pop(finished_track_id, None)
                        self._track_best_conf.pop(finished_track_id, None)
                        self._log_update(
                            update, finished_track_id,
                            self._track_last_score.get(finished_track_id, 0.0),
                            evidence_frame
                        )

                    # Drop any captured frames for tracks that finalized
                    # without ever producing a loggable reading, so they
                    # don't sit in memory for the rest of the run.
                    stale = set(self._track_best_frame) & self.tracker.finalized_track_ids
                    for stale_id in stale:
                        self._track_best_frame.pop(stale_id, None)
                        self._track_best_conf.pop(stale_id, None)

                    self._track_vehicles(frame, boxes, vehicle_raw)
            else:
                # Redraw the last real detection's boxes as-is, so the
                # overlay stays visually continuous instead of
                # flickering on/off between detection passes.
                for x1, y1, x2, y2, label in self._last_drawn_boxes:
                    dashboard.draw_detection_box(frame, x1, y1, x2, y2, label)

            with self.lock:
                self.latest_frame = frame
                self.frame_count += 1

            if frame_interval:
                next_frame_due += frame_interval
                sleep_for = next_frame_due - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    # Fell behind (e.g. a slow detection frame) - resync
                    # to now instead of bursting through frames to catch
                    # up, which would just trade one kind of jank for
                    # another.
                    next_frame_due = time.monotonic()

        for finished_track_id, update in self.tracker.finalize_all():
            evidence_frame = self._track_best_frame.pop(finished_track_id, None)
            self._log_update(
                update, finished_track_id,
                self._track_last_score.get(finished_track_id, 0.0),
                evidence_frame
            )
        self._finalize_all_vehicles()

        cap.release()
        self._set_status("offline")
        self.db_conn.close()
        print(f"[{self.camera_id}] finished")
