from pathlib import Path

# src/config.py -> project root is one level up
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = str(PROJECT_ROOT / "model" / "exp.pt")

# Pretrained COCO detector, used only to classify vehicle type
# (car/truck/bus/motorcycle) around a detected plate - not trained on
# our data, so it's kept separate from the plate detector above.
VEHICLE_MODEL_PATH = str(PROJECT_ROOT / "model" / "yolov8n.pt")
VEHICLE_DETECT_IMGSZ = 480
VEHICLE_CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
LOG_FILE = str(PROJECT_ROOT / "logs" / "detections.txt")

# One evidence photo is saved per logged (finalized) detection - not
# per frame, so this stays small even over a long run.
EVIDENCE_DIR = PROJECT_ROOT / "data" / "evidence"

_TOLL_VIDEO = str(PROJECT_ROOT / "data" / "cars_extended.mp4")

# Each entry is one camera feed. video_path accepts any of:
#   - a local file path            (testing/demo, e.g. _TOLL_VIDEO below)
#   - "rtsp://user:pass@host/..."  (IP cameras, most VMS/NVR/DVR RTSP
#                                    exports, and ONVIF cameras once
#                                    their RTSP URL has been resolved)
#   - "http://..." / "https://..." (MJPEG / HTTP camera streams)
#   - an integer device index, e.g. 0
#                                   (USB capture card - the standard way
#                                    to bring an analog camera in, since
#                                    the capture card/DVR digitizes it
#                                    and exposes it as a local device or
#                                    its own RTSP export)
# camera_worker.py dispatches all of these through the same
# cv2.VideoCapture call and auto-reconnects on drops for anything that
# looks "live" (rtsp/http/device index) rather than treating a dropped
# frame as end-of-stream the way it does for a local file.
#
# start_frame lets one source file stand in for a second, "further down
# the road" camera during testing, until real distinct footage/streams
# are available.
CAMERAS = [
    {
        "camera_id": "CAM-01",
        "name": "Toll Plaza Camera 1",
        "department": "Police",
        "location": "Demo Toll Plaza, Ahmedabad",
        "lat": 23.0225,
        "lon": 72.5714,
        "video_path": _TOLL_VIDEO,
        "start_frame": 0,
    },
    {
        "camera_id": "CAM-03",
        "name": "New Bypass Jn Camera",
        "department": "Municipal",
        "location": "New Bypass Nr 66KV FIX-2 (Vadla Fatak)",
        "lat": 23.06,
        "lon": 72.62,
        "video_path": str(PROJECT_ROOT / "data" / "locatiopn1.mp4"),
        "start_frame": 0,
        # Real overhead traffic-junction footage - plates are small
        # and distant, and 1280 measurably raises detection confidence
        # here (0.94 vs 0.87 in testing) where it didn't matter for the
        # close-up toll cameras above.
        "imgsz": 1280,
    },
    {
        # Real Dahua-protocol NVR on the local network - confirmed
        # working with RTSP (channel=3) + TCP transport after tracking
        # down: (a) the correct channel via trial, (b) that UDP RTP
        # was silently losing packets on this network ("bad cseq"),
        # and (c) that its ONVIF auth was unrelated to both (its clock
        # is genuinely 18 days off, which blocks ONVIF's WS-Security
        # digest but not plain RTSP auth). This channel happens to be
        # an indoor office camera, not a road, so it won't produce
        # plate detections - it's here to prove the NVR onboarding
        # path itself works against real hardware, channel/location
        # can be swapped once a road-facing channel is identified.
        "camera_id": "CAM-04",
        "name": "NVR Channel 3",
        "department": "Municipal",
        "location": "Local NVR (10.0.42.152), Channel 3",
        "lat": 23.03,
        "lon": 72.58,
        # Real credentials for this local NVR are not committed - fill
        # them in locally to use this camera (see README).
        "video_path": "rtsp://<user>:<password>@10.0.42.152:554/cam/realmonitor?channel=3&subtype=0",
        "start_frame": 0,
    },
    # Example real feed - uncomment and fill in once a government/VMS
    # RTSP URL or ONVIF-resolved stream is available:
    # {
    #     "camera_id": "CAM-05",
    #     "name": "Government Feed",
    #     "department": "Municipal",
    #     "location": "Provided by organizers",
    #     "lat": 23.03,
    #     "lon": 72.58,
    #     "video_path": "rtsp://user:pass@192.168.1.50:554/stream1",
    #     "start_frame": 0,
    # },
]

# Sentinel sandbox integration (Gujarat CCTV hackathon test grid).
# Fill in once registered - POST /api/catalogue/sync (web_api.py) reads
# this host's camera catalogue and onboards whatever it currently
# lists. Per the sandbox's own integration guide: "the catalogue is
# the contract, the URL pattern is not" - camera ids and the set of
# available cameras can change, so this is read live at sync time
# rather than hardcoded. Leave blank to disable; nothing else in this
# project depends on it being set.
SANDBOX_HOST = ""  # e.g. "sentinel.example.gov.in" - no scheme, no port
SANDBOX_INGEST_PATH = "/api/ingest"

CONFIDENCE = 0.5

# Default YOLO inference size, used by any camera that doesn't set its
# own "imgsz" in its CAMERAS entry. 1280 helps when a camera has small
# or distant plates (e.g. a wide highway shot), but costs real
# throughput - measured ~16 fps/camera at 640 vs ~12 fps/camera at
# 1280 with 2 concurrent workers on this GPU. Set it per-camera below
# instead of raising this default, so a close-up feed that already
# reads fine at 640 doesn't pay for resolution it doesn't need.
DETECT_IMGSZ = 640

# Run the full detect+OCR+classify pipeline on 1 out of every N frames
# rather than every single one - consecutive frames barely differ, so
# this is mostly wasted GPU time otherwise (the same tradeoff Frigate
# NVR makes: their guidance is detection rarely needs to run above
# ~10fps even though the recorded/displayed stream stays at full fps).
# Every frame is still shown at full rate - only inference is skipped
# on the frames in between (see camera_worker.py's run_detection).
# Override per-camera with "detect_every_n" in a CAMERAS entry if a
# particular feed needs tighter tracking (fast traffic, tiny window
# between frames) or can tolerate looser (slow/static scenes).
DETECT_EVERY_N = 2

# A vehicle idling in frame, or - for a looping demo/recorded clip -
# the source looping back to the start, can otherwise cause the same
# plate to be logged again and again under a brand new track_id: not a
# new sighting, just the same one being re-detected. Suppress a repeat
# log for the same plate at the same camera within this window; a
# genuine revisit well past it (the vehicle actually passing through
# again later) still logs normally.
DUPLICATE_SUPPRESS_MINUTES = 15

# Unclear sightings can't be deduped by plate text (there's no
# trustworthy text to key on - two garbled guesses from the *same*
# lingering vehicle, e.g. one far away then near the camera, usually
# won't even match each other). This dedupes by (camera, vehicle type)
# instead - but type+timing alone isn't precise enough on its own: too
# short a window misses a real slow approach across a wide junction,
# too long a window merges two different real vehicles of the same
# type that simply passed close together (confirmed against real
# footage both ways - 20s missed a genuine far-then-near approach that
# took longer than that; an earlier 2-minute window merged two
# distinct cars 100+ seconds apart on a flowing bypass junction).
# CameraWorker._log_update now also requires the new detection's
# position to be a plausible continuation of the old one before it
# counts as a match, so this window can safely be longer - the
# position check is what actually keeps genuinely different vehicles
# apart, not this number.
UNCLEAR_SUPPRESS_SECONDS = 90

PLATE_OCR_MODEL = "global-plates-mobile-vit-v2-model"

# A reading only counts toward a track's consensus above this OCR confidence
MIN_VOTE_CONFIDENCE = 0.50

# A track's consensus plate is only logged once it's backed by either
# this many matching reads, or this much cumulative OCR confidence.
# Was 3 - but DETECT_EVERY_N means only every other frame even gets an
# OCR attempt, halving how many reads any one track can accumulate
# before it leaves frame, so a genuinely clear plate could still fall
# short on a brief pass. Lowered to 2, which is now safe to do:
# CameraWorker._log_update merges fragments of the same physical
# vehicle across track_ids, so a short fragment that crosses this bar
# a little early on 2 lucky frames still gets corrected later if a
# clearer fragment of the same vehicle comes in and outscores it.
WELL_SUPPORTED_COUNT = 2
WELL_SUPPORTED_WEIGHT = 2.2

# Side panel shown next to the video
PANEL_WIDTH = 420
MAX_PANEL_ROWS = 10

# =========================
# DATABASE
# =========================

DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "licence_plate"
DB_USER = "plate_admin"
DB_PASSWORD = "plate_admin_dev"
