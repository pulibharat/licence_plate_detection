"""Unified server: starts the camera pipeline (same workers app.py
uses) in-process, and serves a REST API + the SENTINEL dashboard over
that same live state and the Postgres data it writes.

Run: uvicorn web_api:app --host 127.0.0.1 --port 8000
"""

import asyncio
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import cv2
import requests
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import db
from camera_worker import CameraWorker, _SerializedModel
from plate_ocr import create_reader
from ultralytics import YOLO

WEB_DIR = Path(__file__).resolve().parent / "web"

# camera_id -> CameraWorker, populated at startup and grown at runtime
# whenever an operator onboards a new feed through the wizard.
workers = {}

# camera_id -> the Thread running that worker's run() loop. Kept so an
# edit/reconnect can join() the old thread (and so its cap.release()
# has actually happened) before opening the replacement - starting a
# second connection while the first is still live can get refused by
# the source itself (some NVRs cap concurrent RTSP clients per channel).
worker_threads = {}

# camera_id -> protocol label. Populated from config.CAMERAS at startup
# using _protocol_for(); cameras added later via POST /api/cameras record
# the type the operator actually selected (RTSP vs ONVIF vs VMS all end
# up as an rtsp:// URL under the hood, but the operator's stated intent
# is meaningful metadata worth keeping distinct).
camera_protocols = {}

DEVICE = 0 if torch.cuda.is_available() else "cpu"

# Loaded once and shared read-only across every camera, instead of
# each CameraWorker loading its own private copy. Both are stateless
# per call (no persist=True tracker), unlike the plate-tracking model,
# so this is safe - and it's the difference between N cameras costing
# roughly 3x one camera's model memory vs. roughly 1x, which matters
# once N gets into the dozens (see _SerializedModel in camera_worker.py
# for why calls are serialized rather than just shared bare).
_shared_vehicle_model = _SerializedModel(YOLO(config.VEHICLE_MODEL_PATH))
_shared_ocr_reader = _SerializedModel(create_reader())


def _protocol_for(video_path):
    if isinstance(video_path, int):
        return "Capture Device"
    if isinstance(video_path, str):
        if video_path.startswith("rtsp://"):
            return "RTSP"
        if video_path.startswith(("http://", "https://")):
            return "HTTP/MJPEG"
    return "Local File"


def _start_worker(cam_cfg, protocol_label):
    worker = CameraWorker(
        cam_cfg, DEVICE,
        vehicle_model=_shared_vehicle_model, reader=_shared_ocr_reader
    )
    workers[worker.camera_id] = worker
    camera_protocols[worker.camera_id] = protocol_label
    thread = threading.Thread(target=worker.run, daemon=True)
    worker_threads[worker.camera_id] = thread
    thread.start()
    return worker


def _stop_worker(camera_id, join_timeout=35):
    """Signal a worker to stop and wait for its thread to actually
    exit, so its capture is released before anything reconnects to
    the same source. A live RTSP read can block for a while, so this
    can take a few seconds - join_timeout just stops it hanging the
    request forever if the thread is unusually slow to notice."""

    worker = workers.pop(camera_id, None)
    thread = worker_threads.pop(camera_id, None)
    if worker is None:
        return
    worker.stop()
    if thread is not None:
        thread.join(timeout=join_timeout)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("CUDA available:", torch.cuda.is_available())

    schema_conn = db.connect()
    db.init_schema(schema_conn)
    schema_conn.close()

    for cam_cfg in config.CAMERAS:
        _start_worker(cam_cfg, _protocol_for(cam_cfg["video_path"]))

    print(f"Started {len(workers)} camera worker(s): {', '.join(workers)}")

    yield

    for worker in workers.values():
        worker.stop()


app = FastAPI(title="CCTV Intelligence Platform API", lifespan=lifespan)


class NewCameraRequest(BaseModel):
    camera_id: str
    name: str
    department: str = "Police"
    location: str = ""
    lat: float = 23.03
    lon: float = 72.58
    protocol: str
    video_path: str


class UpdateCameraRequest(BaseModel):
    name: str | None = None
    department: str | None = None
    location: str | None = None
    lat: float | None = None
    lon: float | None = None
    protocol: str | None = None
    video_path: str | None = None


def _serialize(row):
    """RealDictRow -> plain JSON-safe dict (datetimes -> isoformat)."""

    out = dict(row)
    for key, value in out.items():
        if isinstance(value, datetime):
            out[key] = value.isoformat()
    return out


@app.get("/api/stats")
def get_stats():
    conn = db.connect()
    try:
        return db.get_stats(conn)
    finally:
        conn.close()


@app.get("/api/cameras")
def get_cameras():
    conn = db.connect()
    try:
        rows = [_serialize(row) for row in db.list_cameras(conn)]
    finally:
        conn.close()

    for row in rows:
        row["protocol"] = camera_protocols.get(row["camera_id"], "Unknown")
        worker = workers.get(row["camera_id"])
        row["frame_count"] = worker.frame_count if worker else None

    return rows


@app.post("/api/cameras")
def add_camera(req: NewCameraRequest):
    camera_id = req.camera_id.strip().upper()
    if not camera_id:
        raise HTTPException(status_code=400, detail="camera_id is required")
    if camera_id in workers:
        raise HTTPException(status_code=409, detail=f"{camera_id} is already onboarded")

    cam_cfg = {
        "camera_id": camera_id,
        "name": req.name or camera_id,
        "department": req.department,
        "location": req.location,
        "lat": req.lat,
        "lon": req.lon,
        "video_path": req.video_path,
        "start_frame": 0,
    }

    worker = _start_worker(cam_cfg, req.protocol)
    print(f"[{camera_id}] onboarded live via wizard - protocol={req.protocol} "
          f"source={req.video_path!r}")

    return {"status": "ok", "camera_id": worker.camera_id}


def _parse_catalogue_entry(entry):
    """Best-effort extraction, tolerant of a few reasonable key-name
    variants - the sandbox's integration guide describes what the
    catalogue contains ("id, location, codec, live status, stream
    properties, and all three URLs") but not its exact JSON shape, so
    this can't be pinned down without a real sample response."""

    cam_id = entry.get("id") or entry.get("camera_id") or entry.get("stream_id")
    if cam_id is None:
        raise ValueError(f"catalogue entry has no id field: {entry!r}")

    location = (
        entry.get("location") or entry.get("location_name") or f"Sandbox camera {cam_id}"
    )
    name = entry.get("name") or f"Sandbox {cam_id}"

    rtsp_url = None
    urls = entry.get("urls") or entry.get("streams")
    if isinstance(urls, dict):
        rtsp_url = urls.get("rtsp")
    rtsp_url = rtsp_url or entry.get("rtsp") or entry.get("rtsp_url")

    if not rtsp_url:
        # Fall back to the documented URL pattern only if the entry
        # itself doesn't carry an explicit RTSP URL - the guide states
        # this pattern directly, even while warning not to assume it
        # holds indefinitely ("the catalogue is the contract, the URL
        # pattern is not").
        rtsp_url = f"rtsp://{config.SANDBOX_HOST}:8554/stream/{cam_id}"

    live = entry.get("live")
    if live is None:
        live = entry.get("status") or entry.get("live_status")
    is_live = True if live is None else str(live).lower() not in ("false", "0", "offline", "down", "dead")

    return cam_id, name, location, rtsp_url, is_live


@app.post("/api/catalogue/sync")
def sync_sandbox_catalogue():
    """Pull the current camera list from the sandbox's own catalogue
    endpoint and onboard whatever it lists that isn't already
    onboarded. Per the sandbox's integration guide: "the catalogue is
    the contract, the URL pattern is not" - camera ids and the set of
    available cameras can change, so this is read live at sync time
    rather than a fixed list baked into config.py."""

    if not config.SANDBOX_HOST:
        raise HTTPException(
            status_code=400,
            detail="config.SANDBOX_HOST is not set - fill in the sandbox host first"
        )

    ingest_url = f"http://{config.SANDBOX_HOST}{config.SANDBOX_INGEST_PATH}"
    try:
        resp = requests.get(ingest_url, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach catalogue at {ingest_url}: {exc}")
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"Catalogue response wasn't valid JSON: {exc}")

    entries = payload if isinstance(payload, list) else payload.get("cameras", payload.get("streams", []))
    if not isinstance(entries, list):
        raise HTTPException(status_code=502, detail="Unrecognized catalogue response shape")

    added, skipped, not_live, errors = [], [], [], []

    for entry in entries:
        try:
            cam_id, name, location, rtsp_url, is_live = _parse_catalogue_entry(entry)
        except ValueError as exc:
            errors.append(str(exc))
            continue

        camera_id = f"SANDBOX-{cam_id}"
        if camera_id in workers:
            skipped.append(camera_id)
            continue
        if not is_live:
            # "Pace your load: open only the cameras you are actively
            # processing" - no point opening a capture for a feed the
            # catalogue itself says isn't live right now.
            not_live.append(camera_id)
            continue

        cam_cfg = {
            "camera_id": camera_id,
            "name": name,
            "department": "Municipal",
            "location": location,
            "lat": 23.03,
            "lon": 72.58,
            "video_path": rtsp_url,
            "start_frame": 0,
        }
        _start_worker(cam_cfg, "RTSP")
        added.append(camera_id)

    print(f"[catalogue sync] {len(added)} added, {len(skipped)} already onboarded, "
          f"{len(not_live)} not live, {len(errors)} unparseable")

    return {
        "total_in_catalogue": len(entries),
        "added": added,
        "already_onboarded": skipped,
        "not_live": not_live,
        "errors": errors,
    }


@app.get("/api/cameras/{camera_id}")
def get_camera(camera_id: str):
    """Full connection details for one camera, including credentials -
    kept out of the bulk list above so a password doesn't ride along
    on every poll; only fetched on demand when an operator opens the
    edit panel for that specific feed."""

    worker = workers.get(camera_id)
    if worker is None:
        raise HTTPException(status_code=404, detail="Unknown camera")

    cfg = worker.camera_cfg
    return {
        "camera_id": camera_id,
        "name": cfg.get("name", camera_id),
        "department": cfg.get("department", "Police"),
        "location": cfg.get("location", ""),
        "lat": cfg.get("lat", 23.03),
        "lon": cfg.get("lon", 72.58),
        "protocol": camera_protocols.get(camera_id, "Unknown"),
        "video_path": cfg.get("video_path"),
    }


@app.put("/api/cameras/{camera_id}")
def update_camera(camera_id: str, req: UpdateCameraRequest):
    camera_id = camera_id.strip().upper()
    old_worker = workers.get(camera_id)
    if old_worker is None:
        raise HTTPException(status_code=404, detail="Unknown camera")

    old_cfg = old_worker.camera_cfg
    new_cfg = {
        "camera_id": camera_id,
        "name": req.name if req.name is not None else old_cfg.get("name", camera_id),
        "department": req.department if req.department is not None else old_cfg.get("department", "Police"),
        "location": req.location if req.location is not None else old_cfg.get("location", ""),
        "lat": req.lat if req.lat is not None else old_cfg.get("lat", 23.03),
        "lon": req.lon if req.lon is not None else old_cfg.get("lon", 72.58),
        "video_path": req.video_path if req.video_path is not None else old_cfg.get("video_path"),
        "start_frame": old_cfg.get("start_frame", 0),
    }
    if "imgsz" in old_cfg:
        new_cfg["imgsz"] = old_cfg["imgsz"]

    protocol_label = req.protocol or camera_protocols.get(camera_id, "Unknown")

    # Fully release the old capture before opening the replacement -
    # some sources (this project has hit this firsthand against a real
    # NVR) refuse a second simultaneous connection to the same channel.
    _stop_worker(camera_id)
    worker = _start_worker(new_cfg, protocol_label)

    print(f"[{camera_id}] reconfigured live via edit panel - "
          f"protocol={protocol_label} source={new_cfg['video_path']!r}")

    return {"status": "ok", "camera_id": worker.camera_id}


@app.delete("/api/cameras/{camera_id}")
def remove_camera(camera_id: str):
    camera_id = camera_id.strip().upper()

    # A camera can exist in the DB with no live worker - e.g. one added
    # at runtime via the wizard, then orphaned by a server restart
    # (the wizard writes straight to the DB + in-memory workers, not to
    # config.py, so it doesn't come back on its own). Still deletable.
    _stop_worker(camera_id)
    camera_protocols.pop(camera_id, None)

    conn = db.connect()
    try:
        existed = db.delete_camera(conn, camera_id)
    finally:
        conn.close()

    if not existed:
        raise HTTPException(status_code=404, detail="Unknown camera")

    return {"status": "ok"}


@app.post("/api/admin/clear-data")
def clear_all_data():
    """Wipes every stored detection, alert, and evidence photo -
    watchlist entries and onboarded cameras are left untouched. Also
    resets each camera worker's in-memory dedup/display state, so
    nothing stale (a dedup entry pointing at a detection_id that no
    longer exists) lingers after the underlying rows are gone."""

    conn = db.connect()
    try:
        detections_deleted, alerts_deleted = db.clear_all_detections(conn)
    finally:
        conn.close()

    evidence_deleted = 0
    for f in config.EVIDENCE_DIR.glob("*.jpg"):
        f.unlink(missing_ok=True)
        evidence_deleted += 1

    for worker in workers.values():
        with worker.lock:
            worker.recent_detections.clear()
            worker.active_alerts.clear()
        worker._last_logged.clear()

    print(f"[admin] Cleared all data - {detections_deleted} detections, "
          f"{alerts_deleted} alerts, {evidence_deleted} evidence photos")

    return {
        "status": "ok",
        "detections_deleted": detections_deleted,
        "alerts_deleted": alerts_deleted,
        "evidence_deleted": evidence_deleted,
    }


@app.get("/api/cameras/{camera_id}/snapshot.jpg")
def get_snapshot(camera_id: str):
    worker = workers.get(camera_id)
    if worker is None:
        raise HTTPException(status_code=404, detail="Unknown camera")

    frame, _, _, _ = worker.snapshot()
    if frame is None:
        raise HTTPException(status_code=503, detail="No frame yet")

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise HTTPException(status_code=500, detail="Encode failed")

    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/api/cameras/{camera_id}/stream.mjpg")
def stream_camera(camera_id: str):
    """Continuous multipart/x-mixed-replace push instead of the
    dashboard polling snapshot.jpg on a timer - the classic MJPEG-over-
    HTTP pattern (same one Flask/FastAPI camera-streaming tutorials all
    use), so the browser repaints every new frame as it's produced
    instead of being capped at whatever interval a JS timer polls on."""

    worker = workers.get(camera_id)
    if worker is None:
        raise HTTPException(status_code=404, detail="Unknown camera")

    async def gen():
        last_count = -1
        try:
            while True:
                frame, _, _, _ = worker.snapshot()
                count = worker.frame_count
                if frame is not None and count != last_count:
                    last_count = count
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n"
                            + buf.tobytes() + b"\r\n"
                        )
                # Cheap in-memory check, not an I/O wait - safe to poll
                # fast; the (comparatively expensive) encode+send above
                # only happens when a genuinely new frame is ready.
                await asyncio.sleep(0.02)
        except asyncio.CancelledError:
            pass

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/search")
def search(plate: str = Query(..., min_length=1)):
    plate = plate.strip().upper()
    conn = db.connect()
    try:
        history = [_serialize(row) for row in db.search_plate(conn, plate)]
        watch_hit = db.check_watchlist(conn, plate)
        return {
            "plate": plate,
            "history": history,
            "watchlist": _serialize(watch_hit) if watch_hit else None,
        }
    finally:
        conn.close()


@app.get("/api/detections/recent")
def get_recent_detections(limit: int = 25):
    conn = db.connect()
    try:
        return [_serialize(row) for row in db.recent_detections(conn, limit)]
    finally:
        conn.close()


@app.get("/api/detections/log")
def get_detections_log(
    page: int = 1,
    page_size: int = 100,
    keyword: str = "",
    date_from: str = "",
    date_to: str = "",
    camera_id: str = "",
    status: str = "",
):
    conn = db.connect()
    try:
        rows, total = db.list_detections_log(
            conn, page=page, page_size=page_size,
            keyword=keyword or None,
            date_from=date_from or None,
            date_to=date_to or None,
            camera_id=camera_id or None,
            status=status or None,
        )
        records = [_serialize(row) for row in rows]
    finally:
        conn.close()

    for row in records:
        row["protocol"] = camera_protocols.get(row["camera_id"], "Unknown")

    return {"records": records, "total": total, "page": page, "page_size": page_size}


@app.get("/api/alerts/recent")
def get_recent_alerts(limit: int = 25):
    conn = db.connect()
    try:
        return [_serialize(row) for row in db.recent_alerts(conn, limit)]
    finally:
        conn.close()


config.EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/evidence", StaticFiles(directory=str(config.EVIDENCE_DIR)), name="evidence")

# The dashboard is a single-page app that does its own client-side
# routing (History API) once loaded. Each of these paths just needs
# to serve the same index.html on a hard load/refresh/bookmark - the
# page's own JS then shows the matching section from the URL.
SPA_PAGES = ["/multiview", "/integration", "/search", "/map", "/alerts", "/cameras"]


def _serve_spa():
    return FileResponse(str(WEB_DIR / "index.html"))


for _path in SPA_PAGES:
    app.add_api_route(_path, _serve_spa, methods=["GET"], include_in_schema=False)

# Serve the dashboard last, so it doesn't shadow the routes above.
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="static")
