# SENTINEL — CCTV License-Plate Intelligence Platform

A CCTV/ANPR dashboard built for a police CCTV integration hackathon. It pulls in live
camera feeds (RTSP, HTTP/MJPEG, or a local video file), runs them through a computer
vision pipeline (vehicle detection → tracking → plate detection → OCR), and gives an
operator a live map, a searchable plate history, and watchlist alerts across every
onboarded camera.

## How it works

- **Camera ingest** (`src/camera_worker.py`) — one worker per camera, reading frames
  via OpenCV from whatever source is configured (RTSP URL, HTTP stream, or a local
  file for demo/testing).
- **Vehicle + plate detection** — a YOLOv8 model detects and classifies vehicles
  (car/truck/bus/motorcycle), a second model locates the plate, and OCR reads it.
  Each vehicle is tracked across frames so it's only logged once, not once per frame.
- **Storage** — Postgres + PostGIS, so camera locations and plate sightings are
  geospatial from the start (used for the movement-history map/route).
- **API + dashboard** (`src/web_api.py`, `src/web/index.html`) — a FastAPI backend
  serving both the JSON API and the (plain HTML/JS, no build step) dashboard: live
  feed grid, plate search, movement history, and watchlist alerts.

## Prerequisites

- Python 3.13
- Docker Desktop (for Postgres/PostGIS — no local Postgres install needed)
- A CUDA-capable GPU is strongly recommended (the detection pipeline runs on
  every camera's frames continuously); it'll run on CPU but noticeably slower.

## Setup

1. **Start the database**
   ```
   docker compose up -d
   ```
   This pulls the standard public `postgis/postgis:16-3.4` image — nothing to build.
   The app creates its own tables on first run (`src/db.py: init_schema`), so no
   separate migration step is needed.

   > If this fails with a container name conflict, you (or someone) already has a
   > `licence_plate_pg` container from before this `docker-compose.yml` existed.
   > Either `docker rm -f licence_plate_pg` (drops its data) and rerun, or just
   > `docker start licence_plate_pg` to reuse it as-is.

2. **Create a virtualenv and install dependencies**
   ```
   python -m venv env
   env\Scripts\activate          # Windows
   pip install -r requirements.txt
   ```
   Then install PyTorch separately — it's a CUDA build, not on PyPI's default index:
   ```
   pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
   ```
   No CUDA GPU? Drop the `--index-url` flag to get the CPU-only build instead.

3. **Run the app**
   ```
   cd src
   python -m uvicorn web_api:app --host 127.0.0.1 --port 8000
   ```
   Open **http://127.0.0.1:8000/**.

## Camera setup notes

Cameras are configured in `src/config.py`. Three ship configured out of the box:

| Camera | Source | Works out of the box? |
|---|---|---|
| CAM-01 | `data/cars_extended.mp4` (local demo video) | Yes |
| CAM-03 | `data/locatiopn1.mp4` | **No** — excluded from git (1.9GB). Supply your own video at that path, or repoint `config.py` at something smaller. |
| CAM-04 | RTSP, real local NVR | **No** — credentials are redacted in `config.py`. Fill in `<user>`/`<password>` locally to use it; this points at a specific device on the original network, not a public stream. |

Add a real camera by appending to `CAMERAS` in `config.py` — `video_path` accepts an
RTSP URL, an HTTP/MJPEG URL, a local file path, or a USB device index. See the
comments above the `CAMERAS` list for details.

## Notes

- DB credentials in `docker-compose.yml` and `config.py` are dev-only defaults —
  fine for local/demo use, not meant for a real deployment.
- `data/evidence/` (snapshot images) and `logs/` are generated at runtime and are
  git-ignored.
