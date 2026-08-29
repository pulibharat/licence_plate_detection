#!/bin/bash
# Starts everything needed to use Reticle against this project:
# Postgres (Docker) -> FastAPI backend (:8000) -> Vite/Reticle mirror (:5174).
# Run this once per day, then open http://localhost:5174/ - the Reticle
# daemon and the browser SDK connect on their own after that.
set -e

docker start licence_plate_pg 2>/dev/null || echo "licence_plate_pg already running or missing"

cd /c/licence_plate/src
nohup /c/licence_plate/env/Scripts/python.exe -m uvicorn web_api:app --host 127.0.0.1 --port 8000 > /tmp/uvicorn.log 2>&1 &
disown

cd /c/licence_plate/src/web
nohup npx vite --port 5174 > /tmp/vite.log 2>&1 &
disown

sleep 4
echo "backend  (:8000): $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/api/stats)"
echo "reticle  (:5174): $(curl -s -o /dev/null -w '%{http_code}' http://localhost:5174/)"
echo "Now open http://localhost:5174/ in a browser - Reticle connects automatically."
