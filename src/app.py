import os
import threading

import cv2
import torch

import config
import dashboard
from camera_worker import CameraWorker

# =========================
# GPU
# =========================

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    DEVICE = 0
else:
    print("Using CPU")
    DEVICE = "cpu"

# =========================
# LOG FILE
# =========================

if not os.path.exists(config.LOG_FILE):
    with open(config.LOG_FILE, "w", encoding="utf-8") as f:
        f.write(
            "timestamp | camera | car_id | track_id | plate | detection | ocr_weight\n"
        )

# =========================
# START ONE THREAD PER CAMERA
# =========================

workers = [CameraWorker(cam_cfg, DEVICE) for cam_cfg in config.CAMERAS]
threads = [threading.Thread(target=w.run, daemon=True) for w in workers]

for t in threads:
    t.start()

print(f"Started {len(workers)} camera worker(s): "
      f"{', '.join(w.camera_id for w in workers)}")

# =========================
# MAIN LOOP: DRAW COMBINED DASHBOARD
# =========================

try:
    while any(t.is_alive() for t in threads):
        snapshots = []
        for w in workers:
            frame, recent_detections, active_alerts, status = w.snapshot()
            snapshots.append((w.camera_id, frame, recent_detections, active_alerts, status))

        dashboard.show_multi(snapshots)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
finally:
    for w in workers:
        w.stop()
    for t in threads:
        t.join(timeout=5)
    cv2.destroyAllWindows()

print("\nFinished.")
print("Saved:", config.LOG_FILE)
