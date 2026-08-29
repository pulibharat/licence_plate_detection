import math

import cv2
import numpy as np

from config import PANEL_WIDTH, MAX_PANEL_ROWS

TILE_BORDER_ONLINE = (0, 255, 0)
TILE_BORDER_OFFLINE = (0, 0, 255)
TILE_BG = (40, 40, 40)

WINDOW_NAME = "License Plate Recognition"

PANEL_BG = (24, 24, 24)
HEADER_COLOR = (0, 255, 0)
DIVIDER_COLOR = (80, 80, 80)
CAR_ID_COLOR = (0, 200, 255)
PLATE_COLOR = (255, 255, 255)
META_COLOR = (150, 150, 150)

ALERT_BG = (20, 20, 90)
ALERT_HEADER_COLOR = (0, 0, 255)
ALERT_TEXT_COLOR = (200, 200, 255)
MAX_ALERT_ROWS = 3


def draw_detection_box(frame, x1, y1, x2, y2, label=None):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if not label:
        return

    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )


def _draw_alerts(panel, active_alerts):
    """Draw a red high-priority alert banner at the top of the panel.
    Returns the y-coordinate where normal content can start."""

    if not active_alerts:
        return 0

    shown = list(reversed(active_alerts[-MAX_ALERT_ROWS:]))
    banner_height = 34 + len(shown) * 46 + 12

    cv2.rectangle(panel, (0, 0), (PANEL_WIDTH, banner_height), ALERT_BG, -1)
    cv2.putText(
        panel, "!! HIGH PRIORITY ALERT !!", (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.62, ALERT_HEADER_COLOR, 2
    )

    y = 56
    for alert in shown:
        camera_tag = f"[{alert['camera_id']}] " if alert.get("camera_id") else ""
        cv2.putText(
            panel,
            f"{camera_tag}{alert['plate']}  ({alert['priority'].upper()})",
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, ALERT_HEADER_COLOR, 2
        )
        cv2.putText(
            panel,
            f"{alert['category']}  {alert['timestamp']}",
            (16, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, ALERT_TEXT_COLOR, 1
        )
        y += 46

    return banner_height


def _build_panel(height, recent_detections, active_alerts=None):
    panel = np.full((height, PANEL_WIDTH, 3), PANEL_BG, dtype=np.uint8)

    top = _draw_alerts(panel, active_alerts or [])

    header_y = top + 26
    cv2.putText(
        panel, "DETECTED PLATES", (16, header_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, HEADER_COLOR, 2
    )
    cv2.line(
        panel, (16, header_y + 12), (PANEL_WIDTH - 16, header_y + 12),
        DIVIDER_COLOR, 1
    )

    y = header_y + 48
    row_height = 54

    for entry in reversed(recent_detections[-MAX_PANEL_ROWS:]):

        camera_tag = f"[{entry['camera_id']}] " if entry.get("camera_id") else ""
        cv2.putText(
            panel,
            f"{camera_tag}Car {entry['car_id']}  {entry['plate']}",
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, CAR_ID_COLOR, 2
        )
        cv2.putText(
            panel,
            f"{entry['timestamp']}  conf {entry['weight']:.2f}",
            (16, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, META_COLOR, 1
        )

        y += row_height
        if y > height - 30:
            break

    return panel


def show(frame, recent_detections, active_alerts=None):
    """Compose the CCTV frame and the detected-plates panel side by
    side and display them in a single window."""

    panel = _build_panel(frame.shape[0], recent_detections, active_alerts)
    dashboard = cv2.hconcat([frame, panel])
    cv2.imshow(WINDOW_NAME, dashboard)


def _build_grid(tiles, tile_width=640):
    """tiles: list of (label, frame_or_None, status). Resizes each
    to tile_width and arranges them in a roughly square grid."""

    rendered = []
    tile_height = None

    for label, frame, status in tiles:
        if frame is None:
            height = tile_height or int(tile_width * 9 / 16)
            tile = np.full((height, tile_width, 3), TILE_BG, dtype=np.uint8)
        else:
            h, w = frame.shape[:2]
            height = int(tile_width * h / w)
            tile = cv2.resize(frame, (tile_width, height))

        tile_height = tile_height or tile.shape[0]

        border = TILE_BORDER_ONLINE if status == "online" else TILE_BORDER_OFFLINE
        cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), border, 3)
        cv2.putText(
            tile, f"{label} [{status}]", (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
        )
        rendered.append(tile)

    if not rendered:
        return np.full((int(tile_width * 9 / 16), tile_width, 3), TILE_BG, dtype=np.uint8)

    cols = math.ceil(math.sqrt(len(rendered)))
    rows = math.ceil(len(rendered) / cols)

    blank = np.full(rendered[0].shape, (0, 0, 0), dtype=np.uint8)
    while len(rendered) < rows * cols:
        rendered.append(blank.copy())

    row_images = [
        cv2.hconcat(rendered[r * cols:(r + 1) * cols])
        for r in range(rows)
    ]
    return cv2.vconcat(row_images)


def show_multi(camera_snapshots, tile_width=640):
    """Compose a grid of camera feeds plus a combined side panel of
    detections/alerts merged across all cameras.

    camera_snapshots: list of (camera_id, frame, recent_detections,
    active_alerts, status) tuples, as returned by
    CameraWorker.snapshot().
    """

    tiles = [
        (camera_id, frame, status)
        for camera_id, frame, _, _, status in camera_snapshots
    ]
    grid = _build_grid(tiles, tile_width=tile_width)

    all_detections = []
    all_alerts = []
    for _, _, detections, alerts, _ in camera_snapshots:
        all_detections.extend(detections)
        all_alerts.extend(alerts)

    all_detections.sort(key=lambda d: d["timestamp"])
    all_alerts.sort(key=lambda a: a["timestamp"])

    panel = _build_panel(grid.shape[0], all_detections, all_alerts)
    combined = cv2.hconcat([grid, panel])
    cv2.imshow(WINDOW_NAME, combined)
