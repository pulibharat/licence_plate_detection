import psycopg2
import psycopg2.extras

import config

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS cameras (
    camera_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT,
    location_name TEXT,
    geom GEOGRAPHY(Point, 4326),
    stream_status TEXT DEFAULT 'offline'
);

CREATE TABLE IF NOT EXISTS vehicle_detections (
    id SERIAL PRIMARY KEY,
    plate_number TEXT NOT NULL,
    camera_id TEXT REFERENCES cameras(camera_id),
    detected_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    detection_confidence REAL,
    ocr_weight REAL,
    snapshot_path TEXT,
    vehicle_type TEXT DEFAULT 'vehicle'
);
ALTER TABLE vehicle_detections ADD COLUMN IF NOT EXISTS snapshot_path TEXT;
ALTER TABLE vehicle_detections ADD COLUMN IF NOT EXISTS vehicle_type TEXT DEFAULT 'vehicle';
-- 'confirmed': OCR consensus cleared the confidence bar - a real,
-- searchable plate number. 'unclear': a vehicle genuinely passed and
-- was captured, but OCR never got a confident enough reading to
-- trust the plate text - kept separate so a police reviewer can still
-- see "something passed here" (with its photo) without it polluting
-- plate-number search results with guesses.
ALTER TABLE vehicle_detections ADD COLUMN IF NOT EXISTS plate_status TEXT DEFAULT 'confirmed';
CREATE INDEX IF NOT EXISTS idx_detections_plate ON vehicle_detections(plate_number);
CREATE INDEX IF NOT EXISTS idx_detections_time ON vehicle_detections(detected_at);

CREATE TABLE IF NOT EXISTS watchlist (
    plate_number TEXT PRIMARY KEY,
    category TEXT,
    description TEXT,
    priority TEXT DEFAULT 'medium',
    added_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    plate_number TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    camera_id TEXT REFERENCES cameras(camera_id),
    detection_id INTEGER REFERENCES vehicle_detections(id),
    detected_at TIMESTAMPTZ NOT NULL,
    status TEXT DEFAULT 'new'
);
"""


def connect():
    return psycopg2.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
    )


def init_schema(conn):
    with conn.cursor() as cur:
        cur.execute(SCHEMA)
    conn.commit()


def upsert_camera(conn, camera_id, name, department, location_name, lat, lon):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO cameras
                (camera_id, name, department, location_name, geom, stream_status)
            VALUES (
                %s, %s, %s, %s,
                ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography,
                'online'
            )
            ON CONFLICT (camera_id) DO UPDATE SET
                name = EXCLUDED.name,
                department = EXCLUDED.department,
                location_name = EXCLUDED.location_name,
                geom = EXCLUDED.geom,
                stream_status = 'online'
            """,
            (camera_id, name, department, location_name, lon, lat)
        )
    conn.commit()


def insert_detection(
    conn, plate_number, camera_id, detected_at,
    detection_confidence, ocr_weight, snapshot_path=None, vehicle_type="vehicle",
    plate_status="confirmed"
):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO vehicle_detections
                (plate_number, camera_id, detected_at,
                 detection_confidence, ocr_weight, snapshot_path, vehicle_type, plate_status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (plate_number, camera_id, detected_at,
             detection_confidence, ocr_weight, snapshot_path, vehicle_type, plate_status)
        )
        detection_id = cur.fetchone()[0]
    conn.commit()
    return detection_id


def update_detection(
    conn, detection_id, plate_number, detected_at,
    detection_confidence, ocr_weight, snapshot_path, vehicle_type, plate_status
):
    """Replaces an existing detection row in place - used when a
    better (higher-scoring, or confirmed-over-unclear) reading of the
    same recent sighting comes in, instead of inserting a duplicate."""

    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE vehicle_detections
            SET plate_number = %s, detected_at = %s, detection_confidence = %s,
                ocr_weight = %s, snapshot_path = %s, vehicle_type = %s, plate_status = %s
            WHERE id = %s
            """,
            (plate_number, detected_at, detection_confidence, ocr_weight,
             snapshot_path, vehicle_type, plate_status, detection_id)
        )
    conn.commit()


def search_plate(conn, plate_number):
    """Full cross-camera detection history for a plate, oldest first."""

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                d.id, d.plate_number, d.detected_at,
                d.detection_confidence, d.ocr_weight, d.snapshot_path, d.vehicle_type,
                c.camera_id, c.name AS camera_name,
                c.department, c.location_name,
                ST_Y(c.geom::geometry) AS lat,
                ST_X(c.geom::geometry) AS lon
            FROM vehicle_detections d
            JOIN cameras c ON c.camera_id = d.camera_id
            WHERE d.plate_number = %s
            ORDER BY d.detected_at ASC
            """,
            (plate_number,)
        )
        return cur.fetchall()


def check_watchlist(conn, plate_number):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT * FROM watchlist WHERE plate_number = %s",
            (plate_number,)
        )
        return cur.fetchone()


def create_alert(conn, plate_number, alert_type, camera_id, detection_id, detected_at):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO alerts
                (plate_number, alert_type, camera_id, detection_id, detected_at)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (plate_number, alert_type, camera_id, detection_id, detected_at)
        )
        alert_id = cur.fetchone()[0]
    conn.commit()
    return alert_id


def get_stats(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM cameras")
        total_cameras = cur.fetchone()[0]

        cur.execute("SELECT count(*) FROM cameras WHERE stream_status = 'online'")
        online_cameras = cur.fetchone()[0]

        cur.execute(
            "SELECT count(*) FROM vehicle_detections "
            "WHERE detected_at::date = now()::date"
        )
        detections_today = cur.fetchone()[0]

        cur.execute("SELECT count(*) FROM alerts WHERE status = 'new'")
        active_alerts = cur.fetchone()[0]

        cur.execute("SELECT count(*) FROM watchlist")
        watchlist_records = cur.fetchone()[0]

    return {
        "total_cameras": total_cameras,
        "online_cameras": online_cameras,
        "detections_today": detections_today,
        "active_alerts": active_alerts,
        "watchlist_records": watchlist_records,
    }


def update_camera_status(conn, camera_id, status):
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE cameras SET stream_status = %s WHERE camera_id = %s",
            (status, camera_id)
        )
    conn.commit()


def delete_camera(conn, camera_id):
    """Returns True if a camera row actually existed and was removed."""

    with conn.cursor() as cur:
        cur.execute("DELETE FROM alerts WHERE camera_id = %s", (camera_id,))
        cur.execute("DELETE FROM vehicle_detections WHERE camera_id = %s", (camera_id,))
        cur.execute("DELETE FROM cameras WHERE camera_id = %s", (camera_id,))
        existed = cur.rowcount > 0
    conn.commit()
    return existed


def clear_all_detections(conn):
    """Wipes every detection and alert - across every camera. The
    watchlist and the cameras themselves are left untouched; only the
    accumulated history/evidence trail is reset. Returns
    (detections_deleted, alerts_deleted)."""

    with conn.cursor() as cur:
        cur.execute("DELETE FROM alerts")
        alerts_deleted = cur.rowcount
        cur.execute("DELETE FROM vehicle_detections")
        detections_deleted = cur.rowcount
    conn.commit()
    return detections_deleted, alerts_deleted


def list_cameras(conn):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                camera_id, name, department, location_name, stream_status,
                ST_Y(geom::geometry) AS lat, ST_X(geom::geometry) AS lon
            FROM cameras
            ORDER BY camera_id
            """
        )
        return cur.fetchall()


def recent_detections(conn, limit=25):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                d.id, d.plate_number, d.detected_at,
                d.detection_confidence, d.ocr_weight, d.snapshot_path, d.vehicle_type,
                c.camera_id, c.name AS camera_name
            FROM vehicle_detections d
            JOIN cameras c ON c.camera_id = d.camera_id
            ORDER BY d.detected_at DESC
            LIMIT %s
            """,
            (limit,)
        )
        return cur.fetchall()


def list_detections_log(conn, page=1, page_size=100, keyword=None, date_from=None, date_to=None, camera_id=None, status=None):
    """Paginated, filterable detection log - the data behind the
    Detection Log screen. Returns (rows, total_matching_count)."""

    page = max(1, page)
    page_size = max(1, min(page_size, 500))
    offset = (page - 1) * page_size

    where = []
    params = []
    if keyword:
        where.append("d.plate_number ILIKE %s")
        params.append(f"%{keyword}%")
    if camera_id:
        where.append("d.camera_id = %s")
        params.append(camera_id)
    if date_from:
        where.append("d.detected_at >= %s")
        params.append(date_from)
    if date_to:
        where.append("d.detected_at <= %s")
        params.append(date_to)
    if status:
        where.append("d.plate_status = %s")
        params.append(status)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            f"""
            SELECT COUNT(*) AS n
            FROM vehicle_detections d
            JOIN cameras c ON c.camera_id = d.camera_id
            {where_sql}
            """,
            params
        )
        total = cur.fetchone()["n"]

        cur.execute(
            f"""
            SELECT
                d.id, d.plate_number, d.detected_at,
                d.detection_confidence, d.ocr_weight, d.snapshot_path, d.vehicle_type,
                d.plate_status,
                c.camera_id, c.name AS camera_name, c.department, c.location_name,
                (w.plate_number IS NOT NULL) AS watchlisted
            FROM vehicle_detections d
            JOIN cameras c ON c.camera_id = d.camera_id
            LEFT JOIN watchlist w ON w.plate_number = d.plate_number
            {where_sql}
            ORDER BY d.detected_at DESC
            LIMIT %s OFFSET %s
            """,
            params + [page_size, offset]
        )
        rows = cur.fetchall()

    return rows, total


def recent_alerts(conn, limit=25):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                a.id, a.plate_number, a.alert_type, a.status, a.detected_at,
                a.camera_id, c.name AS camera_name,
                w.priority, w.description
            FROM alerts a
            JOIN cameras c ON c.camera_id = a.camera_id
            LEFT JOIN watchlist w ON w.plate_number = a.plate_number
            ORDER BY a.detected_at DESC
            LIMIT %s
            """,
            (limit,)
        )
        return cur.fetchall()


def seed_watchlist(conn, entries):
    """entries: iterable of (plate_number, category, description, priority)."""

    with conn.cursor() as cur:
        for plate_number, category, description, priority in entries:
            cur.execute(
                """
                INSERT INTO watchlist
                    (plate_number, category, description, priority)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (plate_number) DO NOTHING
                """,
                (plate_number, category, description, priority)
            )
    conn.commit()
