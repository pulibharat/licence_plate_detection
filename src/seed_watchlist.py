"""One-off script to seed demo watchlist entries.

Run manually: python seed_watchlist.py
"""

import db

# (plate_number, category, description, priority)
DEMO_WATCHLIST = [
    ("34A36837", "stolen_vehicle", "Reported stolen - demo entry", "high"),
    ("34A23126", "suspicious", "Flagged for surveillance - demo entry", "medium"),
]

if __name__ == "__main__":
    conn = db.connect()
    db.init_schema(conn)
    db.seed_watchlist(conn, DEMO_WATCHLIST)
    conn.close()
    print(f"Seeded {len(DEMO_WATCHLIST)} watchlist entries.")
