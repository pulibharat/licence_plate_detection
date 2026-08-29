"""Search a plate number across all camera detections and show its
route + any watchlist match.

Usage: python search.py <PLATE_NUMBER>
"""

import sys

import db


def run(plate_number):
    plate_number = plate_number.strip().upper()
    conn = db.connect()

    print(f"\nSEARCH: {plate_number}\n")

    history = db.search_plate(conn, plate_number)

    if not history:
        print("No detections found for this plate.")
        conn.close()
        return

    print("DETECTION HISTORY")
    print("-" * 60)
    for row in history:
        print(
            f"{row['detected_at']:%Y-%m-%d %H:%M:%S}  "
            f"{row['camera_id']} ({row['department']})  "
            f"{row['location_name']}"
        )

    print("\nROUTE (camera -> camera)")
    print("-" * 60)
    route = " -> ".join(
        f"{row['camera_id']} [{row['lat']:.4f}, {row['lon']:.4f}]"
        for row in history
    )
    print(route)

    watch_hit = db.check_watchlist(conn, plate_number)
    print()
    if watch_hit:
        print("!! HIGH PRIORITY ALERT !!")
        print(f"Watchlist match: {watch_hit['category']} "
              f"({watch_hit['priority']})")
        print(f"Description: {watch_hit['description']}")
        last = history[-1]
        print(f"Last seen: {last['camera_id']} at {last['detected_at']:%Y-%m-%d %H:%M:%S}")
    else:
        print("Watchlist match: none")

    conn.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python search.py <PLATE_NUMBER>")
        sys.exit(1)

    run(sys.argv[1])
