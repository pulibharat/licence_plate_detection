from collections import Counter, defaultdict
from typing import NamedTuple, Optional

import config


class PlateUpdate(NamedTuple):
    car_id: Optional[int]
    plate_text: str
    weight: float
    vehicle_type: str
    # False: OCR consensus cleared the confidence bar - a real,
    # trustworthy plate reading. True: a vehicle genuinely passed and
    # was tracked, but no reading (or none of its readings) ever got
    # confident/repeated enough to trust - still worth a record (with
    # its photo) for human review, just not as a confirmed plate.
    unclear: bool = False
    # Last known (x1, y1, x2, y2) box for this track, or None if it
    # never produced one. Lets the caller judge whether a *different*
    # track is plausibly a continuation of the same physical vehicle
    # (e.g. re-acquired closer to the camera) by position, not just by
    # matching text or timing - see CameraWorker._log_update.
    last_box: Optional[tuple] = None


class PlateTracker:
    """Turns noisy per-frame OCR readings into one stable plate
    reading per tracked vehicle.

    Readings are accumulated silently while a vehicle is in frame.
    A track is only ever logged once - when it leaves the frame (or
    the video ends) - using whichever plate text collected the most
    confidence-weighted votes over the track's whole lifetime. This
    avoids re-logging every time the leading guess flips mid-track.

    Every finalized track produces a record - a vehicle really did
    pass, so it's worth keeping even when OCR never got confident
    enough to trust. Only the "confirmed" bar (see _is_well_supported)
    decides whether that record carries a trusted plate number or
    comes back marked unclear instead of being dropped outright.

    Deliberately does not dedupe the same plate across different
    track_ids itself (a brief tracking dropout splitting one physical
    vehicle into several tracks, or the same plate genuinely passing
    again much later, look identical from in here). That needs a
    notion of *time* and *position* this class doesn't have - it's
    handled one layer up, in CameraWorker._log_update, using each
    update's last_box alongside how recently the same plate/type was
    last actually logged.
    """

    def __init__(self):
        self.next_car_id = 1
        self.plate_to_car_id = {}
        self.track_readings = defaultdict(list)
        self.track_vehicle_types = defaultdict(Counter)
        self.track_last_box = {}
        self.active_track_ids = set()
        self.finalized_track_ids = set()

    def record(self, track_id, plate_text, ocr_confidence, vehicle_type=None, box=None):
        """Record a raw OCR reading for a track, if it's usable.
        vehicle_type (car/truck/bus/motorcycle/...) is voted on
        separately from the plate text - a single classifier miss on
        one frame shouldn't flip the whole track's vehicle type. box
        is remembered regardless of whether the reading itself was
        usable, so a track that only ever produced weak/garbled OCR
        still leaves behind where it last was."""

        if track_id is None:
            return
        if box is not None:
            self.track_last_box[track_id] = box
        if vehicle_type:
            self.track_vehicle_types[track_id][vehicle_type] += 1
        if len(plate_text) < 5:
            return
        if ocr_confidence < config.MIN_VOTE_CONFIDENCE:
            return
        if track_id in self.finalized_track_ids:
            return

        self.track_readings[track_id].append((plate_text, ocr_confidence))

    def _best_vehicle_type(self, track_id):
        votes = self.track_vehicle_types.get(track_id)
        if not votes:
            return "vehicle"
        return votes.most_common(1)[0][0]

    def _consensus(self, track_id):
        readings = self.track_readings.get(track_id)
        if not readings:
            return None

        weight = Counter()
        count = Counter()

        for text, conf in readings:
            weight[text] += conf
            count[text] += 1

        best_text, best_weight = weight.most_common(1)[0]
        return best_text, best_weight, count[best_text]

    def _assign_car_id(self, plate_text):
        if plate_text not in self.plate_to_car_id:
            self.plate_to_car_id[plate_text] = self.next_car_id
            self.next_car_id += 1
        return self.plate_to_car_id[plate_text]

    def best_known(self, track_id) -> Optional[PlateUpdate]:
        """Best consensus so far for a track, for live display. None
        if nothing well-supported has been read yet - callers should
        avoid showing a label in that case rather than guessing."""

        consensus = self._consensus(track_id)
        if consensus is None:
            return None

        best_text, best_weight, best_count = consensus
        if not self._is_well_supported(best_weight, best_count):
            return None

        return PlateUpdate(
            car_id=self._assign_car_id(best_text),
            plate_text=best_text,
            weight=best_weight,
            vehicle_type=self._best_vehicle_type(track_id),
            last_box=self.track_last_box.get(track_id)
        )

    @staticmethod
    def _is_well_supported(weight, count):
        return (
            count >= config.WELL_SUPPORTED_COUNT
            or weight >= config.WELL_SUPPORTED_WEIGHT
        )

    def finalize(self, track_id) -> Optional[PlateUpdate]:
        """Lock in the final reading for a track. Safe to call more
        than once; only returns an update the first time - None after
        that, or if the track had nothing recordable at all. A genuine
        sighting with no confident reading still comes back, marked
        unclear rather than being silently thrown away; whether it (or
        a confirmed reading) should actually produce a new log entry
        vs. update a recent one is decided by the caller, not here."""

        if track_id in self.finalized_track_ids:
            return None
        self.finalized_track_ids.add(track_id)

        vehicle_type = self._best_vehicle_type(track_id)
        last_box = self.track_last_box.get(track_id)
        consensus = self._consensus(track_id)

        if consensus is None:
            return PlateUpdate(
                car_id=None, plate_text="", weight=0.0,
                vehicle_type=vehicle_type, unclear=True, last_box=last_box
            )

        best_text, best_weight, best_count = consensus

        if not self._is_well_supported(best_weight, best_count):
            return PlateUpdate(
                car_id=None, plate_text=best_text, weight=best_weight,
                vehicle_type=vehicle_type, unclear=True, last_box=last_box
            )

        car_id = self._assign_car_id(best_text)

        return PlateUpdate(
            car_id=car_id,
            plate_text=best_text,
            weight=best_weight,
            vehicle_type=vehicle_type,
            unclear=False,
            last_box=last_box
        )

    def sync(self, current_track_ids):
        """Call once per frame with the track_ids visible in the
        current frame. Returns PlateUpdates for any tracks that just
        left the frame."""

        current_track_ids = set(current_track_ids) - {None}
        dropped = self.active_track_ids - current_track_ids
        self.active_track_ids = current_track_ids

        return self._finalize_many(dropped)

    def finalize_all(self):
        """Finalize every still-active track (call at video end)."""

        return self._finalize_many(list(self.active_track_ids))

    def _finalize_many(self, track_ids):
        """Finalize track_ids and return (track_id, PlateUpdate)
        pairs for the ones that produced a usable reading."""

        pairs = ((t, self.finalize(t)) for t in track_ids)
        return [(t, u) for t, u in pairs if u is not None]
