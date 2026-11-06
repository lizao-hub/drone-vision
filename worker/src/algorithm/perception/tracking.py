"""Object tracking: one ultralytics BYTETracker per enabled label.

ByteTrack associates on IoU/Kalman alone and ignores the class column, so a
single shared tracker could hand a truck's identity to a neighbouring car.
Each label therefore owns its own tracker (and its own id namespace), which
also keeps ``bus``/``car``/``truck`` identities independent.

The same class tracks lane lines: it is the label set that differs, not the
algorithm (``labels=LINE_LABELS``).

Only matched detection geometry is published; the Kalman prediction is never
exposed, matching the rest of the pipeline's "facts only" contract.
"""

from __future__ import annotations

from math import ceil
from types import SimpleNamespace
from typing import Any

import numpy as np

from algorithm.domain.observations import DetectedObject, TrackedObject

VEHICLE_LABELS: tuple[str, ...] = ("car",)
LINE_LABELS: tuple[str, ...] = ("line",)


class ByteTracker:
    def __init__(
        self,
        *,
        labels: tuple[str, ...] = VEHICLE_LABELS,
        detection_threshold: float = 0.30,
        low_threshold: float = 0.10,
        max_lost_ms: int = 1000,
        fps: float = 24.0,
        match_threshold: float = 0.80,
        new_track_threshold: float | None = None,
    ) -> None:
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("tracker labels must be a non-empty unique sequence")
        if not 0.0 < low_threshold <= detection_threshold <= 1.0:
            raise ValueError("thresholds must satisfy 0 < low <= detection <= 1")
        if max_lost_ms <= 0:
            raise ValueError("max_lost_ms must be positive")
        if fps <= 0:
            raise ValueError("fps must be positive")
        if not 0.0 < match_threshold <= 1.0:
            raise ValueError("match_threshold must be within (0, 1]")
        self.labels = tuple(labels)
        self.detection_threshold = detection_threshold
        self.low_threshold = low_threshold
        self.max_lost_ms = max_lost_ms
        self.fps = fps
        self.match_threshold = match_threshold
        self.new_track_threshold = (
            detection_threshold if new_track_threshold is None else new_track_threshold
        )
        self._trackers: dict[str, Any] = {}
        self._seen: dict[str, dict[int, int]] = {}
        self._width = 0
        self._height = 0
        self._last_ms: int | None = None
        self._generation: int | None = None

    # -- lifecycle ---------------------------------------------------------
    def update(
        self,
        timestamp_ms: int,
        detections: tuple[DetectedObject, ...],
        *,
        width: int,
        height: int,
        generation: int = 0,
    ) -> tuple[TrackedObject, ...]:
        if timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative")
        if width <= 0 or height <= 0:
            raise ValueError("frame geometry must be positive")
        # The detector may report bike/person; only configured labels get id.
        detections = tuple(item for item in detections if item.label in self.labels)
        if (
            self._last_ms is not None
            and generation == self._generation
            and timestamp_ms <= self._last_ms
        ):
            raise ValueError("tracker timestamps must strictly increase")

        if (
            not self._trackers
            or generation != self._generation
            or (width, height) != (self._width, self._height)
        ):
            self._build(width, height)
            self._generation = generation
        self._expire(timestamp_ms)

        published: list[TrackedObject] = []
        for label in self.labels:
            published.extend(
                self._update_label(label, timestamp_ms, [item for item in detections if item.label == label])
            )
        self._last_ms = timestamp_ms
        return tuple(sorted(published, key=lambda item: item.track_id))

    def reset(self) -> None:
        self._trackers.clear()
        self._seen.clear()
        self._last_ms = None
        self._generation = None

    # -- internals ---------------------------------------------------------
    def _build(self, width: int, height: int) -> None:
        from ultralytics.trackers.byte_tracker import BYTETracker  # type: ignore[import-not-found]

        self._width, self._height = width, height
        buffer_frames = max(1, ceil(self.max_lost_ms * self.fps / 1000.0))
        self._trackers = {
            label: BYTETracker(self._args(buffer_frames)) for label in self.labels
        }
        self._seen = {label: {} for label in self.labels}
        self._last_ms = None

    def _args(self, buffer_frames: int) -> SimpleNamespace:
        return SimpleNamespace(
            track_high_thresh=self.detection_threshold,
            track_low_thresh=self.low_threshold,
            new_track_thresh=self.new_track_threshold,
            track_buffer=buffer_frames,
            match_thresh=self.match_threshold,
            fuse_score=True,
        )

    def _expire(self, timestamp_ms: int) -> None:
        """Expire tracks by source time, not by an assumed frame rate."""
        for label, tracker in self._trackers.items():
            seen = self._seen[label]
            for name in ("tracked_stracks", "lost_stracks"):
                retained = []
                for track in getattr(tracker, name):
                    identity = int(track.track_id)
                    last_seen = seen.get(identity, timestamp_ms)
                    if timestamp_ms - last_seen > self.max_lost_ms:
                        track.mark_removed()
                        seen.pop(identity, None)
                    else:
                        retained.append(track)
                setattr(tracker, name, retained)

    def _update_label(
        self,
        label: str,
        timestamp_ms: int,
        items: list[DetectedObject],
    ) -> list[TrackedObject]:
        from ultralytics.engine.results import Boxes  # type: ignore[import-not-found]

        tracker = self._trackers[label]
        seen = self._seen[label]
        values = np.asarray(
            [
                [
                    item.box.x1 * self._width,
                    item.box.y1 * self._height,
                    item.box.x2 * self._width,
                    item.box.y2 * self._height,
                    item.confidence,
                    0.0,
                ]
                for item in items
            ],
            dtype=np.float32,
        ).reshape(-1, 6)
        rows = np.asarray(
            tracker.update(Boxes(values, (self._height, self._width))),
            dtype=np.float32,
        ).reshape(-1, 8)
        output: list[TrackedObject] = []
        used: set[int] = set()
        for row in rows:
            index, identity = int(row[7]), int(row[4])
            if index in used or not 0 <= index < len(items):
                raise RuntimeError("invalid ByteTrack detection association")
            used.add(index)
            seen[identity] = timestamp_ms
            detection = items[index]
            # Weak boxes keep identity alive but must never reach Observation.
            if detection.confidence < self.detection_threshold:
                continue
            output.append(
                TrackedObject(
                    track_id=f"{label}-bt-{identity:06d}",
                    label=label,
                    confidence=detection.confidence,
                    box=detection.box,
                )
            )
        live = {int(track.track_id) for track in (*tracker.tracked_stracks, *tracker.lost_stracks)}
        self._seen[label] = {
            key: value for key, value in seen.items() if key in live
        }
        return output
