"""The three per-frame stages, as separately runnable pieces.

``ObservationPipeline`` runs them back to back in one process; the two-card
pipeline runs the first on one worker (GPU 0), the second on another (GPU 1) and
the third in the coordinator.  Both paths call *this* code, which is the only
reason their observations are comparable at all.

The split follows what can be moved between processes:

* :class:`SegmentStage` and :class:`DetectStage` wrap models: pure functions of
  the frame (and the deck mask), so they may run anywhere, in any order.
* :class:`TrackStage` wraps the three trackers.  ByteTrack rejects a
  non-advancing timestamp, so this stage -- not the models -- is the one that
  has to see frames in order, and the one that stays where the frame order is
  known.

The two predictor protocols live here rather than in the pipeline module so a
stage never has to import the orchestrator that composes it.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from algorithm.domain.observations import DetectedObject, TrackedObject
from algorithm.perception.viaduct_tracker import ViaductTracker


class SegmentationPredictor(Protocol):
    def predict(self, frame: np.ndarray) -> tuple[DetectedObject, ...]: ...


class DetectionPredictor(Protocol):
    def predict(self, frame: np.ndarray) -> tuple[DetectedObject, ...]: ...


class Tracker(Protocol):
    def update(
        self,
        timestamp_ms: int,
        detections: tuple[DetectedObject, ...],
        *,
        width: int,
        height: int,
        generation: int = 0,
    ) -> tuple[TrackedObject, ...]: ...


class SegmentStage:
    """Frame in, viaduct instances out: one per deck, each carrying its mask.

    Instances below ``mask_min_confidence`` are dropped here, once: the blackout
    then unions exactly the masks the observation reports, and no later stage
    filters again.
    """

    def __init__(
        self,
        viaduct_segment: SegmentationPredictor,
        *,
        mask_min_confidence: float = 0.25,
    ) -> None:
        if not 0.0 <= mask_min_confidence <= 1.0:
            raise ValueError("mask_min_confidence must be within [0, 1]")
        self.viaduct_segment = viaduct_segment
        self.mask_min_confidence = mask_min_confidence

    def predict(self, frame: np.ndarray) -> tuple[DetectedObject, ...]:
        """Every instance above the mask threshold; raises if the backend does."""
        return tuple(
            item
            for item in self.viaduct_segment.predict(frame)
            if item.confidence >= self.mask_min_confidence
        )


class DetectStage:
    """Blacked-out frame in, vehicle and lane-line detections out.

    Both models read the *same* masked image, so the blackout is computed once
    by the caller and shared.  They are kept as two calls rather than one so a
    failure names the model that failed.
    """

    def __init__(
        self,
        vehicle_detection: DetectionPredictor,
        line_detection: DetectionPredictor,
    ) -> None:
        self.vehicle_detection = vehicle_detection
        self.line_detection = line_detection

    def vehicles(self, masked: np.ndarray) -> tuple[DetectedObject, ...]:
        return self.vehicle_detection.predict(masked)

    def lines(self, masked: np.ndarray) -> tuple[DetectedObject, ...]:
        return self.line_detection.predict(masked)


class TrackStage:
    """Detections in, identities out: the three stateful trackers."""

    def __init__(
        self,
        *,
        viaduct_tracker: ViaductTracker,
        vehicle_tracker: Tracker,
        line_tracker: Tracker,
    ) -> None:
        self.viaduct_tracker = viaduct_tracker
        self.vehicle_tracker = vehicle_tracker
        self.line_tracker = line_tracker

    def viaduct(
        self, timestamp_ms: int, instances: tuple[DetectedObject, ...]
    ) -> tuple[TrackedObject, ...]:
        return self.viaduct_tracker.update(timestamp_ms, instances)

    def vehicles(
        self,
        timestamp_ms: int,
        detections: tuple[DetectedObject, ...],
        *,
        width: int,
        height: int,
        generation: int = 0,
    ) -> tuple[TrackedObject, ...]:
        return self.vehicle_tracker.update(
            timestamp_ms,
            detections,
            width=width,
            height=height,
            generation=generation,
        )

    def lines(
        self,
        timestamp_ms: int,
        detections: tuple[DetectedObject, ...],
        *,
        width: int,
        height: int,
        generation: int = 0,
    ) -> tuple[TrackedObject, ...]:
        return self.line_tracker.update(
            timestamp_ms,
            detections,
            width=width,
            height=height,
            generation=generation,
        )
