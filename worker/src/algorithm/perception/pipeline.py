"""Frame-level orchestration: segment, mask, detect, track, observe."""

from __future__ import annotations

from time import perf_counter
from typing import Protocol

import numpy as np

from algorithm.domain.observations import (
    DetectedObject,
    Diagnostics,
    Observation,
    TrackedObject,
)
from algorithm.perception.masking import apply_mask, union_mask
from algorithm.perception.raw import RawDetection
from algorithm.perception.viaduct_tracker import ViaductTracker


class SegmentationPredictor(Protocol):
    def predict(self, frame: np.ndarray) -> tuple[DetectedObject, ...]: ...


class DetectionPredictor(Protocol):
    """Detector output before it becomes domain state: see :class:`RawDetection`."""

    def predict(self, frame: np.ndarray) -> tuple[RawDetection, ...]: ...


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


class ObservationPipeline:
    """One frame in, one :class:`Observation` out.

    Model failures invalidate the whole frame (same policy as the reference
    project): a partially filled observation must never reach downstream logic.
    All three models (viaduct segmentation, vehicle detection, lane-line
    detection), the two object trackers and the deck tracker are required.

    The deck mask is a plain union of this frame's instances: no temporal
    hold-over, no boundary dilation, no centre-in-mask post-filter.  Deck
    identities, however, are matched across frames by the injected tracker.
    """

    def __init__(
        self,
        *,
        viaduct_segment: SegmentationPredictor,
        vehicle_detection: DetectionPredictor,
        vehicle_tracker: Tracker,
        line_detection: DetectionPredictor,
        line_tracker: Tracker,
        viaduct_tracker: ViaductTracker,
        mask_min_confidence: float = 0.25,
    ) -> None:
        if not 0.0 <= mask_min_confidence <= 1.0:
            raise ValueError("mask_min_confidence must be within [0, 1]")
        self.viaduct_segment = viaduct_segment
        self.vehicle_detection = vehicle_detection
        self.vehicle_tracker = vehicle_tracker
        self.line_detection = line_detection
        self.line_tracker = line_tracker
        self.viaduct_tracker = viaduct_tracker
        self.mask_min_confidence = mask_min_confidence
        # Reused blackout buffer: the masked frame only lives for this call, and
        # allocating 6 MB per frame is measurable at 1080p.
        self._masked: np.ndarray | None = None

    def process(
        self,
        frame: np.ndarray,
        timestamp_ms: int,
        frame_sequence: int,
        generation: int = 0,
        *,
        decode_ms: float = 0.0,
        queue_wait_ms: float = 0.0,
    ) -> Observation:
        started = perf_counter()

        def invalid(*args: object, **kwargs: object) -> Observation:
            """Void the frame, still reporting the caller's own I/O time."""
            return self._invalid(
                *args,
                **kwargs,
                decode_ms=decode_ms,
                queue_wait_ms=queue_wait_ms,
            )

        if (
            not isinstance(frame, np.ndarray)
            or frame.ndim != 3
            or frame.size == 0
        ):
            return invalid(
                timestamp_ms, frame_sequence, generation, ("frame_invalid",), started
            )
        height, width = int(frame.shape[0]), int(frame.shape[1])

        # Step 1: viaduct segmentation. A backend failure voids the frame.
        segment_started = perf_counter()
        try:
            viaduct_segmentations = self.viaduct_segment.predict(frame)
        except Exception:  # noqa: BLE001 - third-party runtimes raise their own types
            return invalid(
                timestamp_ms,
                frame_sequence,
                generation,
                ("viaduct_adapter_error",),
                started,
                segment_ms=(perf_counter() - segment_started) * 1000.0,
            )
        segment_ms = (perf_counter() - segment_started) * 1000.0

        # Every instance keeps its own mask in the observation; the union below
        # is only the derived region used for blackout and detection.
        viaduct = tuple(
            item
            for item in viaduct_segmentations
            if item.confidence >= self.mask_min_confidence
        )

        # Step 2: give each instance a deck id that survives frame-to-frame
        # instance reordering. A tracker failure voids the frame like a model one.
        try:
            viaduct_tracks = self.viaduct_tracker.update(timestamp_ms, viaduct)
        except Exception:  # noqa: BLE001
            return invalid(
                timestamp_ms,
                frame_sequence,
                generation,
                ("viaduct_tracker_error",),
                started,
                segment_ms=segment_ms,
            )

        # Step 3: merge every instance into the one region used for blackout.
        mask = union_mask(viaduct, height=height, width=width)

        if not mask.data.any():
            return Observation(
                timestamp_ms=timestamp_ms,
                frame_sequence=frame_sequence,
                generation=generation,
                frame_valid=True,
                viaduct=viaduct,
                viaduct_tracks=viaduct_tracks,
                vehicle=(),
                vehicle_tracks=(),
                line_tracks=(),
                unknown_reasons=("viaduct_not_detected",),
                diagnostics=Diagnostics(
                    segment_ms=segment_ms,
                    detect_ms=0.0,
                    line_ms=0.0,
                    processing_ms=(perf_counter() - started) * 1000.0,
                    decode_ms=decode_ms,
                    queue_wait_ms=queue_wait_ms,
                ),
            )

        # Step 4: blacken everything off the deck, then detect on that image.
        if (
            self._masked is None
            or self._masked.shape != frame.shape
            or self._masked.dtype != frame.dtype
        ):
            self._masked = np.empty_like(frame)
        masked = apply_mask(frame, mask, out=self._masked)
        detect_started = perf_counter()
        try:
            vehicle = tuple(
                item.to_detected() for item in self.vehicle_detection.predict(masked)
            )
        except Exception:  # noqa: BLE001
            return invalid(
                timestamp_ms,
                frame_sequence,
                generation,
                ("vehicle_adapter_error",),
                started,
                segment_ms=segment_ms,
                detect_ms=(perf_counter() - detect_started) * 1000.0,
            )
        detect_ms = (perf_counter() - detect_started) * 1000.0

        # Step 5: assign vehicle identities.
        try:
            vehicle_tracks = self.vehicle_tracker.update(
                timestamp_ms,
                vehicle,
                width=width,
                height=height,
                generation=generation,
            )
        except Exception:  # noqa: BLE001
            return invalid(
                timestamp_ms,
                frame_sequence,
                generation,
                ("tracker_error",),
                started,
                segment_ms=segment_ms,
                detect_ms=detect_ms,
            )

        # Step 6: lane markings on the same blacked-out image, then track them.
        line_started = perf_counter()
        try:
            line = tuple(
                item.to_detected() for item in self.line_detection.predict(masked)
            )
        except Exception:  # noqa: BLE001
            return invalid(
                timestamp_ms,
                frame_sequence,
                generation,
                ("line_adapter_error",),
                started,
                segment_ms=segment_ms,
                detect_ms=detect_ms,
                line_ms=(perf_counter() - line_started) * 1000.0,
            )
        line_ms = (perf_counter() - line_started) * 1000.0
        try:
            line_tracks = self.line_tracker.update(
                timestamp_ms,
                line,
                width=width,
                height=height,
                generation=generation,
            )
        except Exception:  # noqa: BLE001
            return invalid(
                timestamp_ms,
                frame_sequence,
                generation,
                ("line_tracker_error",),
                started,
                segment_ms=segment_ms,
                detect_ms=detect_ms,
                line_ms=line_ms,
            )

        return Observation(
            timestamp_ms=timestamp_ms,
            frame_sequence=frame_sequence,
            generation=generation,
            frame_valid=True,
            viaduct=viaduct,
            viaduct_tracks=viaduct_tracks,
            vehicle=vehicle,
            vehicle_tracks=vehicle_tracks,
            line_tracks=line_tracks,
            unknown_reasons=(),
            diagnostics=Diagnostics(
                segment_ms=segment_ms,
                detect_ms=detect_ms,
                line_ms=line_ms,
                processing_ms=(perf_counter() - started) * 1000.0,
                decode_ms=decode_ms,
                queue_wait_ms=queue_wait_ms,
            ),
        )

    @staticmethod
    def _invalid(
        timestamp_ms: int,
        frame_sequence: int,
        generation: int,
        unknown_reasons: tuple[str, ...],
        started: float,
        *,
        segment_ms: float = 0.0,
        detect_ms: float = 0.0,
        line_ms: float = 0.0,
        decode_ms: float = 0.0,
        queue_wait_ms: float = 0.0,
    ) -> Observation:
        return Observation(
            timestamp_ms=timestamp_ms,
            frame_sequence=frame_sequence,
            generation=generation,
            frame_valid=False,
            viaduct=(),
            viaduct_tracks=(),
            vehicle=(),
            vehicle_tracks=(),
            line_tracks=(),
            unknown_reasons=unknown_reasons,
            diagnostics=Diagnostics(
                segment_ms=segment_ms,
                detect_ms=detect_ms,
                line_ms=line_ms,
                processing_ms=(perf_counter() - started) * 1000.0,
                decode_ms=decode_ms,
                queue_wait_ms=queue_wait_ms,
            ),
        )
