"""Frame-level orchestration: segment, mask, detect, track, observe."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from algorithm.domain.observations import Diagnostics, Observation
from algorithm.perception.masking import blackout_off_deck
from algorithm.perception.stages import (
    DetectionPredictor,
    DetectStage,
    SegmentationPredictor,
    SegmentStage,
    Tracker,
    TrackStage,
)
from algorithm.perception.viaduct_tracker import ViaductTracker


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
        # The stage objects own the models and the trackers; the pipeline is
        # only the order they run in, which is what the split path reuses.
        self.segment = SegmentStage(
            viaduct_segment, mask_min_confidence=mask_min_confidence
        )
        self.detect = DetectStage(vehicle_detection, line_detection)
        self.track = TrackStage(
            viaduct_tracker=viaduct_tracker,
            vehicle_tracker=vehicle_tracker,
            line_tracker=line_tracker,
        )

    def process(
        self,
        frame: np.ndarray,
        timestamp_ms: int,
        frame_sequence: int,
        generation: int = 0,
    ) -> Observation:
        started = perf_counter()
        if (
            not isinstance(frame, np.ndarray)
            or frame.ndim != 3
            or frame.size == 0
        ):
            return self._invalid(
                timestamp_ms, frame_sequence, generation, ("frame_invalid",), started
            )
        height, width = int(frame.shape[0]), int(frame.shape[1])

        # Step 1: viaduct segmentation. A backend failure voids the frame.
        segment_started = perf_counter()
        try:
            viaduct = self.segment.predict(frame)
        except Exception:  # noqa: BLE001 - third-party runtimes raise their own types
            return self._invalid(
                timestamp_ms,
                frame_sequence,
                generation,
                ("viaduct_adapter_error",),
                started,
                segment_ms=(perf_counter() - segment_started) * 1000.0,
            )
        segment_ms = (perf_counter() - segment_started) * 1000.0

        # Step 2: give each instance a deck id that survives frame-to-frame
        # instance reordering. A tracker failure voids the frame like a model one.
        try:
            viaduct_tracks = self.track.viaduct(timestamp_ms, viaduct)
        except Exception:  # noqa: BLE001
            return self._invalid(
                timestamp_ms,
                frame_sequence,
                generation,
                ("viaduct_tracker_error",),
                started,
                segment_ms=segment_ms,
            )

        # Step 3: merge every instance into the one masked image the detectors
        # see. Nothing off the deck survives, so no post-filter is needed later.
        masked = blackout_off_deck(frame, viaduct)

        if masked is None:
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
                ),
            )

        # Step 4: detect vehicles on the masked image.
        detect_started = perf_counter()
        try:
            vehicle = self.detect.vehicles(masked)
        except Exception:  # noqa: BLE001
            return self._invalid(
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
            vehicle_tracks = self.track.vehicles(
                timestamp_ms,
                vehicle,
                width=width,
                height=height,
                generation=generation,
            )
        except Exception:  # noqa: BLE001
            return self._invalid(
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
            line = self.detect.lines(masked)
        except Exception:  # noqa: BLE001
            return self._invalid(
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
            line_tracks = self.track.lines(
                timestamp_ms,
                line,
                width=width,
                height=height,
                generation=generation,
            )
        except Exception:  # noqa: BLE001
            return self._invalid(
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
    ) -> Observation:
        return invalid_observation(
            timestamp_ms,
            frame_sequence,
            generation,
            unknown_reasons,
            processing_ms=(perf_counter() - started) * 1000.0,
            segment_ms=segment_ms,
            detect_ms=detect_ms,
            line_ms=line_ms,
        )


def invalid_observation(
    timestamp_ms: int,
    frame_sequence: int,
    generation: int,
    unknown_reasons: tuple[str, ...],
    *,
    processing_ms: float,
    segment_ms: float = 0.0,
    detect_ms: float = 0.0,
    line_ms: float = 0.0,
) -> Observation:
    """One frame's verdict when a stage failed.

    Shared with the split pipeline's coordinator: an invalid frame must look the
    same whichever topology produced it.
    """
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
            processing_ms=processing_ms,
        ),
    )
