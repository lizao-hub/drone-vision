"""Merging model output into an observation: the one pair of contracts.

Two things live here, and both are model-free and GPU-free:

``StageResult``
    What one frame's model stages produced -- raw detections, each stage's fate
    and its duration -- with no notion of tracking or frame validity.
``ObservationAssembler``
    What those outputs mean: the three trackers, the order they are called in,
    and every invalidation rule.

This is the seam between "what the models saw" and "what it means".  It runs in
the orchestrating process, on stage outputs, so one object decides every frame
and the trackers see frames in submission order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from algorithm.domain.observations import (
    DetectedObject,
    Diagnostics,
    Observation,
    TrackedObject,
)
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


@dataclass(frozen=True, slots=True)
class StageResult:
    """One frame's raw model outputs plus each stage's fate and duration.

    Produced by the parent from the workers' messages -- one
    :class:`~algorithm.domain.observations.DetectedObject` per instance, the
    exception that ended a stage if one did -- and consumed by
    :class:`ObservationAssembler` alone.  It deliberately says nothing about
    tracking or frame validity: those decisions belong to the assembler, which is
    why a frame can be invalid for six different reasons and still be described
    by the same object.

    ``viaduct`` is already confidence-filtered and every instance carries its
    mask; ``deck_empty`` means the union mask has no pixel at all, so no detector
    ran (the workers skip both detectors in exactly that case).  A ``*_error``
    field carries the exception that ended that stage rather than a bare flag, so
    the cause survives the boundary instead of being flattened to ``True``.
    """

    viaduct: tuple[DetectedObject, ...] = ()
    vehicle: tuple[DetectedObject, ...] = ()
    line: tuple[DetectedObject, ...] = ()
    segment_ms: float = 0.0
    detect_ms: float = 0.0
    line_ms: float = 0.0
    viaduct_error: BaseException | None = None
    vehicle_error: BaseException | None = None
    line_error: BaseException | None = None
    deck_empty: bool = False

    def __post_init__(self) -> None:
        for name in ("segment_ms", "detect_ms", "line_ms"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("viaduct_error", "vehicle_error", "line_error"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, BaseException):
                raise TypeError(f"{name} must be an exception or None")


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
    """An invalid frame: no observations may travel with it, only the reason."""
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


def valid_observation(
    timestamp_ms: int,
    frame_sequence: int,
    generation: int,
    *,
    viaduct: tuple[DetectedObject, ...],
    viaduct_tracks: tuple[TrackedObject, ...],
    vehicle: tuple[DetectedObject, ...] = (),
    vehicle_tracks: tuple[TrackedObject, ...] = (),
    line_tracks: tuple[TrackedObject, ...] = (),
    unknown_reasons: tuple[str, ...] = (),
    processing_ms: float,
    segment_ms: float = 0.0,
    detect_ms: float = 0.0,
    line_ms: float = 0.0,
) -> Observation:
    """A valid frame: the deck (possibly empty) plus whatever the detectors found."""
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
        unknown_reasons=unknown_reasons,
        diagnostics=Diagnostics(
            segment_ms=segment_ms,
            detect_ms=detect_ms,
            line_ms=line_ms,
            processing_ms=processing_ms,
        ),
    )


class ObservationAssembler:
    """Turn one frame's :class:`StageResult` into an :class:`Observation`.

    Named after what it produces, not after what it consumes: it never sees an
    image, only detections, masks and timings, so it is not a frame assembler.

    Owns the three trackers and the whole invalidation policy: a model failure
    voids the frame, a tracker failure voids it, an empty deck is a *valid*
    frame with ``viaduct_not_detected`` and no detection, and the reason that is
    reported is the earliest stage that failed (segment, then deck tracking, then
    vehicle, then vehicle tracking, then lane detection, then lane tracking).

    The ``diagnostics`` of an invalid frame only describe the stages the policy
    reached, so a frame that died in deck tracking reports no detection time even
    when the detectors had already run on it.

    The first occurrence of each distinct failure is printed once, the same way
    the worker processes report theirs: a frame that goes invalid must never
    disappear without a cause.

    This class performs no inference, holds no model handle and touches no GPU,
    so the multi-process parent can run it while the workers own the models.
    Trackers are called exactly once per assembled frame, in submission order.
    """

    def __init__(
        self,
        *,
        vehicle_tracker: Tracker,
        line_tracker: Tracker,
        viaduct_tracker: ViaductTracker,
    ) -> None:
        self.vehicle_tracker = vehicle_tracker
        self.line_tracker = line_tracker
        self.viaduct_tracker = viaduct_tracker
        # Distinct failures already reported, so a broken model logs once, not
        # once per frame.
        self._reported: set[str] = set()

    def assemble(
        self,
        timestamp_ms: int,
        frame_sequence: int,
        generation: int,
        *,
        height: int,
        width: int,
        stages: StageResult,
        processing_ms: float,
    ) -> Observation:
        if stages.viaduct_error is not None:
            self._report_once("segment", frame_sequence, stages.viaduct_error)
            return invalid_observation(
                timestamp_ms,
                frame_sequence,
                generation,
                ("viaduct_adapter_error",),
                processing_ms=processing_ms,
                segment_ms=stages.segment_ms,
            )
        # Step 2: give each instance a deck id that survives frame-to-frame
        # instance reordering. A tracker failure voids the frame like a model one.
        try:
            viaduct_tracks = self.viaduct_tracker.update(timestamp_ms, stages.viaduct)
        except Exception as exc:  # noqa: BLE001 - third-party runtimes raise their own types
            self._report_once("deck tracker", frame_sequence, exc)
            return invalid_observation(
                timestamp_ms,
                frame_sequence,
                generation,
                ("viaduct_tracker_error",),
                processing_ms=processing_ms,
                segment_ms=stages.segment_ms,
            )

        if stages.deck_empty:
            return valid_observation(
                timestamp_ms,
                frame_sequence,
                generation,
                viaduct=stages.viaduct,
                viaduct_tracks=viaduct_tracks,
                unknown_reasons=("viaduct_not_detected",),
                processing_ms=processing_ms,
                segment_ms=stages.segment_ms,
            )

        if stages.vehicle_error is not None:
            self._report_once("vehicle detector", frame_sequence, stages.vehicle_error)
            return invalid_observation(
                timestamp_ms,
                frame_sequence,
                generation,
                ("vehicle_adapter_error",),
                processing_ms=processing_ms,
                segment_ms=stages.segment_ms,
                detect_ms=stages.detect_ms,
            )
        try:
            vehicle_tracks = self.vehicle_tracker.update(
                timestamp_ms,
                stages.vehicle,
                width=width,
                height=height,
                generation=generation,
            )
        except Exception as exc:  # noqa: BLE001
            self._report_once("vehicle tracker", frame_sequence, exc)
            return invalid_observation(
                timestamp_ms,
                frame_sequence,
                generation,
                ("tracker_error",),
                processing_ms=processing_ms,
                segment_ms=stages.segment_ms,
                detect_ms=stages.detect_ms,
            )

        if stages.line_error is not None:
            self._report_once("lane detector", frame_sequence, stages.line_error)
            return invalid_observation(
                timestamp_ms,
                frame_sequence,
                generation,
                ("line_adapter_error",),
                processing_ms=processing_ms,
                segment_ms=stages.segment_ms,
                detect_ms=stages.detect_ms,
                line_ms=stages.line_ms,
            )
        try:
            line_tracks = self.line_tracker.update(
                timestamp_ms,
                stages.line,
                width=width,
                height=height,
                generation=generation,
            )
        except Exception as exc:  # noqa: BLE001
            self._report_once("lane tracker", frame_sequence, exc)
            return invalid_observation(
                timestamp_ms,
                frame_sequence,
                generation,
                ("line_tracker_error",),
                processing_ms=processing_ms,
                segment_ms=stages.segment_ms,
                detect_ms=stages.detect_ms,
                line_ms=stages.line_ms,
            )

        return valid_observation(
            timestamp_ms,
            frame_sequence,
            generation,
            viaduct=stages.viaduct,
            viaduct_tracks=viaduct_tracks,
            vehicle=stages.vehicle,
            vehicle_tracks=vehicle_tracks,
            line_tracks=line_tracks,
            processing_ms=processing_ms,
            segment_ms=stages.segment_ms,
            detect_ms=stages.detect_ms,
            line_ms=stages.line_ms,
        )

    def _report_once(
        self, stage: str, frame_sequence: int, error: BaseException
    ) -> None:
        message = repr(error)
        if message in self._reported:
            return
        self._reported.add(message)
        print(
            f"[pipeline] frame {frame_sequence}: {stage} failed: {message}",
            flush=True,
        )
