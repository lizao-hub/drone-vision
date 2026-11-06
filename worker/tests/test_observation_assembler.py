"""The observation assembler is the one place a frame's validity is decided."""

import numpy as np

from algorithm.domain.geometry import Box, Mask
from algorithm.domain.observations import DetectedObject, TrackedObject
from algorithm.perception.assembly import (
    ObservationAssembler,
    StageResult,
    invalid_observation,
)
from algorithm.perception.viaduct_tracker import ViaductTracker

HEIGHT = 4
WIDTH = 6


def _deck_mask() -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.bool_)
    mask[:, : WIDTH // 2] = True
    return mask


def _segmentation(mask: np.ndarray, confidence: float = 0.9) -> DetectedObject:
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    box = Box(
        float(columns[0]) / WIDTH,
        float(rows[0]) / HEIGHT,
        float(columns[-1] + 1) / WIDTH,
        float(rows[-1] + 1) / HEIGHT,
    )
    return DetectedObject(label="viaduct", confidence=confidence, box=box, mask=Mask(mask))


class RecordingTracker:
    def __init__(self) -> None:
        self.calls: list[tuple[int, tuple[DetectedObject, ...]]] = []

    def update(
        self,
        timestamp_ms: int,
        detections: tuple[DetectedObject, ...],
        *,
        width: int,
        height: int,
        generation: int = 0,
    ) -> tuple[TrackedObject, ...]:
        self.calls.append((timestamp_ms, detections))
        return tuple(
            TrackedObject(
                track_id=f"{item.label}-bt-{index + 1:06d}",
                label=item.label,
                confidence=item.confidence,
                box=item.box,
            )
            for index, item in enumerate(detections)
        )


def _assembler(
    *,
    vehicle_tracker: RecordingTracker | None = None,
    line_tracker: RecordingTracker | None = None,
    viaduct_tracker: ViaductTracker | None = None,
) -> ObservationAssembler:
    return ObservationAssembler(
        vehicle_tracker=vehicle_tracker or RecordingTracker(),
        line_tracker=line_tracker or RecordingTracker(),
        viaduct_tracker=viaduct_tracker or ViaductTracker(),
    )


def test_segment_failure_invalidates_before_any_tracking():
    vehicle_tracker = RecordingTracker()
    deck_tracker = ViaductTracker()
    assembler = _assembler(vehicle_tracker=vehicle_tracker, viaduct_tracker=deck_tracker)

    observation = assembler.assemble(
        1000,
        0,
        0,
        height=HEIGHT,
        width=WIDTH,
        stages=StageResult(viaduct_error=RuntimeError("segment exploded"), segment_ms=12.0),
        processing_ms=13.0,
    )

    assert not observation.frame_valid
    assert observation.unknown_reasons == ("viaduct_adapter_error",)
    assert observation.diagnostics.segment_ms == 12.0
    assert vehicle_tracker.calls == []


def test_empty_deck_is_valid_and_skips_both_detectors():
    vehicle_tracker = RecordingTracker()
    line_tracker = RecordingTracker()
    assembler = _assembler(
        vehicle_tracker=vehicle_tracker, line_tracker=line_tracker
    )

    observation = assembler.assemble(
        1000,
        0,
        0,
        height=HEIGHT,
        width=WIDTH,
        stages=StageResult(viaduct=(), deck_empty=True, segment_ms=9.0),
        processing_ms=10.0,
    )

    assert observation.frame_valid
    assert observation.unknown_reasons == ("viaduct_not_detected",)
    assert observation.viaduct_tracks == ()
    assert vehicle_tracker.calls == []
    assert line_tracker.calls == []


def test_vehicle_failure_keeps_the_deck_tracks_but_no_vehicle_identity():
    vehicle_tracker = RecordingTracker()
    assembler = _assembler(vehicle_tracker=vehicle_tracker)
    instance = _segmentation(_deck_mask())

    observation = assembler.assemble(
        1000,
        0,
        0,
        height=HEIGHT,
        width=WIDTH,
        stages=StageResult(
            viaduct=(instance,),
            vehicle_error=RuntimeError("vehicle exploded"),
            segment_ms=9.0,
            detect_ms=8.0,
        ),
        processing_ms=20.0,
    )

    assert not observation.frame_valid
    assert observation.unknown_reasons == ("vehicle_adapter_error",)
    assert observation.viaduct_tracks == ()
    assert observation.diagnostics.detect_ms == 8.0
    assert vehicle_tracker.calls == []


def test_vehicle_tracker_failure_is_reported_as_tracker_error():
    class BrokenTracker(RecordingTracker):
        def update(self, timestamp_ms, detections, *, width, height, generation=0):
            raise RuntimeError("tracker exploded")

    assembler = _assembler(vehicle_tracker=BrokenTracker())
    car = DetectedObject(label="car", confidence=0.8, box=Box(0.0, 0.0, 0.3, 0.5))

    observation = assembler.assemble(
        1000,
        0,
        0,
        height=HEIGHT,
        width=WIDTH,
        stages=StageResult(viaduct=(_segmentation(_deck_mask()),), vehicle=(car,)),
        processing_ms=20.0,
    )

    assert not observation.frame_valid
    assert observation.unknown_reasons == ("tracker_error",)


def test_line_failure_arrives_after_the_vehicle_tracker_ran():
    # Matches the serial order: the vehicle tracker has already advanced when the
    # lane detector fails, and the lane tracker never runs.
    vehicle_tracker = RecordingTracker()
    line_tracker = RecordingTracker()
    assembler = _assembler(
        vehicle_tracker=vehicle_tracker, line_tracker=line_tracker
    )
    car = DetectedObject(label="car", confidence=0.8, box=Box(0.0, 0.0, 0.3, 0.5))

    observation = assembler.assemble(
        1000,
        0,
        0,
        height=HEIGHT,
        width=WIDTH,
        stages=StageResult(
            viaduct=(_segmentation(_deck_mask()),),
            vehicle=(car,),
            line_error=RuntimeError("lane exploded"),
            detect_ms=8.0,
            line_ms=7.0,
        ),
        processing_ms=25.0,
    )

    assert not observation.frame_valid
    assert observation.unknown_reasons == ("line_adapter_error",)
    assert len(vehicle_tracker.calls) == 1
    assert line_tracker.calls == []


def test_full_frame_carries_every_stage_and_timing():
    car = DetectedObject(label="car", confidence=0.8, box=Box(0.0, 0.0, 0.3, 0.5))
    marking = DetectedObject(label="line", confidence=0.7, box=Box(0.0, 0.0, 0.2, 1.0))
    assembler = _assembler()

    observation = assembler.assemble(
        1000,
        0,
        0,
        height=HEIGHT,
        width=WIDTH,
        stages=StageResult(
            viaduct=(_segmentation(_deck_mask()),),
            vehicle=(car,),
            line=(marking,),
            segment_ms=11.0,
            detect_ms=8.0,
            line_ms=7.0,
        ),
        processing_ms=27.5,
    )

    assert observation.frame_valid
    assert observation.unknown_reasons == ()
    assert [item.track_id for item in observation.vehicle_tracks] == ["car-bt-000001"]
    assert [item.track_id for item in observation.line_tracks] == ["line-bt-000001"]
    assert [item.track_id for item in observation.viaduct_tracks] == ["deck-000001"]
    assert observation.diagnostics.segment_ms == 11.0
    assert observation.diagnostics.detect_ms == 8.0
    assert observation.diagnostics.line_ms == 7.0
    assert observation.diagnostics.processing_ms == 27.5


def test_invalid_observation_never_carries_payload():
    observation = invalid_observation(1000, 3, 0, ("frame_invalid",), processing_ms=1.0)

    assert not observation.frame_valid
    assert observation.unknown_reasons == ("frame_invalid",)
    assert observation.viaduct == ()
    assert observation.vehicle_tracks == ()


def test_stage_result_rejects_negative_timings():
    import pytest

    with pytest.raises(ValueError):
        StageResult(segment_ms=-1.0)


def test_stage_result_carries_the_cause_not_a_bare_flag():
    import pytest

    error = RuntimeError("segment exploded")

    assert StageResult(viaduct_error=error).viaduct_error is error
    with pytest.raises(TypeError):
        StageResult(viaduct_error=True)
