"""Multi-process pipeline plumbing that can be tested without spawning GPUs."""

import sys
from pathlib import Path

import numpy as np
import pytest

from algorithm.config import MpPipelineSettings, PipelineSettings
from algorithm.engine import shared
from algorithm.engine.pipeline import MultiProcessPipeline, _Parts, _start_method
from algorithm.engine.shared import (
    DetectionDone,
    SegmentDone,
    Slot,
    frame_view,
    mask_view,
)
from algorithm.perception.assembly import ObservationAssembler
from algorithm.perception.viaduct_tracker import ViaductTracker
from drone_worker.args import parse_gpus


class _NullTracker:
    """Stands in for ByteTrack: the assembler only needs the tracker protocol."""

    def update(self, timestamp_ms, detections, *, width, height, generation=0):
        return ()


def _settings() -> PipelineSettings:
    return PipelineSettings(
        viaduct_weight=Path("viaduct.pt"),
        vehicle_weight=Path("vehicle.pt"),
        line_weight=Path("line.pt"),
    )


def _pipeline() -> MultiProcessPipeline:
    """A parent-side engine with no workers started: slots and queues only."""
    return MultiProcessPipeline(
        _settings(),
        MpPipelineSettings(),
        fps=25.0,
        assembler=ObservationAssembler(
            vehicle_tracker=_NullTracker(),
            line_tracker=_NullTracker(),
            viaduct_tracker=ViaductTracker(),
        ),
    )


def test_mp_settings_defaults_put_the_three_models_on_three_cards():
    settings = MpPipelineSettings()

    assert settings.segment_gpus == (0,)
    assert (settings.vehicle_gpu, settings.line_gpu) == (1, 2)
    assert settings.window == 4
    assert settings.gpus == (0, 1, 2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"segment_gpus": ()},
        {"segment_gpus": [0]},
        {"segment_gpus": (-1,)},
        {"segment_gpus": (True,)},
        {"vehicle_gpu": -1},
        {"line_gpu": 1.5},
        {"window": 0},
        {"window": True},
        {"mask_capacity": 0},
        {"start_timeout_s": 0.0},
        {"start_timeout_s": float("nan")},
    ],
)
def test_mp_settings_reject_invalid_layouts(kwargs):
    with pytest.raises((ValueError, TypeError)):
        MpPipelineSettings(**kwargs)


def test_mp_settings_allow_two_segment_cards():
    settings = MpPipelineSettings(segment_gpus=(0, 4), vehicle_gpu=5, line_gpu=6)

    assert settings.gpus == (0, 4, 5, 6)


def test_slot_views_share_the_shared_memory_backing_store():
    slot = Slot(height=4, width=6, mask_capacity=2)
    try:
        frame_view(slot.frame.buf, 4, 6)[:] = 7
        assert int(np.asarray(frame_view(slot.frame.buf, 4, 6))[0, 0, 2]) == 7
        assert int(np.asarray(frame_view(slot.frame.buf, 4, 6)).sum()) == 4 * 6 * 3 * 7

        mask_view(slot.masks.buf, 1, 4, 6)[:] = 1
        assert int(np.asarray(mask_view(slot.masks.buf, 0, 4, 6)).sum()) == 0
        assert int(np.asarray(mask_view(slot.masks.buf, 1, 4, 6)).sum()) == 24
    finally:
        slot.close_unlink()


def test_slot_names_are_unique_per_buffer():
    first = Slot(height=2, width=2, mask_capacity=1)
    second = Slot(height=2, width=2, mask_capacity=1)
    try:
        names = set(first.names) | set(second.names)
        assert len(names) == 6
    finally:
        first.close_unlink()
        second.close_unlink()


def _seg(*, error=None, empty=False) -> SegmentDone:
    return SegmentDone(
        seq=0,
        slot=0,
        count=0,
        boxes=(),
        confidences=(),
        segment_ms=1.0,
        empty=empty,
        error=error,
    )


def _detection(role: str) -> DetectionDone:
    return DetectionDone(
        role=role, seq=0, slot=0, detections=(), elapsed_ms=1.0, error=None
    )


def test_parts_wait_for_both_detectors_on_a_normal_frame():
    parts = _Parts(timestamp_ms=1000, generation=0, seg=_seg())

    assert not parts.ready()
    parts.vehicle = _detection(shared.VEHICLE)
    assert not parts.ready()
    parts.line = _detection(shared.LINE)
    assert parts.ready()


def test_failed_or_empty_segment_never_waits_for_detectors():
    assert _Parts(timestamp_ms=0, generation=0, seg=_seg(error="boom")).ready()
    assert _Parts(timestamp_ms=0, generation=0, seg=_seg(empty=True)).ready()
    assert not _Parts(timestamp_ms=0, generation=0).ready()


@pytest.mark.parametrize(
    ("text", "expected"),
    [("0", (0,)), ("0,1,2", (0, 1, 2)), (" 4 , 5 ", (4, 5))],
)
def test_parse_gpus_accepts_comma_separated_ids(text, expected):
    assert parse_gpus(text) == expected


@pytest.mark.parametrize("text", ["", "a", "0,-1", "0,,1"])
def test_parse_gpus_rejects_junk(text):
    with pytest.raises(ValueError):
        parse_gpus(text)


def test_start_method_is_spawn_on_windows():
    if sys.platform == "win32":
        assert _start_method() == "spawn"


def test_submit_rejects_frames_the_shared_buffers_cannot_hold():
    pipeline = _pipeline()
    try:
        with pytest.raises(ValueError):
            pipeline.submit(
                np.zeros((4, 6), dtype=np.uint8), timestamp_ms=1000, frame_sequence=0
            )
        with pytest.raises(ValueError):
            pipeline.submit(
                np.zeros((4, 6, 3), dtype=np.float32),
                timestamp_ms=1000,
                frame_sequence=0,
            )
        with pytest.raises(ValueError):
            pipeline.submit(
                np.zeros((4, 6, 1), dtype=np.uint8), timestamp_ms=1000, frame_sequence=0
            )
        assert not pipeline.started
    finally:
        pipeline.close()


def test_the_engine_holds_the_assembler_that_decides_validity():
    # The parent never loads a model; the one object holding cross-frame state is
    # handed in, so the entry point and the tests can inspect the same instance.
    pipeline = _pipeline()
    try:
        assert isinstance(pipeline.assembler, ObservationAssembler)
    finally:
        pipeline.close()


def test_a_worker_error_text_becomes_a_stage_error():
    # Workers report failures as text; the parent must rebuild the cause, not a flag.
    pipeline = _pipeline()
    try:
        observation = pipeline._assemble(
            0, _Parts(timestamp_ms=1000, generation=0, seg=_seg(error="cuda OOM"))
        )
    finally:
        pipeline.close()

    assert not observation.frame_valid
    assert observation.unknown_reasons == ("viaduct_adapter_error",)
