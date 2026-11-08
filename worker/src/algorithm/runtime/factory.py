"""Shared production composition: settings in, wired pipeline out.

Every entry point (CLI, batch regression, the split pipeline) builds its stages
here, so the wiring is not duplicated in the runner packages.  The split path
builds one stage per worker: each worker loads only the models it runs.

Nothing value-like lives in this module: tuning defaults and the bundled weight
paths are declared in :mod:`algorithm.config`, and this file only translates one
``PipelineSettings`` into adapters, trackers and the pipeline.
"""

from __future__ import annotations

from algorithm.config import PipelineSettings
from algorithm.perception.detect_models import DetectionAdapter
from algorithm.perception.pipeline import ObservationPipeline
from algorithm.perception.segment_model import ViaductSegmentAdapter
from algorithm.perception.stages import DetectStage, SegmentStage, TrackStage
from algorithm.perception.tracking import (
    LINE_LABELS,
    VEHICLE_LABELS,
    ByteTracker,
)
from algorithm.perception.viaduct_tracker import ViaductTracker


def stage_devices(settings: PipelineSettings) -> tuple[int | str, int | str]:
    """``(segment card, detect card)``; one card unless ``devices`` names two."""
    if settings.devices:
        return settings.devices[0], settings.devices[1]
    return settings.device, settings.device


def build_segment_stage(
    settings: PipelineSettings, device: int | str | None = None
) -> SegmentStage:
    """The GPU 0 half: one segmenter, nothing else."""
    return SegmentStage(
        ViaductSegmentAdapter(
            settings.viaduct_weight,
            device=settings.device if device is None else device,
            imgsz=settings.segment_imgsz,
            confidence=settings.viaduct_confidence,
            expected_label=settings.viaduct_label,
        ),
        mask_min_confidence=settings.mask_min_confidence,
    )


def build_detect_stage(
    settings: PipelineSettings, device: int | str | None = None
) -> DetectStage:
    """The GPU 1 half: both detectors, on the one card they share."""
    card = settings.device if device is None else device
    return DetectStage(
        DetectionAdapter(
            settings.vehicle_weight,
            device=card,
            imgsz=settings.detect_imgsz,
            confidence=settings.vehicle_confidence,
        ),
        DetectionAdapter(
            settings.line_weight,
            device=card,
            imgsz=settings.line_imgsz,
            confidence=settings.line_confidence,
            # A swapped checkpoint must fail at wiring time, not silently track
            # junk.
            expected_labels=(settings.line_label,),
        ),
    )


def build_track_stage(settings: PipelineSettings) -> TrackStage:
    """The coordinator half: the three stateful trackers, CPU only."""
    return TrackStage(
        viaduct_tracker=ViaductTracker(),
        vehicle_tracker=ByteTracker(
            # Fixed by design: identities are only issued for VEHICLE_LABELS.
            labels=VEHICLE_LABELS,
            detection_threshold=settings.vehicle_confidence,
            # Every detection already clears the adapter threshold, so
            # ByteTrack's low-score stage is intentionally collapsed onto the
            # same value.
            low_threshold=settings.vehicle_confidence,
            max_lost_ms=settings.track_max_lost_ms,
            match_threshold=settings.track_match_threshold,
        ),
        line_tracker=ByteTracker(
            labels=LINE_LABELS,
            detection_threshold=settings.line_confidence,
            low_threshold=settings.line_confidence,
            max_lost_ms=settings.track_max_lost_ms,
            match_threshold=settings.track_match_threshold,
        ),
    )


def build_pipeline(settings: PipelineSettings) -> ObservationPipeline:
    """Wire adapters, trackers and pipeline from one settings object."""
    segment_card, detect_card = stage_devices(settings)
    # One stage object per group, built once: constructing a stage loads its
    # models, so `build_pipeline` must never build the same stage twice.
    segment = build_segment_stage(settings, segment_card)
    detect = build_detect_stage(settings, detect_card)
    track = build_track_stage(settings)
    return ObservationPipeline(
        viaduct_segment=segment.viaduct_segment,
        vehicle_detection=detect.vehicle_detection,
        vehicle_tracker=track.vehicle_tracker,
        line_detection=detect.line_detection,
        line_tracker=track.line_tracker,
        viaduct_tracker=track.viaduct_tracker,
        mask_min_confidence=settings.mask_min_confidence,
    )
