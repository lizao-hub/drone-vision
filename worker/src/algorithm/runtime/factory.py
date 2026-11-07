"""Shared production composition: settings in, wired pipeline out.

Every entry point (CLI, batch regression, a future container loop) builds its
pipeline here, so the wiring is not duplicated in the runner packages.

Nothing value-like lives in this module: tuning defaults and the bundled weight
paths are declared in :mod:`algorithm.config`, and this file only translates one
``PipelineSettings`` into adapters, trackers and the pipeline.
"""

from __future__ import annotations

from algorithm.config import PipelineSettings
from algorithm.perception.detect_models import DetectionAdapter
from algorithm.perception.pipeline import ObservationPipeline
from algorithm.perception.segment_model import ViaductSegmentAdapter
from algorithm.perception.tracking import (
    LINE_LABELS,
    VEHICLE_LABELS,
    ByteTracker,
)
from algorithm.perception.viaduct_tracker import ViaductTracker


def build_pipeline(settings: PipelineSettings) -> ObservationPipeline:
    """Wire adapters, trackers and pipeline from one settings object."""
    vehicle_detection = DetectionAdapter(
        settings.vehicle_weight,
        device=settings.device,
        imgsz=settings.detect_imgsz,
        confidence=settings.vehicle_confidence,
    )
    line_detection = DetectionAdapter(
        settings.line_weight,
        device=settings.device,
        imgsz=settings.line_imgsz,
        confidence=settings.line_confidence,
        # A swapped checkpoint must fail at wiring time, not silently track junk.
        expected_labels=(settings.line_label,),
    )
    tracker = ByteTracker(
        # Fixed by design: identities are only issued for bus/car/truck.
        labels=VEHICLE_LABELS,
        detection_threshold=settings.vehicle_confidence,
        # Every detection already clears the adapter threshold, so ByteTrack's
        # low-score stage is intentionally collapsed onto the same value.
        low_threshold=settings.vehicle_confidence,
        max_lost_ms=settings.track_max_lost_ms,
        match_threshold=settings.track_match_threshold,
    )
    line_tracker = ByteTracker(
        labels=LINE_LABELS,
        detection_threshold=settings.line_confidence,
        low_threshold=settings.line_confidence,
        max_lost_ms=settings.track_max_lost_ms,
        match_threshold=settings.track_match_threshold,
    )
    return ObservationPipeline(
        viaduct_segment=ViaductSegmentAdapter(
            settings.viaduct_weight,
            device=settings.device,
            imgsz=settings.segment_imgsz,
            confidence=settings.viaduct_confidence,
            expected_label=settings.viaduct_label,
        ),
        vehicle_detection=vehicle_detection,
        vehicle_tracker=tracker,
        line_detection=line_detection,
        line_tracker=line_tracker,
        viaduct_tracker=ViaductTracker(),
        mask_min_confidence=settings.mask_min_confidence,
    )
