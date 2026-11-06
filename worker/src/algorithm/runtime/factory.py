"""Shared production composition: settings in, wired pipeline out.

Every entry point (CLI, batch regression, a live container loop) builds its
pipeline here, so the wiring is not duplicated in the runner packages.
"""

from __future__ import annotations

from pathlib import Path

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

# .../worker/src/algorithm/runtime/factory.py -> worker/
WORKER_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VIADUCT_WEIGHT = WORKER_ROOT / "models" / "viaduct" / "weights" / "best.pt"
DEFAULT_VEHICLE_WEIGHT = WORKER_ROOT / "models" / "vehicle" / "weights" / "best.pt"
DEFAULT_LINE_WEIGHT = WORKER_ROOT / "models" / "line" / "weights" / "best.pt"
DEFAULT_VIDEO = WORKER_ROOT / "videos" / "test_Video1.mp4"


def default_settings(
    *,
    viaduct_weight: Path = DEFAULT_VIADUCT_WEIGHT,
    vehicle_weight: Path = DEFAULT_VEHICLE_WEIGHT,
    line_weight: Path = DEFAULT_LINE_WEIGHT,
    device: int | str = 0,
    segment_imgsz: int = 960,
    detect_imgsz: int = 640,
    line_imgsz: int = 640,
    viaduct_confidence: float = 0.25,
    vehicle_confidence: float = 0.30,
    line_confidence: float = 0.30,
    car_width_m: float = 1.95,
) -> PipelineSettings:
    return PipelineSettings(
        viaduct_weight=viaduct_weight,
        vehicle_weight=vehicle_weight,
        line_weight=line_weight,
        device=device,
        segment_imgsz=segment_imgsz,
        detect_imgsz=detect_imgsz,
        line_imgsz=line_imgsz,
        viaduct_confidence=viaduct_confidence,
        vehicle_confidence=vehicle_confidence,
        line_confidence=line_confidence,
        car_width_m=car_width_m,
    )


def build_pipeline(settings: PipelineSettings, *, fps: float) -> ObservationPipeline:
    """Wire adapters, trackers and pipeline; the tracker buffers need the source fps."""
    if fps <= 0:
        raise ValueError("fps must be positive")
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
        fps=fps,
        match_threshold=settings.track_match_threshold,
    )
    line_tracker = ByteTracker(
        labels=LINE_LABELS,
        detection_threshold=settings.line_confidence,
        low_threshold=settings.line_confidence,
        max_lost_ms=settings.track_max_lost_ms,
        fps=fps,
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
