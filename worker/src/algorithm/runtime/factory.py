"""Shared production composition: settings in, wired pipeline out.

Every entry point (CLI, batch regression, a live container loop) builds its
pipeline here, so the wiring is not duplicated in the runner packages.
"""

from __future__ import annotations

from pathlib import Path

from algorithm.config import MpPipelineSettings, PipelineSettings
from algorithm.engine.pipeline import MultiProcessPipeline
from algorithm.perception.assembly import ObservationAssembler
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
    track_max_lost_ms: int = 1000,
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
        track_max_lost_ms=track_max_lost_ms,
    )


def build_assembler(settings: PipelineSettings, *, fps: float) -> ObservationAssembler:
    """Trackers only: no model handle, so no GPU is touched in this process."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    vehicle_tracker = ByteTracker(
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
    return ObservationAssembler(
        vehicle_tracker=vehicle_tracker,
        line_tracker=line_tracker,
        viaduct_tracker=ViaductTracker(),
    )


def build_pipeline(
    settings: PipelineSettings,
    mp_settings: MpPipelineSettings,
    *,
    fps: float,
) -> MultiProcessPipeline:
    """Wire the multi-process pipeline; the parent loads no model and runs no CUDA.

    The assembler is built here and handed in, so the engine never imports this
    module: the composition root depends on the engine, not the other way round.
    """
    return MultiProcessPipeline(
        settings,
        mp_settings,
        fps=fps,
        assembler=build_assembler(settings, fps=fps),
    )
