"""Pipeline tuning knobs.  Plain immutable data: no I/O, no model handles."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PipelineSettings:
    viaduct_weight: Path
    vehicle_weight: Path
    line_weight: Path
    device: int | str = 0
    segment_imgsz: int = 960
    detect_imgsz: int = 640
    line_imgsz: int = 640
    viaduct_confidence: float = 0.25
    viaduct_label: str = "viaduct"
    vehicle_confidence: float = 0.30
    line_confidence: float = 0.30
    line_label: str = "line"
    mask_min_confidence: float = 0.25
    track_max_lost_ms: int = 1000
    track_match_threshold: float = 0.80
    # Reference width of a "car" for the vehicle-speed scale.  Chinese passenger
    # cars are ~1.8-1.9 m; 2.2 m would bias the speed high by ~15%.
    car_width_m: float = 1.95

    def __post_init__(self) -> None:
        for name in ("viaduct_weight", "vehicle_weight", "line_weight"):
            if not isinstance(getattr(self, name), Path):
                raise TypeError(f"{name} must be a pathlib.Path")
        if isinstance(self.device, bool) or not isinstance(self.device, (int, str)):
            raise TypeError("device must be an int index or a string such as 'cpu'")
        for name in ("segment_imgsz", "detect_imgsz", "line_imgsz"):
            if isinstance(getattr(self, name), bool) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "viaduct_confidence",
            "vehicle_confidence",
            "line_confidence",
            "mask_min_confidence",
        ):
            value = getattr(self, name)
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1]")
        if not isfinite(self.track_match_threshold) or not 0.0 < self.track_match_threshold <= 1.0:
            raise ValueError("track_match_threshold must be within (0, 1]")
        if self.track_max_lost_ms <= 0:
            raise ValueError("track_max_lost_ms must be positive")
        if not self.viaduct_label.strip():
            raise ValueError("viaduct_label must be a non-empty string")
        if not self.line_label.strip():
            raise ValueError("line_label must be a non-empty string")
        if not 0.0 < self.vehicle_confidence <= 1.0:
            raise ValueError("vehicle_confidence must be within (0, 1]")
        if not 0.0 < self.line_confidence <= 1.0:
            raise ValueError("line_confidence must be within (0, 1]")
        if not isfinite(self.car_width_m) or self.car_width_m <= 0.0:
            raise ValueError("car_width_m must be finite and positive")


@dataclass(frozen=True, slots=True)
class MpPipelineSettings:
    """Process layout of the multi-process pipeline (``engine.pipeline``).

    Each entry is a *physical* GPU index as reported by ``nvidia-smi``; every
    worker pins itself to exactly one of them through ``CUDA_VISIBLE_DEVICES``
    before torch is imported.  ``segment_gpus`` may hold several cards: one
    segment worker is started per card, which is the only stage worth fanning out
    because it is the longest one.

    ``window`` is the number of frames allowed to be in flight, i.e. the shared
    memory ring size and the throughput/latency knob: the pipeline keeps that
    many frames queued and shares memory for each (roughly 80 MB per 1080p slot,
    66 MB of it instance masks at the default capacity).

    ``mask_capacity`` caps how many segmentation instances fit in one slot.  A
    frame with more decks than that is reported as a segment adapter failure, so
    the cap can never silently drop a deck.  The default covers what the shipped
    weights actually emit: on a 1080p viaduct clip the segmenter reports 2
    instances on most frames but up to 13 on some (see
    ``scripts/count_instances.py``), so 8 would have invalidated a fifth of the
    frames.
    """

    segment_gpus: tuple[int, ...] = (0,)
    vehicle_gpu: int = 1
    line_gpu: int = 2
    window: int = 4
    mask_capacity: int = 32
    start_timeout_s: float = 240.0

    def __post_init__(self) -> None:
        if not isinstance(self.segment_gpus, tuple) or not self.segment_gpus:
            raise ValueError("segment_gpus must be a non-empty tuple of GPU ids")
        for name, value in (
            ("segment_gpus", self.segment_gpus),
            ("vehicle_gpu", (self.vehicle_gpu,)),
            ("line_gpu", (self.line_gpu,)),
        ):
            for gpu in value:
                if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
                    raise ValueError(f"{name} must hold non-negative GPU ids")
        for name in ("window", "mask_capacity"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isfinite(self.start_timeout_s) or self.start_timeout_s <= 0.0:
            raise ValueError("start_timeout_s must be finite and positive")

    @property
    def gpus(self) -> tuple[int, ...]:
        """Every physical GPU the pipeline will touch, duplicated ids included."""
        return (*self.segment_gpus, self.vehicle_gpu, self.line_gpu)
