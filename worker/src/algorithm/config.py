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
