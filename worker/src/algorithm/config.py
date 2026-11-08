"""Pipeline settings: the one place every tuning value and bundled path lives.

Plain immutable data: no I/O, no model handles.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

# .../worker/src/algorithm/config.py -> worker/
WORKER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIADUCT_WEIGHT = (
    WORKER_ROOT / "models" / "viaduct_seg" / "yolo26m-seg_1080" / "weights" / "best.pt"
)
DEFAULT_VEHICLE_WEIGHT = (
    WORKER_ROOT / "models" / "vehicle_detect" / "yolo26m_1080" / "weights" / "best.pt"
)
DEFAULT_LINE_WEIGHT = (
    WORKER_ROOT / "models" / "line_detect" / "yolo26m_1080" / "weights" / "best.pt"
)


@dataclass(frozen=True, slots=True)
class PipelineSettings:
    viaduct_weight: Path
    vehicle_weight: Path
    line_weight: Path
    # Where the batch runner writes the annotated mp4 and the JSONL.
    project: Path = WORKER_ROOT / "output"
    device: int | str = 0
    # All three models were trained at 1280; a 1080p 16:9 frame becomes 1280x736
    # under rect letterboxing.  Keeping inference at the training size keeps the
    # apparent object scale identical between training and deployment.
    segment_imgsz: int = 1280
    detect_imgsz: int = 1280
    line_imgsz: int = 1280
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
        for name in ("viaduct_weight", "vehicle_weight", "line_weight", "project"):
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


# The only deployment-dependent part of the settings; every knob keeps the
# default declared above.
DEFAULT_SETTINGS = PipelineSettings(
    viaduct_weight=DEFAULT_VIADUCT_WEIGHT,
    vehicle_weight=DEFAULT_VEHICLE_WEIGHT,
    line_weight=DEFAULT_LINE_WEIGHT,
)
