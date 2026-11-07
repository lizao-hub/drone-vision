"""Pipeline profile: the tunables that used to be command-line flags.

Model sizes, confidences, weight paths and the car-width prior are deployment
data now.  They live in ``worker/config/pipeline.yaml`` and are validated when
loaded, so every entry point (batch, relay, a future service) builds the same
settings from the same file and a run cannot silently deviate from the tuning the
weights were validated with.

Loading is strict on purpose: unknown keys, missing sections and out-of-range
values fail at startup instead of being ignored.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
)

from algorithm.config import PipelineSettings

# .../worker/src/algorithm/runtime/profile.py -> worker/
_WORKER_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROFILE = _WORKER_ROOT / "config" / "pipeline.yaml"


class StrictModel(BaseModel):
    """Reject unknown keys and forbid mutation: a profile is read, not edited."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelProfile(StrictModel):
    """One model: where the weights are and how the adapter runs them."""

    weights: str = Field(min_length=1)
    imgsz: PositiveInt
    confidence: float = Field(gt=0.0, le=1.0)

    def weight_path(self, root: Path) -> Path:
        path = Path(self.weights)
        return path if path.is_absolute() else root / path


class TrackingProfile(StrictModel):
    """ByteTrack tuning shared by the vehicle and lane-line trackers."""

    max_lost_ms: PositiveInt = 1000
    match_threshold: float = Field(default=0.80, gt=0.0, le=1.0)


class PipelineProfile(StrictModel):
    """Everything that is not a per-run choice (source, project, device, stride)."""

    viaduct: ModelProfile
    vehicle: ModelProfile
    line: ModelProfile
    car_width_m: PositiveFloat = 1.95
    tracking: TrackingProfile = Field(default_factory=TrackingProfile)

    def to_settings(
        self,
        *,
        device: int | str,
        track_max_lost_ms: int | None = None,
    ) -> PipelineSettings:
        """Domain value object for the pipeline; ``device`` is per machine."""
        return PipelineSettings(
            viaduct_weight=self.viaduct.weight_path(_WORKER_ROOT),
            vehicle_weight=self.vehicle.weight_path(_WORKER_ROOT),
            line_weight=self.line.weight_path(_WORKER_ROOT),
            device=device,
            segment_imgsz=self.viaduct.imgsz,
            detect_imgsz=self.vehicle.imgsz,
            line_imgsz=self.line.imgsz,
            viaduct_confidence=self.viaduct.confidence,
            vehicle_confidence=self.vehicle.confidence,
            line_confidence=self.line.confidence,
            car_width_m=self.car_width_m,
            track_max_lost_ms=(
                self.tracking.max_lost_ms
                if track_max_lost_ms is None
                else track_max_lost_ms
            ),
            track_match_threshold=self.tracking.match_threshold,
        )


def load_profile(path: Path = DEFAULT_PROFILE) -> PipelineProfile:
    """Read and validate a profile file; raises on anything unexpected."""
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise TypeError(f"profile must be a YAML mapping: {path}")
    return PipelineProfile.model_validate(document)
