"""Congestion labelling: one pure function of density and mean speed.

Split out of :class:`~algorithm.domain.viaduct_stats.ViaductStatsRuntime` so the
threshold policy can be read, tested and tuned without masks, observations or a
speed state machine: the label is a heuristic over two measurements, not another
measurement, and keeping it a free function makes that boundary obvious.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite


class CongestionLevel(StrEnum):
    """Rough congestion label for one deck; ``str`` value is the JSON payload."""

    CLEAR = "clear"
    MODERATE = "moderate"
    SEVERE = "severe"


@dataclass(frozen=True, slots=True)
class CongestionThresholds:
    """Tuning for :func:`classify_congestion`, validated once at construction."""

    clear_density: float = 0.08
    clear_speed_kmh: float = 30.0
    severe_speed_kmh: float = 12.0

    def __post_init__(self) -> None:
        if not isfinite(self.clear_density) or not 0.0 <= self.clear_density <= 1.0:
            raise ValueError("clear_density must be finite and within [0, 1]")
        if not isfinite(self.clear_speed_kmh) or self.clear_speed_kmh <= 0.0:
            raise ValueError("clear_speed_kmh must be finite and positive")
        if not isfinite(self.severe_speed_kmh) or self.severe_speed_kmh <= 0.0:
            raise ValueError("severe_speed_kmh must be finite and positive")
        if self.severe_speed_kmh >= self.clear_speed_kmh:
            raise ValueError("severe_speed_kmh must be below clear_speed_kmh")


def classify_congestion(
    density: float,
    mean_speed_mps: float | None,
    thresholds: CongestionThresholds,
) -> CongestionLevel:
    """Label a deck: crowd first, then speed; no measurable speed is clear.

    ``mean_speed_mps is None`` means "no measurable car", so no label is invented
    from density alone; a deck below ``clear_density`` is never congested because
    one slow car is not a jam.
    """
    if mean_speed_mps is None:
        return CongestionLevel.CLEAR
    if density < thresholds.clear_density:
        return CongestionLevel.CLEAR
    speed_kmh = mean_speed_mps * 3.6
    if speed_kmh >= thresholds.clear_speed_kmh:
        return CongestionLevel.CLEAR
    if speed_kmh <= thresholds.severe_speed_kmh:
        return CongestionLevel.SEVERE
    return CongestionLevel.MODERATE
