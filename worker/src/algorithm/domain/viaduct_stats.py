"""Per-deck geometry: density and mean vehicle speed.

This is the measuring half of the viaduct runtime.  It owns no frame-validity
policy and no congestion policy:

    density(deck)    = |(union of vehicle boxes) ∩ deck mask| / |deck mask|
    mean_speed(deck) = this frame's mean vehicle ground speed over the mature
                       vehicles whose box centre lies on that deck

Validity (which frame may report at all) belongs to
:class:`~algorithm.domain.viaduct_runtime.ViaductRuntime`; the congestion label
belongs to :mod:`algorithm.domain.congestion`.  Deck identity comes from
``Observation.viaduct_tracks``: :class:`ViaductTracker` already matched the
instances across frames, so nothing here matches or keeps deck state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np

from algorithm.domain.congestion import (
    CongestionLevel,
    CongestionThresholds,
    classify_congestion,
)
from algorithm.domain.geometry import Box
from algorithm.domain.observations import DetectedObject, Observation, TrackedObject
from algorithm.domain.tri_state import TriState
from algorithm.domain.vehicle_speed_runtime import VehicleSpeedUpdate


@dataclass(frozen=True, slots=True)
class ViaductDeck:
    """One viaduct instance's density, mean speed and congestion for this frame."""

    deck_id: str
    density: float
    mean_speed_mps: float | None
    vehicles: int
    congestion: CongestionLevel

    def __post_init__(self) -> None:
        if not isinstance(self.deck_id, str) or not self.deck_id:
            raise ValueError("deck_id must be a non-empty string")
        if not isfinite(self.density) or not 0.0 <= self.density <= 1.0:
            raise ValueError("density must be finite and within [0, 1]")
        if self.mean_speed_mps is not None and (
            not isfinite(self.mean_speed_mps) or self.mean_speed_mps < 0.0
        ):
            raise ValueError("mean_speed_mps must be finite and non-negative")
        if (
            isinstance(self.vehicles, bool)
            or not isinstance(self.vehicles, int)
            or self.vehicles < 0
        ):
            raise ValueError("vehicles must be a non-negative integer")
        if not isinstance(self.congestion, CongestionLevel):
            raise TypeError("congestion must be a CongestionLevel")

    @property
    def mean_speed_kmh(self) -> float | None:
        return None if self.mean_speed_mps is None else self.mean_speed_mps * 3.6

    @property
    def speed_state(self) -> TriState:
        """``UNKNOWN`` no measurable car, ``FALSE`` measured standstill, ``TRUE`` moving."""
        if self.mean_speed_mps is None:
            return TriState.UNKNOWN
        return TriState.FALSE if self.mean_speed_mps == 0.0 else TriState.TRUE


class ViaductStatsRuntime:
    """Measures every deck of one frame; the caller decides whether it may."""

    def __init__(self, thresholds: CongestionThresholds | None = None) -> None:
        self.thresholds = thresholds or CongestionThresholds()

    def measure(
        self,
        decks: Sequence[TrackedObject],
        observation: Observation,
        speed: VehicleSpeedUpdate,
    ) -> tuple[ViaductDeck, ...]:
        """Density, mean speed and congestion for each deck, ordered by ``deck_id``."""
        ordered = sorted(decks, key=lambda item: item.track_id)
        owners = _owners(ordered, observation.vehicle_tracks)
        speeds: dict[str, list[float]] = {}
        for item in speed.speeds:
            deck_id = owners.get(item.track_id)
            if deck_id is not None:
                speeds.setdefault(deck_id, []).append(item.speed_mps)
        return tuple(
            self._deck(deck, observation.vehicle, speeds.get(deck.track_id, ()))
            for deck in ordered
        )

    def _deck(
        self,
        deck: TrackedObject,
        vehicles: Sequence[DetectedObject],
        speeds: Sequence[float],
    ) -> ViaductDeck:
        assert deck.mask is not None  # guaranteed by Observation
        owned, covered_px, mask_px = _measure_deck(deck.mask.data, vehicles)
        density = covered_px / mask_px if mask_px else 0.0
        mean_speed_mps = sum(speeds) / len(speeds) if speeds else None
        return ViaductDeck(
            deck_id=deck.track_id,
            density=density,
            mean_speed_mps=mean_speed_mps,
            vehicles=owned,
            congestion=classify_congestion(density, mean_speed_mps, self.thresholds),
        )


# -- helpers ---------------------------------------------------------------
def _measure_deck(
    mask: np.ndarray, vehicles: Sequence[DetectedObject]
) -> tuple[int, int, int]:
    """``(owned boxes, covered pixels, mask pixels)`` for one deck."""
    height, width = mask.shape
    canvas = np.zeros(mask.shape, dtype=np.bool_)
    owned = 0
    for item in vehicles:
        if not _owns(mask, item.box):
            continue
        owned += 1
        x1 = min(max(int(item.box.x1 * width), 0), width)
        y1 = min(max(int(item.box.y1 * height), 0), height)
        x2 = min(max(int(item.box.x2 * width), 0), width)
        y2 = min(max(int(item.box.y2 * height), 0), height)
        if x2 > x1 and y2 > y1:
            canvas[y1:y2, x1:x2] = True
    mask_px = int(np.count_nonzero(mask))
    covered_px = int(np.count_nonzero(canvas & mask))
    return owned, covered_px, mask_px


def _owners(
    decks: Sequence[TrackedObject], tracks: Sequence[TrackedObject]
) -> Mapping[str, str]:
    """Map each vehicle track to the first deck (ordered by id) holding its centre."""
    owners: dict[str, str] = {}
    for track in tracks:
        for deck in decks:
            assert deck.mask is not None  # guaranteed by Observation
            if _owns(deck.mask.data, track.box):
                owners[track.track_id] = deck.track_id
                break
    return owners


def _owns(mask: np.ndarray, box: Box) -> bool:
    """Whether the box centre pixel lies on this deck."""
    height, width = mask.shape
    center_x = (box.x1 + box.x2) / 2 * width
    center_y = (box.y1 + box.y2) / 2 * height
    x = min(max(int(center_x), 0), width - 1)
    y = min(max(int(center_y), 0), height - 1)
    return bool(mask[y, x])
