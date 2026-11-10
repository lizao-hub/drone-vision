"""Per-deck density and mean vehicle speed: the viaduct's own traffic picture.

A runtime-layer component consuming :class:`Observation` values.  It is the
composition point for the two per-deck measurements::

    density(deck)    = |(union of vehicle boxes) ∩ deck mask| / |deck mask|
    mean_speed(deck) = this frame's mean vehicle ground speed over the mature
                       vehicles whose box centre lies on that deck

The pair is reduced to a rough congestion label (:class:`CongestionLevel`), a
dashboard heuristic rather than a traffic-flow model::

    congestion(deck) = clear    when the deck is too empty to be a jam, or the
                                traffic flows (and, on a cold start, when no
                                measurable speed exists yet)
                       severe   when the deck is crowded and the speed has
                                collapsed
                       moderate otherwise

Deck identity comes from ``Observation.viaduct_tracks``: :class:`ViaductTracker`
already matched the instances across frames, so this runtime matches nothing.
The one piece of state is the per-deck label, kept by an injected
:class:`CongestionStabilizer` (dwell filter + missing-speed hold).  Vehicle
speeds come from an injected
:class:`VehicleSpeedRuntime`, and that state machine is called **exactly once**
per observation -- calling it twice with the same timestamp restarts its tracks,
because it treats a non-advancing timestamp as a new appearance.

Policies:
    * the update is the deck measurement: no deck means ``valid=False``
      (``viaduct_unavailable``), and a missing scale or ground flow also voids
      the frame (the speed runtime's reasons pass through), so a deck never
      reports a density without the speed it is meant to carry;
    * ``density=0`` on a valid deck is a measurement (no car on that board);
    * ``mean_speed_mps=None`` on a valid deck means no *measurable* car was on it
      this frame -- no car at all, or only tracks too young for a speed; a
      stopped car is a measured ``0.0`` and never ``None``;
    * ``congestion`` is a heuristic on the two measurements above, not another
      measurement: a deck below ``clear_density`` is never congested, because
      one slow car is not a jam, and a deck with no measurable speed is
      ``clear`` on a cold start (inventing a level from density alone would be
      worse) but keeps its previous label for up to ``max_hold_ms`` afterwards;
    * the reported label is dwell-filtered: a different raw label must persist
      for ``dwell_ms`` of observation time before it is reported, so a value
      hovering at a threshold no longer flickers frame by frame; the unfiltered
      label is still available from :meth:`CongestionStabilizer.classify`;
    * density counts every detected box (``Observation.vehicle``) while the mean
      speed only sees tracks that matured inside :class:`VehicleSpeedRuntime`,
      so ``vehicles`` can be larger than the number of cars behind the mean;
    * a box belongs to the deck holding its centre and contributes to that
      deck's coverage only: with overlapping masks the lowest ``deck_id`` wins,
      so one vehicle never enters a mean twice;
    * invalid and degraded frames produce no stats; freezing is
      :class:`VehicleSpeedRuntime`'s own policy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np

from algorithm.domain.congestion import CongestionLevel, CongestionStabilizer
from algorithm.domain.geometry import Box
from algorithm.domain.observations import DetectedObject, Observation, TrackedObject
from algorithm.domain.vehicle_speed_runtime import VehicleSpeedRuntime, VehicleSpeedUpdate


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


@dataclass(frozen=True, slots=True)
class ViaductUpdate:
    """One frame's per-deck stats plus the frame-level measurement state.

    ``speed`` is the per-vehicle update this frame was built from; it travels
    with the deck stats so a renderer or JSON writer gets the individual vehicle
    speeds without running a second :class:`VehicleSpeedRuntime`.
    """

    timestamp_ms: int
    valid: bool
    decks: tuple[ViaductDeck, ...]
    speed: VehicleSpeedUpdate
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.timestamp_ms, bool) or not isinstance(self.timestamp_ms, int):
            raise TypeError("timestamp_ms must be an integer")
        if self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative")
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be a boolean")
        if not isinstance(self.decks, tuple):
            raise TypeError("decks must be an immutable tuple")
        if not isinstance(self.speed, VehicleSpeedUpdate):
            raise TypeError("speed must be a VehicleSpeedUpdate")
        if not isinstance(self.reasons, tuple):
            raise TypeError("reasons must be an immutable tuple")
        if self.speed.timestamp_ms != self.timestamp_ms:
            raise ValueError("speed must belong to the same frame")
        if self.valid and not self.speed.valid:
            raise ValueError("a valid update requires a valid speed measurement")
        if not self.valid:
            if not self.reasons:
                raise ValueError("an invalid update must record at least one reason")
            if self.decks:
                raise ValueError("an invalid update cannot carry decks")


class ViaductRuntime:
    """Per-deck density and mean speed, one :class:`ViaductUpdate` per frame."""

    def __init__(
        self,
        *,
        speed: VehicleSpeedRuntime,
        clear_density: float = 0.08,
        clear_speed_kmh: float = 30.0,
        severe_speed_kmh: float = 12.0,
        dwell_ms: int = 700,
    ) -> None:
        if not isinstance(speed, VehicleSpeedRuntime):
            raise TypeError("speed must be a VehicleSpeedRuntime")
        self._speed = speed
        self._congestion = CongestionStabilizer(
            clear_density=clear_density,
            clear_speed_kmh=clear_speed_kmh,
            severe_speed_kmh=severe_speed_kmh,
            dwell_ms=dwell_ms,
        )

    def update(self, observation: Observation) -> ViaductUpdate:
        timestamp_ms = observation.timestamp_ms
        # Always advance the speed state machine first, even on frozen frames:
        # it owns the track expiry and its own freeze policy.
        speed = self._speed.update(observation)

        if not observation.frame_valid:
            return ViaductUpdate(timestamp_ms, False, (), speed, ("frame_invalid",))
        if observation.unknown_reasons:
            return ViaductUpdate(
                timestamp_ms, False, (), speed, observation.unknown_reasons
            )

        decks = [item for item in observation.viaduct_tracks if item.mask is not None]
        if not decks:
            return ViaductUpdate(timestamp_ms, False, (), speed, ("viaduct_unavailable",))

        if not speed.valid:
            return ViaductUpdate(timestamp_ms, False, (), speed, speed.reasons)

        # The segmenter's instance order is unstable; the deck ids are not.
        decks.sort(key=lambda item: item.track_id)
        owners = _owners(decks, observation.vehicle_tracks)

        speeds: dict[str, list[float]] = {}
        for item in speed.speeds:
            deck_id = owners.get(item.track_id)
            if deck_id is not None:
                speeds.setdefault(deck_id, []).append(item.speed_mps)

        return ViaductUpdate(
            timestamp_ms=timestamp_ms,
            valid=True,
            decks=tuple(
                self._deck_stats(
                    deck.track_id,
                    timestamp_ms,
                    deck.mask.data,
                    observation.vehicle,
                    speeds.get(deck.track_id, ()),
                )
                for deck in decks
            ),
            speed=speed,
        )

    def _deck_stats(
        self,
        deck_id: str,
        timestamp_ms: int,
        mask: np.ndarray,
        vehicles: Sequence[DetectedObject],
        speeds: Sequence[float],
    ) -> ViaductDeck:
        owned, covered_px, mask_px = _measure_deck(mask, vehicles)
        density = covered_px / mask_px if mask_px else 0.0
        mean_speed_mps = sum(speeds) / len(speeds) if speeds else None
        return ViaductDeck(
            deck_id=deck_id,
            density=density,
            mean_speed_mps=mean_speed_mps,
            vehicles=owned,
            congestion=self._congestion.update(
                deck_id, timestamp_ms, density, mean_speed_mps
            ),
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
) -> dict[str, str]:
    """Map each vehicle track to the first deck (ordered by id) holding its centre."""
    owners: dict[str, str] = {}
    for track in tracks:
        for deck in decks:
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
