"""The viaduct's traffic picture: per-deck stats plus the frame-validity policy.

This is the orchestrator of the viaduct runtime layer, not the measurer:

    * :class:`~algorithm.domain.viaduct_stats.ViaductStatsRuntime` computes the
      per-deck density and mean speed;
    * :mod:`algorithm.domain.congestion` turns those two numbers into a label;
    * this class owns the frame-level policy: when a frame may report at all, and
      which reasons void it.

Vehicle speeds come from an injected :class:`VehicleSpeedRuntime`, and that state
machine is called **exactly once** per observation -- calling it twice with the
same timestamp restarts its tracks, because it treats a non-advancing timestamp
as a new appearance.

Policies:
    * the update is the deck measurement: no deck means ``valid=False``
      (``viaduct_unavailable``), and a missing scale or ground flow also voids
      the frame (the speed runtime's reasons pass through), so a deck never
      reports a density without the speed it is meant to carry;
    * ``density=0`` on a valid deck is a measurement (no car on that board);
    * ``mean_speed_mps=None`` on a valid deck means no *measurable* car was on it
      this frame -- no car at all, or only tracks too young for a speed; a
      stopped car is a measured ``0.0`` and never ``None``
      (:attr:`ViaductDeck.speed_state` exposes that distinction as a TriState);
    * ``congestion`` is a heuristic on the two measurements above, not another
      measurement: a deck without a measurable speed is reported ``clear``
      (inventing a level from density alone would be worse), and a deck below
      ``clear_density`` is never congested, because one slow car is not a jam;
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

from dataclasses import dataclass

from algorithm.domain.congestion import (
    CongestionLevel,
    CongestionThresholds,
    classify_congestion,
)
from algorithm.domain.observations import Observation
from algorithm.domain.vehicle_speed_runtime import (
    VehicleSpeedRuntime,
    VehicleSpeedUpdate,
)
from algorithm.domain.viaduct_stats import ViaductDeck, ViaductStatsRuntime

__all__ = [
    "CongestionLevel",
    "CongestionThresholds",
    "ViaductDeck",
    "ViaductRuntime",
    "ViaductStatsRuntime",
    "ViaductUpdate",
    "classify_congestion",
]


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
    ) -> None:
        if not isinstance(speed, VehicleSpeedRuntime):
            raise TypeError("speed must be a VehicleSpeedRuntime")
        self._speed = speed
        self.thresholds = CongestionThresholds(
            clear_density=clear_density,
            clear_speed_kmh=clear_speed_kmh,
            severe_speed_kmh=severe_speed_kmh,
        )
        self._stats = ViaductStatsRuntime(self.thresholds)

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

        return ViaductUpdate(
            timestamp_ms=timestamp_ms,
            valid=True,
            decks=self._stats.measure(decks, observation, speed),
            speed=speed,
        )
