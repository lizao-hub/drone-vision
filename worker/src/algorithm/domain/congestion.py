"""Per-deck congestion label: a pure classifier plus a dwell-filtered state machine.

The raw level (:meth:`CongestionStabilizer.classify`) is a pure function of one
deck's density and mean speed.  On its own it flickers: a measurement hovering
at a threshold, or a single frame without a measurable speed, flips the label
frame by frame.  This module keeps that function and turns it into a reported
label:

    * cold start: a deck's first observation is reported raw, with no dwell, so
      a deck never shows an invented starting level;
    * dwell: a different raw label must persist for ``dwell_ms`` of observation
      time before it replaces the reported one.  The clock is the observation
      timestamp, not a frame count, so the offline stride and the live relay
      get the same latency;
    * missing speed: a frame where the deck has no measurable speed (no car, or
      no mature track) keeps the previous label instead of falling back to
      ``clear`` -- a missing measurement is not evidence of free flow.  After
      ``max_hold_ms`` of continuous absence the raw label (``clear``) takes
      over, so an empty deck still clears;
    * expiry: a deck unseen for ``state_ttl_ms`` is forgotten, so a recycled
      deck id starts cold instead of inheriting a stale label.

The dwell filter is not threshold hysteresis: it does not change the numbers,
only how fast the label may follow them.  A hysteresis band (enter at 8%,
leave at 6%) belongs to the raw classifier and is a separate change.
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


@dataclass(slots=True)
class _DeckState:
    """What the stabiliser remembers about one deck between observations."""

    level: CongestionLevel
    last_seen_ms: int
    candidate: CongestionLevel | None = None
    candidate_since_ms: int = 0
    missing_since_ms: int | None = None


class CongestionStabilizer:
    """Maps raw per-frame measurements to a stable per-deck congestion label."""

    def __init__(
        self,
        *,
        clear_density: float = 0.08,
        clear_speed_kmh: float = 30.0,
        severe_speed_kmh: float = 12.0,
        dwell_ms: int = 700,
        state_ttl_ms: int = 3000,
        max_hold_ms: int = 3000,
    ) -> None:
        if not isfinite(clear_density) or not 0.0 <= clear_density <= 1.0:
            raise ValueError("clear_density must be finite and within [0, 1]")
        if not isfinite(clear_speed_kmh) or clear_speed_kmh <= 0.0:
            raise ValueError("clear_speed_kmh must be finite and positive")
        if not isfinite(severe_speed_kmh) or severe_speed_kmh <= 0.0:
            raise ValueError("severe_speed_kmh must be finite and positive")
        if severe_speed_kmh >= clear_speed_kmh:
            raise ValueError("severe_speed_kmh must be below clear_speed_kmh")
        if isinstance(dwell_ms, bool) or not isinstance(dwell_ms, int) or dwell_ms <= 0:
            raise ValueError("dwell_ms must be a positive integer")
        if (
            isinstance(state_ttl_ms, bool)
            or not isinstance(state_ttl_ms, int)
            or state_ttl_ms <= 0
        ):
            raise ValueError("state_ttl_ms must be a positive integer")
        if (
            isinstance(max_hold_ms, bool)
            or not isinstance(max_hold_ms, int)
            or max_hold_ms <= 0
        ):
            raise ValueError("max_hold_ms must be a positive integer")
        self.clear_density = clear_density
        self.clear_speed_kmh = clear_speed_kmh
        self.severe_speed_kmh = severe_speed_kmh
        self.dwell_ms = dwell_ms
        self.state_ttl_ms = state_ttl_ms
        self.max_hold_ms = max_hold_ms
        self._states: dict[str, _DeckState] = {}

    # -- raw classification ---------------------------------------------------
    def classify(
        self, density: float, mean_speed_mps: float | None
    ) -> CongestionLevel:
        """The unfiltered per-frame label; also the cold-start label.

        ``None`` is a *missing* speed and reports ``clear`` -- at cold start
        there is no previous label to hold, and inventing one from density
        alone would be worse.  A measured ``0.0`` is a stopped car and goes
        through the severe branch like any other speed.
        """
        if mean_speed_mps is None:
            return CongestionLevel.CLEAR
        if density < self.clear_density:
            return CongestionLevel.CLEAR
        speed_kmh = mean_speed_mps * 3.6
        if speed_kmh >= self.clear_speed_kmh:
            return CongestionLevel.CLEAR
        if speed_kmh <= self.severe_speed_kmh:
            return CongestionLevel.SEVERE
        return CongestionLevel.MODERATE

    # -- stabilised label -----------------------------------------------------
    def update(
        self,
        deck_id: str,
        timestamp_ms: int,
        density: float,
        mean_speed_mps: float | None,
    ) -> CongestionLevel:
        """One observation of one deck; returns the label to report.

        The caller must pass the deck's observations in time order and must not
        call this on frames that carry no deck (frozen frames keep the state
        untouched, like the other runtimes).
        """
        if not isinstance(deck_id, str) or not deck_id:
            raise ValueError("deck_id must be a non-empty string")
        if isinstance(timestamp_ms, bool) or not isinstance(timestamp_ms, int):
            raise TypeError("timestamp_ms must be an integer")
        if timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative")

        raw = self.classify(density, mean_speed_mps)
        state = self._states.get(deck_id)
        if state is not None and timestamp_ms - state.last_seen_ms > self.state_ttl_ms:
            state = None
        if state is None:
            self._states[deck_id] = _DeckState(raw, timestamp_ms)
            self._prune(timestamp_ms)
            return raw

        if timestamp_ms <= state.last_seen_ms:
            # A repeated or rewound clock proves nothing about persistence;
            # hold the label and leave the candidate timer alone.
            return state.level
        state.last_seen_ms = timestamp_ms

        if mean_speed_mps is None:
            # Step 0: a missing speed holds the previous label, but only for so
            # long -- an empty deck must eventually report clear.
            if state.missing_since_ms is None:
                state.missing_since_ms = timestamp_ms
            state.candidate = None
            if timestamp_ms - state.missing_since_ms >= self.max_hold_ms:
                state.level = raw
            return state.level
        state.missing_since_ms = None

        if raw is state.level:
            state.candidate = None
            return state.level
        if state.candidate is not raw:
            state.candidate = raw
            state.candidate_since_ms = timestamp_ms
        if timestamp_ms - state.candidate_since_ms >= self.dwell_ms:
            state.level = raw
            state.candidate = None
        return state.level

    # -- internals ------------------------------------------------------------
    def _prune(self, timestamp_ms: int) -> None:
        expired = [
            deck_id
            for deck_id, state in self._states.items()
            if timestamp_ms - state.last_seen_ms > self.state_ttl_ms
        ]
        for deck_id in expired:
            del self._states[deck_id]
