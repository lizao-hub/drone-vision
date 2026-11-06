"""Frame-level facts.  No decisions, no output shaping."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from algorithm.domain.geometry import Box, Mask


def _validate_identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _validate_confidence(value: float) -> None:
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("confidence must be finite and within [0, 1]")


def _validate_non_negative_ms(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_tuple(name: str, value: object) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be an immutable tuple")


@dataclass(frozen=True, slots=True)
class DetectedObject:
    """One model output for this frame: a box plus an optional instance mask.

    The segmenter fills ``mask`` with that instance's pixels; the detector
    leaves it ``None``.
    """

    label: str
    confidence: float
    box: Box
    mask: Mask | None = None

    def __post_init__(self) -> None:
        _validate_identifier("label", self.label)
        _validate_confidence(self.confidence)
        if self.mask is not None and not isinstance(self.mask, Mask):
            raise TypeError("mask must be a Mask or None")


@dataclass(frozen=True, slots=True)
class TrackedObject:
    """An object with an identity; deck tracks also carry their instance mask."""

    track_id: str
    label: str
    confidence: float
    box: Box
    mask: Mask | None = None

    def __post_init__(self) -> None:
        _validate_identifier("track_id", self.track_id)
        _validate_identifier("label", self.label)
        _validate_confidence(self.confidence)
        if self.mask is not None and not isinstance(self.mask, Mask):
            raise TypeError("mask must be a Mask or None")


@dataclass(frozen=True, slots=True)
class Diagnostics:
    segment_ms: float
    detect_ms: float
    line_ms: float
    processing_ms: float

    def __post_init__(self) -> None:
        for name, value in (
            ("segment_ms", self.segment_ms),
            ("detect_ms", self.detect_ms),
            ("line_ms", self.line_ms),
            ("processing_ms", self.processing_ms),
        ):
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class Observation:
    timestamp_ms: int
    frame_sequence: int
    generation: int
    frame_valid: bool
    viaduct: tuple[DetectedObject, ...]
    viaduct_tracks: tuple[TrackedObject, ...]
    vehicle: tuple[DetectedObject, ...]
    vehicle_tracks: tuple[TrackedObject, ...]
    line_tracks: tuple[TrackedObject, ...]
    unknown_reasons: tuple[str, ...]
    diagnostics: Diagnostics

    def __post_init__(self) -> None:
        _validate_non_negative_ms("timestamp_ms", self.timestamp_ms)
        _validate_non_negative_ms("frame_sequence", self.frame_sequence)
        _validate_non_negative_ms("generation", self.generation)
        if not isinstance(self.frame_valid, bool):
            raise TypeError("frame_valid must be a boolean")
        _validate_tuple("viaduct", self.viaduct)
        _validate_tuple("viaduct_tracks", self.viaduct_tracks)
        _validate_tuple("vehicle", self.vehicle)
        _validate_tuple("vehicle_tracks", self.vehicle_tracks)
        _validate_tuple("line_tracks", self.line_tracks)
        _validate_tuple("unknown_reasons", self.unknown_reasons)
        for reason in self.unknown_reasons:
            _validate_identifier("unknown_reason", reason)
        if any(item.mask is None for item in self.viaduct_tracks):
            raise ValueError("viaduct tracks must carry the deck mask")
        if not self.frame_valid and (
            self.viaduct
            or self.viaduct_tracks
            or self.vehicle
            or self.vehicle_tracks
            or self.line_tracks
        ):
            raise ValueError("an invalid frame cannot carry observations")
        if not self.frame_valid and not self.unknown_reasons:
            raise ValueError("an invalid frame must record at least one reason")
