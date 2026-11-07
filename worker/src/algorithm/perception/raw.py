"""SDK-neutral values that keep torch objects out of the pipeline.

Adapters convert ultralytics ``Results`` into :class:`RawDetection` here; the
pipeline converts raw detections into domain :class:`DetectedObject` values.
Everything below the adapters is numpy/domain only.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any

from algorithm.domain.geometry import Box
from algorithm.domain.observations import DetectedObject


def sdk_to_list(value: Any) -> list[Any]:
    """Detach an SDK tensor (GPU or CPU) into a plain Python list."""
    if value is None:
        return []
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        converted = tolist()
        return converted if isinstance(converted, list) else list(converted)
    return list(value)


def clamp_unit(value: float) -> float:
    """Clamp a detector coordinate into the normalized unit square."""
    return max(0.0, min(1.0, value))


@dataclass(frozen=True, slots=True)
class RawDetection:
    """One detector row before it becomes a domain object.

    Coordinates are normalized ``(x1, y1, x2, y2)`` and already clamped to the
    unit square; a row with an empty or inverted box is rejected here rather than
    silently dropped later.
    """

    label: str
    confidence: float
    box: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and within [0, 1]")
        if len(self.box) != 4:
            raise ValueError("box must hold four normalized coordinates")
        x1, y1, x2, y2 = self.box
        if not all(isfinite(value) for value in self.box):
            raise ValueError("box coordinates must be finite")
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise ValueError("box must be ordered and inside the unit square")

    def to_detected(self) -> DetectedObject:
        return DetectedObject(
            label=self.label,
            confidence=self.confidence,
            box=Box(*self.box),
        )


def verify_weight_file(path: Path, expected_sha256: str | None = None) -> Path:
    """Fail fast on a missing or unexpected model file."""
    if not path.is_file():
        raise FileNotFoundError(f"model weight does not exist: {path}")
    if expected_sha256 is None:
        return path
    if len(expected_sha256) != 64:
        raise ValueError("expected SHA-256 must contain 64 hexadecimal characters")
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected_sha256.lower():
        raise ValueError(
            f"model weight SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return path
