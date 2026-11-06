"""Immutable geometry types.  No torch, no OpenCV, no I/O."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np


@dataclass(frozen=True, slots=True)
class Box:
    """Axis-aligned box in full-frame normalized coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        coordinates = (self.x1, self.y1, self.x2, self.y2)
        if not all(isfinite(value) and 0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError("box coordinates must be finite and within [0, 1]")
        if not (self.x1 < self.x2 and self.y1 < self.y2):
            raise ValueError("box coordinates must be ordered as x1 < x2 and y1 < y2")

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def intersection(self, other: "Box") -> float:
        width = min(self.x2, other.x2) - max(self.x1, other.x1)
        height = min(self.y2, other.y2) - max(self.y1, other.y1)
        if width <= 0.0 or height <= 0.0:
            return 0.0
        return width * height

    def iou(self, other: "Box") -> float:
        overlap = self.intersection(other)
        union = self.area + other.area - overlap
        return overlap / union if union else 0.0

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


@dataclass(frozen=True, slots=True, eq=False)
class Mask:
    """Binary mask aligned with the source frame (True = viaduct deck)."""

    data: np.ndarray

    def __post_init__(self) -> None:
        # Copy first: freezing must never mutate the caller's array.
        array = np.array(self.data, dtype=np.bool_, copy=True)
        if array.ndim != 2:
            raise ValueError("mask must be a 2-D boolean array")
        array.flags.writeable = False
        object.__setattr__(self, "data", array)

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])
