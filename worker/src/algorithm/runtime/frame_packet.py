"""One frame plus its source metadata: the unit the runtime layers exchange."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FramePacket:
    frame: np.ndarray
    frame_index: int
    timestamp_ms: int
    fps: float

    def __post_init__(self) -> None:
        if not isinstance(self.frame, np.ndarray) or self.frame.ndim != 3:
            raise ValueError("frame must be an H x W x C ndarray")
        if self.frame_index < 0 or self.timestamp_ms < 0:
            raise ValueError("frame_index and timestamp_ms must be non-negative")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
