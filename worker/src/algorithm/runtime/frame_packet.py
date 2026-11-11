"""One frame plus its source metadata: the unit the runtime layers exchange.

A packet records where the frame came from and when, so downstream code never has
to reach back into the capture.  ``inference_eligible`` lets a source keep the
timeline dense (every decoded frame) while inference only runs on sampled frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FramePacket:
    source_id: str
    session_id: str
    frame_index: int
    timestamp_ms: int
    image: np.ndarray
    original_width: int
    original_height: int
    inference_eligible: bool = True

    def __post_init__(self) -> None:
        if not self.source_id or not self.session_id:
            raise ValueError("source_id and session_id are required")
        for name, value in (
            ("frame_index", self.frame_index),
            ("timestamp_ms", self.timestamp_ms),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.image, np.ndarray) or self.image.ndim != 3:
            raise ValueError("image must be an H x W x C ndarray")
        if self.original_width <= 0 or self.original_height <= 0:
            raise ValueError("original dimensions must be positive")
        if not isinstance(self.inference_eligible, bool):
            raise TypeError("inference_eligible must be a boolean")
        if (
            self.image.shape[1] != self.original_width
            or self.image.shape[0] != self.original_height
        ):
            raise ValueError("original dimensions must match image dimensions")
