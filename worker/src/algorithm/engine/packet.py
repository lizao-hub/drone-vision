"""One frame plus its source metadata: the unit the engine consumes.

It is produced by the runtime sources (``VideoFileSource``, ``LiveSource``) and
consumed by :class:`algorithm.engine.pipeline.MultiProcessPipeline`, so it
belongs to the engine contract rather than to the runtime that happens to build
it today.
"""

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


class FrameGeometryChanged(ValueError):
    """A frame is not the size the engine was started with.

    The engine fixes its buffers when it starts -- it allocates one shared-memory
    slot per in-flight frame -- so it cannot accept a different resolution.  It
    raises this instead of failing deep inside a worker, so the caller can restart
    it; ``relay`` catches it and rebuilds the pipeline at the new size.
    """
