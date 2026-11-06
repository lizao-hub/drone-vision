"""Deck-mask post-processing: union and blackout.

Pure numpy/OpenCV image operations.  No model is loaded here and nothing is
written to disk, so the whole module is cheap to unit-test.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from algorithm.domain.geometry import Mask
from algorithm.domain.observations import DetectedObject


def union_mask(
    segmentations: Sequence[DetectedObject],
    *,
    height: int,
    width: int,
) -> Mask:
    """Union every instance into one frame-aligned deck mask.

    The viaduct model normally emits two instances (one per carriageway); taking
    only the most confident one would black out half of the road.  Confidence
    filtering is the caller's job: the pipeline keeps the same filtered tuple in
    ``Observation.viaduct``, so this function never filters a second time.
    """
    if height <= 0 or width <= 0:
        raise ValueError("frame geometry must be positive")
    combined = np.zeros((height, width), dtype=np.bool_)
    for item in segmentations:
        if item.mask is None:
            raise ValueError("viaduct segmentations must carry a mask")
        if item.mask.data.shape != combined.shape:
            raise ValueError(
                f"segment mask shape {item.mask.data.shape} != frame {(height, width)}"
            )
        combined |= item.mask.data
    return Mask(combined)


def apply_mask(frame: np.ndarray, mask: Mask) -> np.ndarray:
    """Copy the frame and blacken everything outside the deck mask."""
    if frame.ndim != 3:
        raise ValueError("frame must be an H x W x C array")
    if frame.shape[:2] != (mask.height, mask.width):
        raise ValueError(
            f"mask geometry {(mask.height, mask.width)} != frame {frame.shape[:2]}"
        )
    masked = frame.copy()
    masked[~mask.data] = 0
    return masked
