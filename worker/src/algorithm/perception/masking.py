"""Deck-mask post-processing: black out everything off the deck.

One pure numpy/OpenCV operation: no model is loaded here and nothing is written
to disk, so the module is cheap to unit-test.

The blackout is memory-bandwidth bound at 1080p (a 6 MB copy plus a boolean
fancy-index write), so it goes through OpenCV's SIMD path instead: the union
mask is widened once and `copyTo` writes into a zeroed destination.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from algorithm.domain.observations import DetectedObject


def blackout_off_deck(
    frame: np.ndarray,
    segmentations: Sequence[DetectedObject],
) -> np.ndarray | None:
    """Return a masked copy of ``frame`` for the detection models.

    Everything outside the union of the viaduct instances is set to black; the
    input frame is never touched.  Every instance is merged because the model
    normally emits one per carriageway, and taking only the most confident one
    would black out half the road.

    Returns ``None`` when there is no deck to keep, i.e. no instances or only
    empty masks: the caller then has nothing to detect on.
    """
    if frame.ndim != 3:
        raise ValueError("frame must be an H x W x C array")
    height, width = int(frame.shape[0]), int(frame.shape[1])

    deck = np.zeros((height, width), dtype=np.bool_)
    for item in segmentations:
        if item.mask is None:
            raise ValueError("viaduct segmentations must carry a mask")
        if item.mask.data.shape != deck.shape:
            raise ValueError(
                f"segment mask shape {item.mask.data.shape} != frame {(height, width)}"
            )
        deck |= item.mask.data

    if not deck.any():
        return None

    masked = np.zeros_like(frame)
    cv2.copyTo(frame, deck.astype(np.uint8), masked)
    return masked
