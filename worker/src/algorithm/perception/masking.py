"""Deck-mask post-processing: union and blackout.

Pure numpy/OpenCV image operations.  No model is loaded here and nothing is
written to disk, so the whole module is cheap to unit-test.

The blackout is memory-bandwidth bound at 1080p (a 6 MB copy plus a boolean
fancy-index write), so it goes through OpenCV's SIMD path instead: the boolean
mask is widened once and `bitwise_and` writes into a caller-owned buffer.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
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


def deck_frame(
    frame: np.ndarray,
    segmentations: Sequence[DetectedObject],
    *,
    out: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    """Black out everything outside this frame's deck union.

    The one implementation of "derive the deck region, keep only the deck": the
    segment worker of the engine calls it, so what gets blacked out and what the
    detectors see cannot drift apart.  The union itself is not returned: the
    only thing the caller needs from it is whether it covered a pixel, which is
    exactly what makes it skip both detectors.

    ``out`` reuses a caller-owned buffer across frames (the result must not
    outlive the current frame); a new one is allocated when the frame geometry
    changed.  Returns the masked frame and whether the deck was empty.
    """
    if frame.ndim != 3:
        raise ValueError("frame must be an H x W x C array")
    height, width = int(frame.shape[0]), int(frame.shape[1])
    if out is not None and (out.shape != frame.shape or out.dtype != frame.dtype):
        out = None
    union = union_mask(segmentations, height=height, width=width)
    return apply_mask(frame, union, out=out), not union.data.any()


def apply_mask(
    frame: np.ndarray, mask: Mask, *, out: np.ndarray | None = None
) -> np.ndarray:
    """Copy the frame and blacken everything outside the deck mask.

    Returns a new array unless ``out`` is given, in which case it is filled and
    returned.  Callers may reuse one buffer per frame size (the pipeline does);
    the result must therefore not outlive the current frame.
    """
    if frame.ndim != 3:
        raise ValueError("frame must be an H x W x C array")
    if frame.shape[:2] != (mask.height, mask.width):
        raise ValueError(
            f"mask geometry {(mask.height, mask.width)} != frame {frame.shape[:2]}"
        )
    if out is None:
        masked = np.empty_like(frame)
    else:
        if out.shape != frame.shape or out.dtype != frame.dtype:
            raise ValueError(f"out must have shape {frame.shape} and dtype {frame.dtype}")
        masked = out
    # `copyTo` leaves the destination untouched where the mask is zero, so the
    # buffer is cleared first: at 1080p that is ~0.5 ms, versus ~25-35 ms for a
    # boolean fancy-index blackout.  The mask must be 8-bit; widening 2 M
    # booleans costs ~1 ms.
    masked.fill(0)
    cv2.copyTo(frame, mask.data.astype(np.uint8), masked)
    return masked

