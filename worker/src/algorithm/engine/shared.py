"""Shared-memory slots and queue messages for the multi-process pipeline.

One :class:`Slot` exists per in-flight frame.  A slot owns three buffers::

    frame    H x W x 3 uint8   written by the parent, read by a segment worker
    masked   H x W x 3 uint8   written by that segment worker, read by the two
                               detector workers
    masks    C x H x W uint8   the instances' masks, written by the segment
                               worker, copied out by the parent

Only the parent creates and unlinks shared memory; workers attach by name, so a
1080p frame crosses a process boundary as one ``memcpy`` instead of a six
megabyte pickle through a pipe.  Every process must agree on ``H``/``W``/``C``,
which is why the geometry is fixed for the lifetime of a pipeline.

The messages below are ``NamedTuple``s -- small, picklable, and importable from
both sides (both processes import this module, so no dynamic class lookups).
"""

from __future__ import annotations

from multiprocessing.shared_memory import SharedMemory
from typing import NamedTuple

import numpy as np

FRAME_CHANNELS = 3

# Worker roles.
SEGMENT = "segment"
VEHICLE = "vehicle"
LINE = "line"
DETECTORS = (VEHICLE, LINE)


def frame_nbytes(height: int, width: int) -> int:
    return height * width * FRAME_CHANNELS


def masks_nbytes(height: int, width: int, capacity: int) -> int:
    return capacity * height * width


def frame_view(buffer: object, height: int, width: int) -> np.ndarray:
    """A writable ``H x W x 3`` view over a slot's frame or masked buffer."""
    return np.ndarray(
        (height, width, FRAME_CHANNELS), dtype=np.uint8, buffer=buffer
    )


def mask_view(buffer: object, index: int, height: int, width: int) -> np.ndarray:
    """A writable ``H x W`` view over instance ``index``'s mask bytes (0/1)."""
    size = height * width
    block = np.frombuffer(buffer, dtype=np.uint8, count=(index + 1) * size)
    return block[index * size :].reshape(height, width)


class Slot:
    """One in-flight frame's shared buffers, owned (and unlinked) by the parent."""

    def __init__(self, *, height: int, width: int, mask_capacity: int) -> None:
        if height <= 0 or width <= 0:
            raise ValueError("slot geometry must be positive")
        if mask_capacity < 1:
            raise ValueError("mask_capacity must be positive")
        self.height = height
        self.width = width
        self.mask_capacity = mask_capacity
        self.frame = SharedMemory(create=True, size=frame_nbytes(height, width))
        self.masked = SharedMemory(create=True, size=frame_nbytes(height, width))
        self.masks = SharedMemory(
            create=True, size=masks_nbytes(height, width, mask_capacity)
        )

    @property
    def names(self) -> tuple[str, str, str]:
        """The three shared-memory names, in ``(frame, masked, masks)`` order."""
        return (self.frame.name, self.masked.name, self.masks.name)

    def close_unlink(self) -> None:
        for segment in (self.frame, self.masked, self.masks):
            try:
                segment.close()
            except Exception:  # noqa: BLE001 - shutdown must not mask the cause
                pass
            try:
                segment.unlink()
            except Exception:  # noqa: BLE001
                pass


class WorkerReady(NamedTuple):
    """A worker finished loading its model and is serving its queue."""

    role: str


class WorkerFailed(NamedTuple):
    """A worker could not start (missing weights, no GPU, wrong model task)."""

    role: str
    traceback: str


class SegmentDone(NamedTuple):
    """One frame segmented: instance boxes/confidences, plus the slot it filled.

    Masks are *not* in the message; they are in the slot's ``masks`` buffer at
    indices ``0 .. count-1``.  ``error`` is a short repr of the adapter failure
    (the frame is then invalid and detectors never ran); ``empty`` means the
    union of the kept instances has no pixel, so the detectors were not asked.
    """

    seq: int
    slot: int
    count: int
    boxes: tuple[tuple[float, float, float, float], ...]
    confidences: tuple[float, ...]
    segment_ms: float
    empty: bool
    error: str | None


class DetectionDone(NamedTuple):
    """One detector finished on a frame's masked image."""

    role: str
    seq: int
    slot: int
    detections: tuple[object, ...]  # domain.DetectedObject values
    elapsed_ms: float
    error: str | None
