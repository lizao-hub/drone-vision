"""Shared-memory rings for the split pipeline, and the messages that use them.

Pickling one 6 MB 1080p frame through a ``Queue`` costs ~6 ms; writing it into a
shared buffer and handing over an index costs ~0.7 ms.  With two hops per frame
that is the whole gap between a comfortable 21.8 ms stage and a blown one, so
the frame and the deck masks travel by slot index while everything else (boxes,
timings, reasons) travels as a small message.

Ownership is reference counted by the **coordinator only**: it is the single
writer of those counters, so no lock and no atomic is involved.  A slot is taken
with one reference (the coordinator's own hold while the frame is unfinished);
every worker that is given the slot holds one more reference, dropped when that
worker's result arrives.  The last release frees the slot.

Masks are stored as ``uint8`` planes and viewed as ``bool``: they are only ever
used as masks, and ``bool`` is what the domain layer wants.  ``Mask`` copies on
construction, so nothing in the coordinator ever aliases a slot the ring may
later reuse.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing import shared_memory
from uuid import uuid4

import numpy as np

from algorithm.domain.observations import DetectedObject


@dataclass(frozen=True, slots=True)
class FrameRingSpec:
    """What a worker needs to attach to the frame ring."""

    names: tuple[str, ...]
    height: int
    width: int
    channels: int = 3

    @property
    def slot_bytes(self) -> int:
        return self.height * self.width * self.channels


@dataclass(frozen=True, slots=True)
class MaskRingSpec:
    """What a worker needs to attach to the mask ring.

    ``instances`` is the per-slot capacity of *instance* masks; plane 0 of every
    slot is the union mask the detectors read.
    """

    names: tuple[str, ...]
    height: int
    width: int
    instances: int

    @property
    def planes(self) -> int:
        return 1 + self.instances

    @property
    def plane_bytes(self) -> int:
        return self.height * self.width

    @property
    def slot_bytes(self) -> int:
        return self.planes * self.plane_bytes


class FrameRing:
    """One writable frame buffer per slot, shared by every process."""

    def __init__(
        self, spec: FrameRingSpec, buffers: list[shared_memory.SharedMemory]
    ) -> None:
        if len(buffers) != len(spec.names):
            raise ValueError("frame ring spec does not match its buffers")
        self._spec = spec
        self._buffers = buffers

    @classmethod
    def allocate(
        cls, *, slots: int, height: int, width: int, channels: int = 3
    ) -> FrameRing:
        """Create a ring the caller owns and must :meth:`unlink`."""
        names = _names("frame", slots)
        spec = FrameRingSpec(names, height, width, channels)
        return cls(
            spec,
            [
                shared_memory.SharedMemory(
                    name=name, create=True, size=spec.slot_bytes
                )
                for name in names
            ],
        )

    @classmethod
    def attach(cls, spec: FrameRingSpec) -> FrameRing:
        """Map an existing ring; the caller must :meth:`close` (never unlink)."""
        return cls(spec, [shared_memory.SharedMemory(name=name) for name in spec.names])

    @property
    def spec(self) -> FrameRingSpec:
        return self._spec

    @property
    def slots(self) -> int:
        return len(self._buffers)

    def view(self, slot: int) -> np.ndarray:
        """A writable ``H x W x C`` view of the slot: no copy."""
        spec = self._spec
        # `count` is explicit because a platform may hand back a page-rounded
        # buffer that is larger than the slot (Windows does).
        return np.frombuffer(
            self._buffers[slot].buf, dtype=np.uint8, count=spec.slot_bytes
        ).reshape(spec.height, spec.width, spec.channels)

    def write(self, slot: int, frame: np.ndarray) -> None:
        """Copy ``frame`` into the slot; the only bulk write in the ring."""
        spec = self._spec
        if frame.shape != (spec.height, spec.width, spec.channels):
            raise ValueError(
                f"frame shape {frame.shape} != ring "
                f"{(spec.height, spec.width, spec.channels)}"
            )
        self.view(slot)[:] = frame

    def close(self) -> None:
        for buffer in self._buffers:
            buffer.close()

    def unlink(self) -> None:
        for buffer in self._buffers:
            buffer.unlink()


class MaskRing:
    """Per slot: one union mask plus a fixed number of instance masks."""

    def __init__(
        self, spec: MaskRingSpec, buffers: list[shared_memory.SharedMemory]
    ) -> None:
        if len(buffers) != len(spec.names):
            raise ValueError("mask ring spec does not match its buffers")
        self._spec = spec
        self._buffers = buffers

    @classmethod
    def allocate(
        cls, *, slots: int, height: int, width: int, instances: int
    ) -> MaskRing:
        """Create a ring the caller owns and must :meth:`unlink`."""
        if instances <= 0:
            raise ValueError("instances must be positive")
        names = _names("mask", slots)
        spec = MaskRingSpec(names, height, width, instances)
        return cls(
            spec,
            [
                shared_memory.SharedMemory(
                    name=name, create=True, size=spec.slot_bytes
                )
                for name in names
            ],
        )

    @classmethod
    def attach(cls, spec: MaskRingSpec) -> MaskRing:
        """Map an existing ring; the caller must :meth:`close` (never unlink)."""
        return cls(spec, [shared_memory.SharedMemory(name=name) for name in spec.names])

    @property
    def spec(self) -> MaskRingSpec:
        return self._spec

    @property
    def slots(self) -> int:
        return len(self._buffers)

    def plane(self, slot: int, index: int) -> np.ndarray:
        """A writable ``H x W`` bool view of one plane: no copy."""
        spec = self._spec
        if not 0 <= index < spec.planes:
            raise ValueError(f"mask plane {index} is outside 0..{spec.planes - 1}")
        start = index * spec.plane_bytes
        return np.frombuffer(
            self._buffers[slot].buf,
            dtype=np.uint8,
            count=spec.plane_bytes,
            offset=start,
        ).view(np.bool_).reshape(spec.height, spec.width)

    def union(self, slot: int) -> np.ndarray:
        """The mask both detectors black the frame with."""
        return self.plane(slot, 0)

    def write(
        self,
        slot: int,
        union: np.ndarray,
        instances: Sequence[np.ndarray],
    ) -> None:
        """Store the union and every instance mask, in instance order."""
        spec = self._spec
        if len(instances) > spec.instances:
            raise ValueError(
                f"{len(instances)} deck instances exceed the mask ring's "
                f"per-slot capacity of {spec.instances}"
            )
        if union.shape != (spec.height, spec.width):
            raise ValueError(
                f"union mask shape {union.shape} != ring {(spec.height, spec.width)}"
            )
        self.plane(slot, 0)[:] = union
        for index, mask in enumerate(instances):
            if mask.shape != (spec.height, spec.width):
                raise ValueError(
                    f"instance mask shape {mask.shape} != ring "
                    f"{(spec.height, spec.width)}"
                )
            self.plane(slot, index + 1)[:] = mask

    def close(self) -> None:
        for buffer in self._buffers:
            buffer.close()

    def unlink(self) -> None:
        for buffer in self._buffers:
            buffer.unlink()


class SlotPool:
    """Reference-counted slot ownership; the coordinator is the only writer."""

    def __init__(self, slots: int) -> None:
        if slots <= 0:
            raise ValueError("slots must be positive")
        self._refs = [0] * slots

    def has_free(self) -> bool:
        """Whether a slot could be taken; checked before a frame is read."""
        return any(refs == 0 for refs in self._refs)

    def take(self) -> int:
        """A free slot with one reference held by the caller."""
        for slot, refs in enumerate(self._refs):
            if refs == 0:
                self._refs[slot] = 1
                return slot
        raise RuntimeError("shared-memory ring exhausted: no free slot")

    def retain(self, slot: int) -> None:
        self._refs[slot] += 1

    def release(self, slot: int) -> None:
        self._refs[slot] -= 1
        if self._refs[slot] < 0:
            raise RuntimeError(f"slot {slot} released more often than retained")

    def refs(self, slot: int) -> int:
        return self._refs[slot]


# -- messages --------------------------------------------------------------
# Small, pickled through a Queue: integers, floats, reason strings and the
# boxes/confidences of the detections.  Masks never appear here -- they stay in
# the rings and are referenced by slot index.
@dataclass(frozen=True, slots=True)
class FrameJob:
    """One frame for one worker: which slots hold its pixels and its masks."""

    sequence: int
    timestamp_ms: int
    generation: int
    frame_slot: int
    mask_slot: int


@dataclass(frozen=True, slots=True)
class SegmentResult:
    """What the segmentation worker found, minus the masks themselves.

    ``instances`` carries the detection boxes in instance order, with
    ``mask=None``: instance ``i`` is plane ``i + 1`` of ``mask_slot``.
    """

    sequence: int
    frame_slot: int
    mask_slot: int
    segment_ms: float
    has_deck: bool = False
    instances: tuple[DetectedObject, ...] = ()
    error: str | None = None


@dataclass(frozen=True, slots=True)
class DetectResult:
    """What the detection worker found, plus which model failed, if any.

    The vehicle detections travel even when the line model failed, because the
    serial order is "track vehicles, then detect lines": the coordinator still
    has to run the vehicle tracker before it can honour the failure.
    """

    sequence: int
    frame_slot: int
    mask_slot: int
    detect_ms: float = 0.0
    line_ms: float = 0.0
    vehicle: tuple[DetectedObject, ...] = ()
    line: tuple[DetectedObject, ...] = ()
    vehicle_error: str | None = None
    line_error: str | None = None


def _names(kind: str, slots: int) -> tuple[str, ...]:
    if slots <= 0:
        raise ValueError("slots must be positive")
    # A per-run prefix: two concurrent runs must not collide in /dev/shm.
    prefix = uuid4().hex[:12]
    return tuple(f"drone-{prefix}-{kind}-{index}" for index in range(slots))
