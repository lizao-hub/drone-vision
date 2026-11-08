import numpy as np
import pytest

from algorithm.runtime.shm import (
    FrameRing,
    MaskRing,
    SlotPool,
)


def _frame(height: int = 4, width: int = 6, value: int = 7) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_a_frame_written_to_a_slot_reads_back_unchanged():
    ring = FrameRing.allocate(slots=2, height=4, width=6)
    try:
        frame = _frame()
        ring.write(1, frame)
        assert np.array_equal(ring.view(1), frame)
    finally:
        ring.unlink()


def test_slots_do_not_alias_each_other():
    ring = FrameRing.allocate(slots=2, height=4, width=6)
    try:
        ring.write(0, _frame(value=1))
        ring.write(1, _frame(value=2))
        assert int(ring.view(0)[0, 0, 0]) == 1
        assert int(ring.view(1)[0, 0, 0]) == 2
    finally:
        ring.unlink()


def test_a_frame_of_the_wrong_shape_is_rejected():
    ring = FrameRing.allocate(slots=1, height=4, width=6)
    try:
        with pytest.raises(ValueError):
            ring.write(0, np.zeros((4, 5, 3), dtype=np.uint8))
    finally:
        ring.unlink()


def test_the_union_and_every_instance_mask_round_trip_in_order():
    ring = MaskRing.allocate(slots=1, height=2, width=3, instances=3)
    try:
        union = np.array([[True, True, False], [False, True, False]])
        first = np.array([[True, False, False], [False, False, False]])
        second = np.array([[False, True, False], [False, True, False]])
        ring.write(0, union, (first, second))

        assert np.array_equal(ring.union(0), union)
        assert np.array_equal(ring.plane(0, 1), first)
        assert np.array_equal(ring.plane(0, 2), second)
        # Capacity the frame did not use stays empty, so it cannot leak into a
        # later frame that has fewer instances.
        assert not ring.plane(0, 3).any()
    finally:
        ring.unlink()


def test_more_instances_than_the_ring_holds_is_an_error():
    ring = MaskRing.allocate(slots=1, height=2, width=2, instances=1)
    try:
        mask = np.ones((2, 2), dtype=np.bool_)
        with pytest.raises(ValueError):
            ring.write(0, mask, (mask, mask))
    finally:
        ring.unlink()


def test_an_attached_ring_sees_what_the_owner_wrote():
    owner = FrameRing.allocate(slots=1, height=4, width=6)
    attached = FrameRing.attach(owner.spec)
    try:
        owner.write(0, _frame(value=9))
        assert int(attached.view(0)[3, 5, 2]) == 9
    finally:
        attached.close()
        owner.close()
        owner.unlink()


def test_a_slot_is_reused_only_once_every_reference_is_released():
    pool = SlotPool(2)
    first = pool.take()
    second = pool.take()
    assert {first, second} == {0, 1}
    assert not pool.has_free()

    # One reference for the segmentation worker, one for the detector.
    pool.retain(first)
    assert pool.refs(first) == 2
    pool.release(first)
    assert not pool.has_free()
    pool.release(first)
    assert pool.has_free()
    assert pool.take() == first


def test_taking_a_slot_nobody_released_is_an_error():
    pool = SlotPool(1)
    pool.take()
    with pytest.raises(RuntimeError):
        pool.take()


def test_releasing_a_free_slot_is_an_error():
    pool = SlotPool(1)
    pool.take()
    pool.release(0)
    with pytest.raises(RuntimeError):
        pool.release(0)
