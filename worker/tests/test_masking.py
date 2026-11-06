import numpy as np
import pytest

from algorithm.domain.geometry import Box, Mask
from algorithm.domain.observations import DetectedObject
from algorithm.perception.masking import apply_mask, deck_frame, union_mask


def _detection(mask: np.ndarray, confidence: float = 0.9) -> DetectedObject:
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    box = Box(
        float(columns[0]) / mask.shape[1],
        float(rows[0]) / mask.shape[0],
        float(columns[-1] + 1) / mask.shape[1],
        float(rows[-1] + 1) / mask.shape[0],
    )
    return DetectedObject(label="viaduct", confidence=confidence, box=box, mask=Mask(mask))


def test_union_merges_every_instance():
    left = np.zeros((4, 6), dtype=np.bool_)
    left[:, :3] = True
    right = np.zeros((4, 6), dtype=np.bool_)
    right[:, 3:] = True
    merged = union_mask((_detection(left), _detection(right)), height=4, width=6)
    assert merged.data.all()
    # Taking only the most confident instance would black out half the road.


def test_union_does_not_filter_by_confidence():
    # Filtering happens once in the pipeline; the union must not drop again.
    mask = np.zeros((4, 6), dtype=np.bool_)
    mask[1, 1] = True
    merged = union_mask((_detection(mask, confidence=0.01),), height=4, width=6)
    assert merged.data.sum() == 1


def test_union_of_no_instances_is_empty():
    assert not union_mask((), height=4, width=6).data.any()


def test_union_rejects_geometry_mismatch():
    mask = np.zeros((3, 6), dtype=np.bool_)
    mask[1, 1] = True
    with pytest.raises(ValueError):
        union_mask((_detection(mask),), height=4, width=6)


def test_union_requires_a_mask_on_every_instance():
    without_mask = DetectedObject(
        label="viaduct", confidence=0.9, box=Box(0.1, 0.1, 0.2, 0.2)
    )
    with pytest.raises(ValueError):
        union_mask((without_mask,), height=4, width=6)


def test_deck_frame_returns_the_masked_frame_and_whether_the_deck_is_empty():
    frame = np.full((4, 6, 3), 180, dtype=np.uint8)
    deck = np.zeros((4, 6), dtype=np.bool_)
    deck[:, :3] = True

    masked, empty = deck_frame(frame, (_detection(deck),))

    assert not empty
    assert (masked[:, :3] == 180).all()
    assert (masked[:, 3:] == 0).all()

    # "Empty" means the union covers no pixel, not that the tuple is empty: an
    # instance with a blank mask still tells the caller to skip both detectors.
    blank = DetectedObject(
        label="viaduct",
        confidence=0.9,
        box=Box(0.0, 0.1, 0.2, 0.2),
        mask=Mask(np.zeros((4, 6), dtype=np.bool_)),
    )
    masked, empty = deck_frame(frame, (blank,))

    assert empty
    assert not masked.any()


def test_apply_mask_blacks_out_everything_off_the_deck():
    mask = Mask(np.array([[True, False, False], [True, True, False]], dtype=np.bool_))
    frame = np.full((2, 3, 3), 255, dtype=np.uint8)
    masked = apply_mask(frame, mask)
    assert (masked[mask.data] == 255).all()
    assert (masked[~mask.data] == 0).all()
    assert (frame == 255).all()


def test_apply_mask_fills_a_reused_buffer_in_place():
    mask = Mask(np.array([[True, False], [False, False]], dtype=np.bool_))
    frame = np.full((2, 2, 3), 200, dtype=np.uint8)
    buffer = np.empty_like(frame)

    masked = apply_mask(frame, mask, out=buffer)

    assert masked is buffer
    assert int(masked[0, 0, 0]) == 200
    assert int(masked[1, 1, 2]) == 0
    assert int(frame[1, 1, 2]) == 200  # the source frame is never touched


def test_apply_mask_rejects_a_buffer_of_the_wrong_shape():
    mask = Mask(np.ones((2, 2), dtype=np.bool_))
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        apply_mask(frame, mask, out=np.empty((3, 3, 3), dtype=np.uint8))
